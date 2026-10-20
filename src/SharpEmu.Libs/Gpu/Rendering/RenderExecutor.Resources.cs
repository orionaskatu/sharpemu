// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.VideoOut;
using Silk.NET.Vulkan;

namespace SharpEmu.Libs.Gpu.Rendering;

public sealed partial class RenderExecutor
{
    private const uint TransientIndexAlignment = 16;

    private struct VertexBufferRange
    {
        public ulong BaseAddress;
        public ulong RequestedEnd;
        public ulong AcquiredEnd;
        public BufferBinding Binding;

        public readonly ulong RequestedSize => RequestedEnd - BaseAddress;
    }

    private readonly record struct PreparedIndexBuffer(BufferBinding Binding, ulong Size, IndexType Type);

    private void PrepareDrawBufferAllocations(VertexInputInfo vertexInput, in IndexSource index, ulong indirectAddress)
    {
        if (vertexInput.Buffers.Length > VertexInputInfo.MaxBuffers)
            throw _host.Fatal($"The vertex input has too many buffers: count={vertexInput.Buffers.Length} max={VertexInputInfo.MaxBuffers}.");
        Span<GuestSpan> ranges = stackalloc GuestSpan[VertexInputInfo.MaxBuffers + 2];
        var count = 0;
        foreach (var vertex in vertexInput.Buffers)
        {
            if (vertex.Size == 0) continue;
            if (vertex.Address == 0 || vertex.Size > ulong.MaxValue - vertex.Address)
                throw _host.Fatal($"The vertex buffer range is invalid: address=0x{vertex.Address:X16} size=0x{vertex.Size:X16}.");
            ranges[count++] = new GuestSpan(vertex.Address, _host.ClampMappedSize(vertex.Address, vertex.Size));
        }
        if (index.Enabled && index.HostData is null)
            ranges[count++] = new GuestSpan(index.Address, index.Size);
        if (indirectAddress != 0)
            ranges[count++] = new GuestSpan(indirectAddress, IndexedIndirectArgumentsSize);
        _host.PrepareBufferAllocations(ranges[..count]);
    }

    // Merges the vertex ranges, obtains one host buffer per merged range and offsets every slot into it.
    private BufferBinding[] AcquireVertexBuffers(VertexInputInfo vertexInput)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawVertexBufferAcquisition);
        var buffers = vertexInput.Buffers;
        if (buffers.Length > VertexInputInfo.MaxBuffers)
        {
            throw _host.Fatal($"The vertex input has too many buffers: count={buffers.Length} max={VertexInputInfo.MaxBuffers}.");
        }

        // Sized to this draw's buffers: a MaxBuffers span is over a kilobyte, zeroed on every draw.
        Span<VertexBufferRange> ranges = stackalloc VertexBufferRange[buffers.Length];
        Span<ulong> sizes = stackalloc ulong[buffers.Length];
        for (var slot = 0; slot < buffers.Length; slot++) sizes[slot] = buffers[slot].Size;
        foreach (var attribute in vertexInput.Attributes)
        {
            var slot = attribute.BufferIndex;
            var descriptor = attribute.Descriptor;
            if (buffers[slot].Stride != 0 || descriptor.RecordCount == 0 || descriptor.OutOfBounds != 2) continue;
            // This bounds mode tests only whether the record count is nonzero.
            // Constant attributes still need the full format extent.
            sizes[slot] = Math.Max(sizes[slot], (ulong)attribute.OffsetBytes + VertexAttributeFormats.StorageByteSize(descriptor));
        }
        var rangeCount = 0;
        for (var slot = 0; slot < buffers.Length; slot++)
        {
            ref readonly var vertex = ref buffers[slot];
            var size = sizes[slot];
            if (size == 0)
            {
                continue;
            }

            if (vertex.Address == 0 || size > ulong.MaxValue - vertex.Address)
            {
                throw _host.Fatal($"The vertex buffer range is invalid: address=0x{vertex.Address:X16} size=0x{size:X16}.");
            }

            ranges[rangeCount++] = new VertexBufferRange { BaseAddress = vertex.Address, RequestedEnd = vertex.Address + size };
        }

        ranges[..rangeCount].Sort(static (left, right) => left.BaseAddress.CompareTo(right.BaseAddress));
        Span<VertexBufferRange> merged = stackalloc VertexBufferRange[buffers.Length];
        var mergedCount = 0;
        for (var i = 0; i < rangeCount; i++)
        {
            ref readonly var range = ref ranges[i];
            if (mergedCount != 0 && merged[mergedCount - 1].RequestedEnd >= range.BaseAddress)
            {
                merged[mergedCount - 1].RequestedEnd = Math.Max(merged[mergedCount - 1].RequestedEnd, range.RequestedEnd);
                continue;
            }

            merged[mergedCount++] = new VertexBufferRange { BaseAddress = range.BaseAddress, RequestedEnd = range.RequestedEnd };
        }

        for (var i = 0; i < mergedCount; i++)
        {
            ref var range = ref merged[i];
            var size = _host.ClampMappedSize(range.BaseAddress, range.RequestedSize);
            range.AcquiredEnd = range.BaseAddress + size;
            range.Binding = _host.ObtainBuffer(range.BaseAddress, size, isWritten: false);
        }

        var prepared = new BufferBinding[buffers.Length];
        BufferBinding? nullBuffer = null;
        Span<ulong> acquiredBytes = stackalloc ulong[buffers.Length];
        acquiredBytes.Clear();
        for (var slot = 0; slot < buffers.Length; slot++)
        {
            ref readonly var vertex = ref buffers[slot];
            if (sizes[slot] == 0)
            {
                nullBuffer ??= _host.NullBuffer;
                prepared[slot] = nullBuffer.Value;
                continue;
            }

            var found = -1;
            for (var i = 0; i < mergedCount; i++)
            {
                if (vertex.Address >= merged[i].BaseAddress && vertex.Address < merged[i].AcquiredEnd)
                {
                    found = i;
                    break;
                }
            }

            if (found < 0)
            {
                throw _host.Fatal($"The vertex buffer address is outside the acquired range: address=0x{vertex.Address:X16}.");
            }

            ref readonly var owner = ref merged[found];
            prepared[slot] = new BufferBinding(owner.Binding.Handle, owner.Binding.Offset + vertex.Address - owner.BaseAddress);
            acquiredBytes[slot] = owner.AcquiredEnd - vertex.Address;
        }

        foreach (var attribute in vertexInput.Attributes)
        {
            var slot = attribute.BufferIndex;
            _host.TraceVertexBuffer(vertexInput.Stage.Program?.Hash ?? 0, attribute, buffers[slot],
                prepared[slot], sizes[slot], acquiredBytes[slot]);
        }

        return prepared;
    }

    private PreparedIndexBuffer AcquireIndexBuffer(in IndexSource source)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawIndexBufferAcquisition);
        if (!source.Enabled)
        {
            return default;
        }

        if (source.Size == 0)
        {
            throw _host.Fatal($"The index buffer is empty: address=0x{source.Address:X16} type={(int)source.Type}.");
        }

        var binding = source.HostData is { } hostData
            ? _host.UploadTransient(hostData, TransientIndexAlignment)
            : _host.ObtainBuffer(source.Address, source.Size, isWritten: false);
        return new PreparedIndexBuffer(binding, source.Size, source.Type);
    }

    // A written buffer resource with an address and a footprint needs a barrier after the stage.
    // Device-address accesses are not split into reads and writes, so they count as stores.
    private bool DrawWritesMemory(ShaderStageResources stage) =>
        stage.Program is { } program &&
        (program.UsesDeviceAddresses || WritesStorageImage(program) || HasBufferWrites(stage));

    private bool HasBufferWrites(ShaderStageResources stage)
    {
        var program = stage.Program ?? throw _host.Fatal("A shader stage has no program.");
        var descriptors = stage.Resources.Buffers;
        if (descriptors.Length != program.Buffers.Length)
        {
            throw _host.Fatal($"The buffer descriptor count does not match the program: descriptors={descriptors.Length} program={program.Buffers.Length}.");
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            if (descriptors[i].Length < 4)
            {
                throw _host.Fatal($"A written buffer descriptor is too short: index={i} words={descriptors[i].Length}.");
            }

            var descriptor = BufferDescriptorWords.From(descriptors[i]);
            var footprint = descriptor.Footprint() ?? throw _host.Fatal(
                $"The written buffer footprint overflows: index={i} stride={descriptor.Stride} records={descriptor.RecordCount}.");
            if (descriptor.Address != 0 && footprint != 0)
            {
                return true;
            }
        }

        return false;
    }

    private void SetDrawDebugPhase(ulong submitId, in DrawCall draw, uint phase) =>
        _host.SetDebugInformation(draw.Operation, submitId, phase, draw.Count, 0, draw.InstanceCount, draw.FirstInstance);

    private IPreparedBindings PrepareBindings(in ShaderStageResources stage)
    {
        var bindings = _host.PrepareBindings(stage);
        NoteSampledTextures(stage);
        return bindings;
    }

    private static void NoteSampledTextures(in ShaderStageResources stage)
    {
        if (stage.Program is not { } program)
        {
            return;
        }

        var descriptors = stage.Resources.Images;
        for (var index = 0; index < program.Images.Length && index < descriptors.Length; index++)
        {
            if (program.Images[index].Class == ImageResourceClass.Sampled)
            {
                MipStatistics.Shared.NoteSampled(new TextureDescriptorWords(descriptors[index]));
            }
        }
    }

    private static int _dbgGrassLogs; // TEMP
    private static int _dbgHogLogs; // TEMP

    // Binds everything the draw needs inside one preparation scope, then records it.
    private string RecordDraw(
        ulong submitId,
        RegisterBanks banks,
        in DrawCall draw,
        ref DrawState state,
        PrimitiveTopology topology,
        in DrawEmission emission,
        in IndexSource indexSource,
        bool primitiveRestart,
        bool setBindDebug,
        bool setAutoDebug,
        IndexedDrawTrace.Sample? indexedTrace = null,
        ulong packetAddress = 0)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawResourcePreparation);
        var context = banks.Context;
        var vertexInput = state.Programs.VertexInput;
        if (DbgSkipVertexHashes.Count != 0 && DbgSkipVertexHashes.Contains(vertexInput.Stage.Program?.Hash ?? 0)) // TEMP
        {
            return "dbg-skip";
        }

        Diagnostics.DbgSequence.Tally("DR", vertexInput.Stage.Program?.Hash ?? 0); // TEMP
        if (DbgInstanceCheck.Count != 0 && (DbgInstanceCheck.Contains(0) || DbgInstanceCheck.Contains(vertexInput.Stage.Program?.Hash ?? 0)) && // TEMP
            vertexInput.Stage.Program is not null && System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds > DbgVrefAfter &&
            DbgVrefSeen.GetValueOrDefault(vertexInput.Stage.Program.Hash) is var seen && seen < DbgVrefMax && (DbgVrefSeen[vertexInput.Stage.Program.Hash] = seen + 1) > 0)
            DbgReferenceVertices(vertexInput.Stage.Program!.Hash, vertexInput.Stage.Program!.UserDataBase, vertexInput.Stage.Resources.UserData, draw, emission, indexSource);
        if (DbgProbeOn) // TEMP
        {
            if (emission.IndirectArgumentsAddress != 0)
            {
                var probeBytes = new byte[20];
                if (_host.TryReadGuest(emission.IndirectArgumentsAddress, probeBytes))
                {
                    var probeWords = new uint[5];
                    System.Buffer.BlockCopy(probeBytes, 0, probeWords, 0, 20);
                    DbgProbeDraw("IND", probeWords[0], emission.Indexed ? probeWords[1] : probeWords[1]);
                }
            }
            else
            {
                DbgProbeDraw("DIR", draw.Count, draw.InstanceCount);
            DbgProbeDraw($"VS{vertexInput.Stage.Program?.Hash:X16}/{draw.Name}", draw.Count, draw.InstanceCount, true);
            }
        }
        if (!emission.Indexed && (draw.Count == 16293 || draw.Count == 216 || draw.Count == 2898) && System.Threading.Interlocked.Increment(ref _dbgHogLogs) % 25 == 1) // TEMP
            Console.Error.WriteLine($"[DBG][HOG] vs=0x{vertexInput.Stage.Program?.Hash:X} ps=0x{state.Programs.PixelInput.Stage.Program?.Hash:X} pixelActive={state.PixelActive} colors={state.ColorCount} depth={state.Depth.HasTarget} count={draw.Count} indexed={emission.Indexed} instances={draw.InstanceCount} marker={Diagnostics.DbgSequence.Marker}");
        if (emission.IndirectArgumentsAddress != 0 && Diagnostics.DbgSequence.Marker.Contains("Grass") && System.Threading.Interlocked.Increment(ref _dbgGrassLogs) <= 120) // TEMP
        {
            var dbgBytes = new byte[20];
            var dbgOk = _host.TryReadGuest(emission.IndirectArgumentsAddress, dbgBytes);
            Console.Error.WriteLine($"[DBG][GRASSARGS] t={System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds:F1} marker={Diagnostics.DbgSequence.Marker} indexed={emission.Indexed} addr=0x{emission.IndirectArgumentsAddress:X} vs=0x{vertexInput.Stage.Program?.Hash ?? 0:X} read={dbgOk} words={string.Join(",", Enumerable.Range(0, 5).Select(i => BitConverter.ToUInt32(dbgBytes, i * 4).ToString("X")))}");
        }
        if (Diagnostics.DbgSequence.SkipByMarker) // TEMP
        {
            return "dbg-skip";
        }

        if (DbgBisectSkip(vertexInput.Stage.Program?.Hash ?? 0, state.PixelActive)) // TEMP
        {
            return "dbg-skip";
        }
        var pixelInput = state.Programs.PixelInput;
        var meshInput = state.Programs.MeshInput;
        using var preparation = _host.BeginPreparation();
        using var indirectMesh = meshInput is not null && emission.IndirectArgumentsAddress != 0
            ? _host.PrepareIndirectMeshDraw(new IndirectMeshDrawRequest(emission.IndirectArgumentsAddress,
                indexSource.Address, draw.Count, indexSource.Type == IndexType.Uint16 ? 2u : 4u,
                topology == PrimitiveTopology.PointList, meshInput.Geometry.InputPrimitiveCountPerWorkgroup, meshInput.Execution))
            : null;
        var meshVertexCount = draw.Count;
        if (MeshDrawTrace.Active)
            MeshDrawTrace.Write("programs", $"mesh={meshInput is not null} vertexHash=0x{vertexInput.Stage.Program?.Hash:X16} pixelHash=0x{pixelInput.Stage.Program?.Hash:X16} pixelActive={state.PixelActive}");
        if (meshInput is not null)
        {
            if (topology is not (PrimitiveTopology.PointList or PrimitiveTopology.LineList or PrimitiveTopology.TriangleList or PrimitiveTopology.TriangleStrip))
            {
                throw _host.Fatal("The mesh draw input topology is not supported.");
            }

            ulong indexAddress = 0;
            if (emission.Indexed && indirectMesh is null)
            {
                var byteCount = checked((int)indexSource.Size);
                var elementSize = indexSource.Type == IndexType.Uint16 ? 2u : 4u;
                var triangleStrip = topology == PrimitiveTopology.TriangleStrip;
                var lineList = topology == PrimitiveTopology.LineList;
                var pointList = lineList || topology == PrimitiveTopology.PointList;
                var capacity = pointList ? byteCount / (int)elementSize
                    : MeshIndexAssembly.GetMaximumCount(byteCount, elementSize, triangleStrip);
                if (capacity == 0) return "no-primitives";
                byte[]? rentedBytes = null;
                uint[]? rentedIndices = null;
                try
                {
                    var bytes = indexSource.HostData is { } hostData ? hostData.AsSpan(0, byteCount)
                        : (rentedBytes = ArrayPool<byte>.Shared.Rent(byteCount)).AsSpan(0, byteCount);
                    if (indexSource.HostData is null && !_host.TryReadGuest(indexSource.Address, bytes))
                        throw _host.Fatal("The mesh index buffer cannot be read.");
                    rentedIndices = ArrayPool<uint>.Shared.Rent(capacity);
                    var written = pointList
                        ? MeshIndexAssembly.ExpandPoints(bytes, rentedIndices.AsSpan(0, capacity), elementSize,
                            emission.VertexOffset, primitiveRestart, elementSize == 2 ? ushort.MaxValue : uint.MaxValue)
                        : MeshIndexAssembly.ExpandTriangles(bytes, rentedIndices.AsSpan(0, capacity),
                        elementSize, emission.VertexOffset, triangleStrip, primitiveRestart,
                        elementSize == 2 ? ushort.MaxValue : uint.MaxValue, context.RasterMode.ProvokingVertexLast);
                    if (lineList) written &= ~1;
                    meshVertexCount = checked((uint)written);
                    if (written == 0) return "no-primitives";
                    indexAddress = _host.UploadMeshVertexIndices(rentedIndices.AsSpan(0, written));
                }
                finally
                {
                    if (rentedIndices is not null) ArrayPool<uint>.Shared.Return(rentedIndices);
                    if (rentedBytes is not null) ArrayPool<byte>.Shared.Return(rentedBytes);
                }
            }
            vertexInput.Stage = vertexInput.Stage with
            {
                MeshDraw = indirectMesh is null
                    ? new MeshDrawParameters(meshVertexCount, emission.Indexed ? 0 : emission.FirstVertex,
                        emission.FirstInstance, emission.Indexed ? 4u : 0u, indexAddress)
                    : new MeshDrawParameters(0, 0, 0, uint.MaxValue, indirectMesh.ParametersAddress),
            };
        }
        IPreparedBindings vertexBindings;
        IPreparedBindings? pixelBindings;
        try
        {
            // The pixel program reads its position in host texels; the attachments say how
            // many of those one guest pixel covers.
            var attachmentScale = AttachmentRenderScale(in state);
            vertexBindings = PrepareBindings(vertexInput.Stage with { AttachmentRenderScale = attachmentScale });
            pixelBindings = state.PixelActive
                ? PrepareBindings(pixelInput.Stage with { AttachmentRenderScale = attachmentScale, ObjectId = banks.UserConfig.ObjectId })
                : null;
        }
        catch (DrawImageTypeMismatchException rejection)
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("resource-rejected", rejection.Message);
            IndexedDrawTrace.Write(indexedTrace, "resource-rejected", submitId, packetAddress);
            if (_strictDrawResources)
            {
                throw _host.Fatal(rejection.Message);
            }

            if (_reportedDrawImageTypeMismatches.Add(rejection.WarningKey))
            {
                Console.Error.WriteLine($"[GPU][WARN][DRAW_SKIPPED] {rejection.Message} " +
                    "The draw was not executed. Images and FPS can be incorrect. Set SHARPEMU_STRICT_COMPUTE=1 to stop on this failure.");
            }

            return "resource-rejected";
        }
        var vertexProgram = vertexInput.Stage.Program ?? throw _host.Fatal("The vertex stage has no program.");
        var pixelProgram = pixelBindings is null ? null : pixelInput.Stage.Program ?? throw _host.Fatal("The pixel stage has no program.");
        DropUnwrittenColorTargets(context, ref state, pixelProgram);
        state.Rendering = AcquireAttachments(ref state, context);
        // Attachment uploads can submit work. Finish every allocation merge before any
        // shader descriptor, vertex binding or index binding takes a buffer handle.
        PrepareDrawBufferAllocations(vertexInput, in indexSource, emission.IndirectArgumentsAddress);
        if (vertexProgram.UsesDeviceAddresses || (pixelProgram?.UsesDeviceAddresses ?? false))
        {
            _host.PrepareDeviceAddresses();
        }

        _host.BindResources(vertexBindings);
        if (pixelBindings is not null)
        {
            _host.BindResources(pixelBindings);
        }

        var vertexBuffers = meshInput is null ? AcquireVertexBuffers(vertexInput) : [];
        var indexBuffer = meshInput is null ? AcquireIndexBuffer(in indexSource) : default;
        var indirectArguments = meshInput is null && emission.IndirectArgumentsAddress != 0
            ? _host.ObtainBuffer(emission.IndirectArgumentsAddress, IndexedIndirectArgumentsSize, isWritten: false)
            : default;
        // Nothing after the pipeline touches guest memory.
        var pipeline = _pipelines.CreateGraphicsPipeline(
            BoundColors(ref state),
            in state.Depth,
            vertexInput,
            state.PixelActive ? pixelInput : null,
            context,
            in state.Rendering,
            topology,
            primitiveRestart,
            state.Programs.DisableBlending,
            state.Programs.Vertex,
            state.Programs.Pixel);
        if (setBindDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x100);
        }

        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x200);
        }

        if (meshInput is null)
        {
            _host.BindVertexBuffers(vertexBuffers, vertexInput);
        }

        if (pixelBindings is not null && setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x300);
        }

        Span<IPreparedBindings> stages = pixelBindings is null ? [vertexBindings] : [vertexBindings, pixelBindings];
        _host.CommitBindings(PipelineBindPoint.Graphics, in pipeline, stages);
        if (indexBuffer.Size != 0)
        {
            _host.BindIndexBuffer(indexBuffer.Binding, indexBuffer.Type);
        }

        _host.SetDynamicState(BuildDynamicState(context, in state));
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x400);
        }

        if (DbgClampInstances is { } clamp && emission.IndirectArgumentsAddress != 0 && !emission.Indexed && (vertexInput.Stage.Program?.Hash ?? 0) == clamp.Hash) // TEMP
        {
            _host.EndRendering();
            _host.DebugFillBuffer(emission.IndirectArgumentsAddress + 4, 4, clamp.Count);
        }

        if (DrawWritesMemory(vertexInput.Stage) || (pixelBindings is not null && DrawWritesMemory(pixelInput.Stage)))
        {
            _host.PrepareMemoryWritingDraw();
        }

        var imageSnapshot = _host.BeginImageSnapshot(pixelBindings);
        indirectMesh?.Prepare();
        _host.BeginRendering(in state.Rendering);
        if (MeshDrawTrace.Active) MeshDrawTrace.Write("rendering", $"width={state.Rendering.Width} height={state.Rendering.Height} layers={state.Rendering.Layers} colors={state.ColorCount}");
        _host.BindPipeline(PipelineBindPoint.Graphics, in pipeline);
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x500);
        }

        if (indexedTrace is { } trace)
            _host.InsertDrawTraceMarker(IndexedDrawTrace.CreateLabel(trace, submitId, packetAddress));

        if (indirectMesh is not null)
        {
            indirectMesh.Draw();
            if (MeshDrawTrace.Active) MeshDrawTrace.Recorded();
        }
        else if (meshInput is { } mesh)
        {
            var primitiveCount = mesh.Geometry.InputPointList ? meshVertexCount
                : mesh.Geometry.InputLineList ? meshVertexCount / 2
                : mesh.Geometry.InputTriangleStrip ? (meshVertexCount >= 3 ? meshVertexCount - 2 : 0) : meshVertexCount / 3;
            var groupCount = (uint)(((ulong)primitiveCount + mesh.Geometry.InputPrimitiveCountPerWorkgroup - 1) / mesh.Geometry.InputPrimitiveCountPerWorkgroup);
            var totalGroupCount = (ulong)groupCount * draw.InstanceCount;
            if (groupCount > mesh.Execution.MaxGroupCountX || draw.InstanceCount > mesh.Execution.MaxGroupCountY ||
                totalGroupCount > mesh.Execution.MaxGroupTotalCount)
            {
                throw _host.Fatal($"The mesh draw exceeds device group limits: groups={groupCount}/{mesh.Execution.MaxGroupCountX} " +
                    $"instances={draw.InstanceCount}/{mesh.Execution.MaxGroupCountY} " +
                    $"total={totalGroupCount}/{mesh.Execution.MaxGroupTotalCount}.");
            }
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"MeshDraw shader=0x{banks.Shader.Vertex.ExportAddress:X16} " +
                    $"vertices={draw.Count} instances={draw.InstanceCount} groups={groupCount} " +
                    $"threads={mesh.Geometry.ThreadsPerGroup} ldsDwords={mesh.Geometry.LocalDataShareDwords}");
            }
            _host.DrawMeshTasks(groupCount, draw.InstanceCount, 1);
            if (MeshDrawTrace.Active)
            {
                MeshDrawTrace.Write("command-recorded", $"groups={groupCount}/{draw.InstanceCount}/1 threads={mesh.Geometry.ThreadsPerGroup} wave={mesh.Geometry.WaveSize} lds={mesh.Geometry.LocalDataShareDwords} firstVertex={emission.FirstVertex} firstInstance={emission.FirstInstance}");
                MeshDrawTrace.Recorded();
            }
        }
        else if (emission.IndirectArgumentsAddress != 0)
        {
            // Uploads and shader writes end with barriers to all commands, so the
            // indirect read sees them.
            if (emission.Indexed)
            {
                _host.DrawIndexedIndirect(indirectArguments);
            }
            else
            {
                _host.DrawIndirect(indirectArguments);
            }
        }
        else
        {
            EmitDraw(banks.UserConfig, vertexInput, in draw, in emission);
        }
        _host.EndImageSnapshot(imageSnapshot, BoundColors(ref state));
        IndexedDrawTrace.Write(indexedTrace, "emitted", submitId, packetAddress);
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x600);
        }

        var writeStages = PipelineStageFlags.None;
        if (HasBufferWrites(vertexInput.Stage))
        {
            writeStages |= meshInput is null ? PipelineStageFlags.VertexShaderBit : PipelineStageFlags.MeshShaderBitExt;
        }

        if (state.PixelActive && HasBufferWrites(pixelInput.Stage))
        {
            writeStages |= PipelineStageFlags.FragmentShaderBit;
        }

        if (writeStages != PipelineStageFlags.None)
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("write-barrier", $"sourceStages={writeStages}");
            _host.EndRendering();
            _host.ShaderWriteBarrier(writeStages);
        }

        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x700);
        }

        return "recorded";
    }

    // Ghost of Yotei renders skin with screen-space subsurface scattering: the character meshes are redrawn into an
    // irradiance buffer, blurred, and the composite pass replaces the deferred result of those pixels with the blurred
    // irradiance alone. The blurred buffer carries no albedo (the albedo multiply is not reproduced yet), so skin
    // turns white-gray up close. Skipping the passes keeps the deferred skin shading (what distant skin uses).
    // SHARPEMU_SKIP_PS=off keeps them; a hash list overrides the default.
    // TEMP: SHARPEMU_DBG_CLAMP_INST=vsHash:count clamps the instance count of that vertex shader's non-indexed indirect draws.
    private static readonly (ulong Hash, uint Count)? DbgClampInstances =
        Environment.GetEnvironmentVariable("SHARPEMU_DBG_CLAMP_INST") is { Length: > 0 } dbgClampText && dbgClampText.Split(':') is { Length: 2 } dbgClampParts
            ? (Convert.ToUInt64(dbgClampParts[0].Replace("0x", ""), 16), Convert.ToUInt32(dbgClampParts[1]))
            : null;

    // TEMP: SHARPEMU_DBG_VS_BISECT=start:len:group counts draws per vertex shader before `start` seconds, then each window of
    // `len` seconds skips the next `group` vertex shaders (ranked by draw count) to find which one draws the garbage geometry.
    private static readonly double[]? DbgBisect = Environment.GetEnvironmentVariable("SHARPEMU_DBG_VS_BISECT") is { Length: > 0 } dbgBs
        ? dbgBs.Split(':').Select(part => double.Parse(part, System.Globalization.CultureInfo.InvariantCulture)).ToArray() : null;
    private static readonly Dictionary<ulong, long> DbgBisectCounts = new();
    private static ulong[]? DbgBisectRanked;
    private static int DbgBisectWindowLogged = -1;
    private static bool DbgBisectSkip(ulong vertexHash, bool pixelActive)
    {
        if (DbgBisect is not { Length: 3 } cfg || vertexHash == 0 || !pixelActive) return false;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(DbgNoSkipStart).TotalSeconds;
        if (elapsed < cfg[0])
        {
            DbgBisectCounts[vertexHash] = DbgBisectCounts.GetValueOrDefault(vertexHash) + 1;
            return false;
        }

        DbgBisectRanked ??= DbgBisectCounts.OrderByDescending(pair => pair.Value).Select(pair => pair.Key).ToArray();
        var window = (int)((elapsed - cfg[0]) / cfg[1]);
        var group = (int)cfg[2];
        if (window != DbgBisectWindowLogged)
        {
            DbgBisectWindowLogged = window;
            var skipped = DbgBisectRanked.Skip(window * group).Take(group).Select(hash => $"0x{hash:X16}({DbgBisectCounts[hash]})");
            Console.Error.WriteLine($"[DBG][BISECT] t={elapsed:F1} window={window} skips {string.Join(" ", skipped)}");
        }

        return DbgBisectRanked.Skip(window * group).Take(group).Contains(vertexHash);
    }

    // TEMP: SHARPEMU_DBG_NOSKIP_WINDOW=from:len draws the skipped skin shaders in that window of process time.
    private static readonly (double From, double Length)? DbgNoSkipWindow = Environment.GetEnvironmentVariable("SHARPEMU_DBG_NOSKIP_WINDOW") is { Length: > 0 } dbgNs && dbgNs.Split(':') is { Length: 2 } dbgNsParts
        ? (double.Parse(dbgNsParts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(dbgNsParts[1], System.Globalization.CultureInfo.InvariantCulture)) : null;
    private static readonly long DbgNoSkipStart = System.Diagnostics.Stopwatch.GetTimestamp();
    private static bool DbgNoSkipNow()
    {
        if (DbgNoSkipWindow is not { } window) return false;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(DbgNoSkipStart).TotalSeconds;
        return elapsed >= window.From && elapsed < window.From + window.Length;
    }

    private static readonly HashSet<ulong> SkippedPixelHashes = Environment.GetEnvironmentVariable("SHARPEMU_SKIP_PS") switch
    {
        // Nothing is skipped by default: the skin subsurface composites render correctly since end-of-pipe
        // and flip labels are written at GPU completion. A hash list skips those pixel programs.
        null => [],
        var text => new HashSet<ulong>(text.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Where(item => !item.Equals("off", StringComparison.OrdinalIgnoreCase))
            .Select(item => Convert.ToUInt64(item.Replace("0x", ""), 16))),
    };

    // TEMP: SHARPEMU_DBG_SKIP_VS=hash,... skips draws of those vertex (or domain) programs. The water (domain shader
    // 4D314AF5...) used to be skipped here; it draws correctly since labels are written at GPU completion.
    private static readonly HashSet<ulong> DbgSkipVertexHashes = Environment.GetEnvironmentVariable("SHARPEMU_DBG_SKIP_VS") switch
    {
        null => [],
        var text => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Where(item => !item.Equals("off", StringComparison.OrdinalIgnoreCase))
            .Select(item => Convert.ToUInt64(item.Replace("0x", ""), 16)).ToHashSet(),
    };

    // TEMP: SHARPEMU_DBG_INST_CHECK=vsHash,... scans the 64-byte per-instance matrices the draw reads through the V# at
    // user-data pointer *(s[8:9] + 88) and logs entries that are not finite or are far outside the world.
    private static readonly HashSet<ulong> DbgInstanceCheck = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_INST_CHECK") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(item => item == "all" ? 0UL : Convert.ToUInt64(item.Replace("0x", ""), 16)).ToHashSet();
    private static readonly Dictionary<ulong, int> DbgVrefSeen = new();
    private static readonly double DbgVrefAfter = double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_VREF_AFTER"), out var dbgVrefAfter) ? dbgVrefAfter : 300; // TEMP
    private static readonly int DbgVrefMax = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_VREF_MAX"), out var dbgVrefMax) ? dbgVrefMax : 2; // TEMP
    private static long _dbgInstanceChecks;
    private void DbgCheckInstanceMatrices(ulong hash, uint[] userData, uint instances)
    {
        if (userData.Length < 2 || Interlocked.Increment(ref _dbgInstanceChecks) > 4000) return;
        var table = (userData[0] | ((ulong)userData[1] << 32)) & 0xFFFFFFFFFFFFUL;
        Span<byte> words = stackalloc byte[16];
        Span<byte> pointerBytes = stackalloc byte[8];
        if (!_host.TryReadGuest(table + 88, pointerBytes)) { Console.Error.WriteLine($"[DBG][INST] vs=0x{hash:X16} table=0x{table:X} unreadable"); return; }
        var descriptorAddress = BitConverter.ToUInt64(pointerBytes) & 0xFFFFFFFFFFFFUL;
        if (!_host.TryReadGuest(descriptorAddress, words)) { Console.Error.WriteLine($"[DBG][INST] vs=0x{hash:X16} V# at 0x{descriptorAddress:X} unreadable"); return; }
        var w0 = BitConverter.ToUInt32(words); var w1 = BitConverter.ToUInt32(words[4..]); var w2 = BitConverter.ToUInt32(words[8..]); var w3 = BitConverter.ToUInt32(words[12..]);
        var baseAddress = w0 | ((ulong)(w1 & 0xFFFF) << 32);
        var stride = (w1 >> 16) & 0x3FFF;
        var bytes = Math.Min(stride == 0 ? w2 : (ulong)w2 * stride, 4UL << 20);
        var data = new byte[bytes];
        if (bytes == 0 || !_host.TryReadGuest(baseAddress, data)) { Console.Error.WriteLine($"[DBG][INST] vs=0x{hash:X16} V#=0x{baseAddress:X}/stride{stride}/rec{w2}/w3=0x{w3:X} unreadable"); return; }
        var step = stride == 0 ? 64u : stride;
        int count = 0, bad = 0, zero = 0, firstBad = -1;
        for (ulong offset = 0; offset + 64 <= bytes; offset += step, count++)
        {
            var allZero = true; var ok = true;
            for (var i = 0; i < 16; i++)
            {
                var value = BitConverter.ToSingle(data, (int)offset + i * 4);
                if (value != 0) allZero = false;
                if (!float.IsFinite(value) || Math.Abs(value) > 1e6f) ok = false;
            }
            if (allZero) zero++;
            if (!ok) { bad++; if (firstBad < 0) firstBad = count; }
        }
        var sample = firstBad < 0 ? "" : " first=" + string.Join(",", Enumerable.Range(0, 16).Select(i => BitConverter.ToSingle(data, (int)(firstBad * step) + i * 4).ToString("G4")));
        Console.Error.WriteLine($"[DBG][INST] vs=0x{hash:X16} instances={instances} V#=0x{baseAddress:X}/stride{stride}/rec{w2} matrices={count} bad={bad} zero={zero} firstBad={firstBad}{sample}");
    }

    // TEMP: runs the vertex program on the CPU (Gen5ReferenceInterpreter) for the draw's indices and instances and logs
    // the clip-space position statistics, to tell bad inputs from a bad GPU translation.
    private void DbgReferenceVertices(ulong hash, uint userDataBase, uint[] userData, in DrawCall draw, in DrawEmission emission, in IndexSource indexSource)
    {
        if (Interlocked.Increment(ref _dbgInstanceChecks) > 3000) return;
        SharpEmu.ShaderCompiler.Gen5ShaderProgram? program = null;
        foreach (var (programHash, candidate) in Diagnostics.ReferenceCheck.Programs.Values)
            if (programHash == hash) { program = candidate; break; }
        if (program is null) { Console.Error.WriteLine($"[DBG][VREF] vs=0x{hash:X16} program not registered (set SHARPEMU_DBG_REF_CS)"); return; }
        uint count = draw.Count, instances = draw.InstanceCount, firstIndex = 0, firstInstance = emission.FirstInstance;
        var vertexOffset = emission.VertexOffset;
        if (emission.IndirectArgumentsAddress != 0)
        {
            Span<byte> args = stackalloc byte[20];
            if (!_host.TryReadGuest(emission.IndirectArgumentsAddress, args)) return;
            count = BitConverter.ToUInt32(args); instances = BitConverter.ToUInt32(args[4..]);
            if (emission.Indexed) { firstIndex = BitConverter.ToUInt32(args[8..]); vertexOffset = BitConverter.ToInt32(args[12..]); firstInstance = BitConverter.ToUInt32(args[16..]); }
            else { vertexOffset = (int)BitConverter.ToUInt32(args[8..]); firstInstance = BitConverter.ToUInt32(args[12..]); }
        }
        var vertexLimit = Math.Min(count, 3072u);
        if (vertexLimit == 0 || instances == 0) return;
        var ids = new uint[vertexLimit];
        if (emission.Indexed && indexSource.Enabled)
        {
            var size = indexSource.Type == Silk.NET.Vulkan.IndexType.Uint32 ? 4u : 2u;
            var bytes = new byte[vertexLimit * size];
            if (indexSource.HostData is { } host) Array.Copy(host, Math.Min(host.Length, firstIndex * size), bytes, 0, Math.Min(bytes.Length, Math.Max(0, host.Length - (int)(firstIndex * size))));
            else if (!_host.TryReadGuest(indexSource.Address + firstIndex * size, bytes)) return;
            for (var i = 0; i < vertexLimit; i++) ids[i] = (uint)((size == 4 ? BitConverter.ToUInt32(bytes, i * 4) : BitConverter.ToUInt16(bytes, i * 2)) + vertexOffset);
        }
        else for (var i = 0u; i < vertexLimit; i++) ids[i] = (uint)(i + vertexOffset);
        // The GPU's view: every earlier recorded command finished, read from the GPU buffer copies (guest memory where none exists).
        var gpuPage = new byte[4096];
        bool ReadGpuView(ulong address, Span<byte> destination)
        {
            if (destination.Length == 4096 && _host.DebugReadGpuCopy(address, gpuPage)) { gpuPage.CopyTo(destination); return true; }
            return _host.TryReadGuest(address, destination);
        }
        var gpuView = Environment.GetEnvironmentVariable("SHARPEMU_DBG_VREF_GPU") == "1";
        var interpreter = new SharpEmu.ShaderCompiler.Reference.Gen5ReferenceInterpreter(program, gpuView ? ReadGpuView : (address, destination) => _host.TryReadGuest(address, destination));
        if (Environment.GetEnvironmentVariable("SHARPEMU_DBG_VREF_PC") is { Length: > 0 } capturePc) interpreter.CapturePc = Convert.ToUInt64(capturePc.Replace("0x", ""), 16);
        var instanceLimit = Math.Max(1u, Math.Min(instances, 6144u / Math.Max(1u, vertexLimit)));
        try
        {
            for (var instance = 0u; instance < instanceLimit; instance++)
                for (var start = 0; start < ids.Length; start += 64)
                {
                    var lanes = Math.Min(64, ids.Length - start);
                    var instanceIds = Enumerable.Repeat(firstInstance + instance, lanes).ToArray();
                    interpreter.RunVertexWave(userData, (int)userDataBase, ids[start..(start + lanes)], instanceIds);
                }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[DBG][VREF] vs=0x{hash:X16} interpreter failed: {exception.GetType().Name}: {exception.Message}");
            return;
        }
        int nan = 0, huge = 0, behind = 0;
        float minW = float.MaxValue, maxW = float.MinValue, maxNdc = 0;
        foreach (var (_, x, y, z, w) in interpreter.Positions)
        {
            float fx = BitConverter.UInt32BitsToSingle(x), fy = BitConverter.UInt32BitsToSingle(y), fw = BitConverter.UInt32BitsToSingle(w);
            if (!float.IsFinite(fx) || !float.IsFinite(fy) || !float.IsFinite(fw)) { nan++; continue; }
            if (fw <= 0) { behind++; continue; }
            minW = Math.Min(minW, fw); maxW = Math.Max(maxW, fw);
            var ndc = Math.Max(Math.Abs(fx / fw), Math.Abs(fy / fw));
            maxNdc = Math.Max(maxNdc, ndc);
            if (ndc > 50) huge++;
        }
        if (nan == 0 && huge == 0 && !DbgInstanceCheck.Contains(hash)) { Console.Error.WriteLine($"[DBG][VREF] vs=0x{hash:X16} ok positions={interpreter.Positions.Count} maxNdc={maxNdc:G4}"); return; }
        if (interpreter.CapturedS is { } capturedS) // the V# in s[0:3] at the capture pc: dump its first records
        {
            var vBase = capturedS[0] | ((ulong)(capturedS[1] & 0xFFFF) << 32);
            var vStride = (capturedS[1] >> 16) & 0x3FFF;
            var head = new byte[512];
            if (vBase != 0 && ReadGpuView(vBase, head))
                Console.Error.WriteLine($"[DBG][VREF] vs=0x{hash:X16} V#=0x{vBase:X}/stride{vStride}/rec{capturedS[2]}/w3=0x{capturedS[3]:X} head={string.Join(" ", Enumerable.Range(0, 128).Select(i => BitConverter.ToSingle(head, i * 4)).Select(f => float.IsFinite(f) && (f == 0 || (Math.Abs(f) > 1e-6f && Math.Abs(f) < 1e7f)) ? f.ToString("G4") : "x"))}");
        }
        Console.Error.WriteLine($"[DBG][VREF] vs=0x{hash:X16} BAD view={(gpuView ? "gpu" : "guest")} indexed={emission.Indexed} indirect={emission.IndirectArgumentsAddress != 0} count={count} instances={instances} firstInstance={firstInstance} vertexOffset={vertexOffset} " +
            $"positions={interpreter.Positions.Count} nan={nan} behind={behind} hugeNdc={huge} maxNdc={maxNdc:G4} w=[{minW:G4},{maxW:G4}] " +
            $"captured v={string.Join(",", (interpreter.CapturedV ?? []).Select(v => v.ToString("X")))} s={string.Join(",", (interpreter.CapturedS ?? []).Select(v => v.ToString("X")))} index_type={indexSource.Type} " +
            $"firstLaneV0-23={string.Join(" ", (interpreter.FirstExportRegisters ?? []).Select(v => BitConverter.UInt32BitsToSingle(v) is var f && float.IsFinite(f) && Math.Abs(f) < 1e7f && Math.Abs(f) > 1e-7f ? f.ToString("G5") : $"0x{v:X}"))}");
    }

    private const ulong IndexedIndirectArgumentsSize = 20;
    private const ulong IndirectArgumentsSize = 16;

    private void EmitDraw(UserConfigRegisters userConfig, VertexInputInfo vertexInput, in DrawCall draw, in DrawEmission emission)
    {
        switch ((GuestPrimitiveType)userConfig.PrimitiveType)
        {
            case GuestPrimitiveType.PointList:
            case GuestPrimitiveType.LineList:
            case GuestPrimitiveType.LineStrip:
            case GuestPrimitiveType.TriangleList:
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.TriangleStrip:
            case GuestPrimitiveType.Polygon:
            case GuestPrimitiveType.RectangleList:
                if (emission.Indexed)
                {
                    _host.DrawIndexed(draw.Count, draw.InstanceCount, 0, emission.VertexOffset, emission.FirstInstance);
                }
                else
                {
                    _host.Draw(draw.Count, draw.InstanceCount, emission.FirstVertex, emission.FirstInstance);
                }

                break;
            case GuestPrimitiveType.RectangleListLegacy:
                if (emission.Indexed)
                {
                    throw _host.Fatal($"The primitive type is unknown for an indexed draw: primitiveType={userConfig.PrimitiveType}.");
                }

                if (_host.SupportsNativeRectangles)
                {
                    // Keep guest vertex IDs unchanged; the host fills each three-vertex rectangle.
                    _host.Draw(draw.Count, draw.InstanceCount, emission.FirstVertex, emission.FirstInstance);
                    break;
                }

                if (draw.Count != 3 || vertexInput.Buffers.Length != 0)
                {
                    throw _host.Fatal($"A legacy rectangle list needs three vertices and no vertex buffers: count={draw.Count} buffers={vertexInput.Buffers.Length}.");
                }

                _host.Draw(4, draw.InstanceCount, emission.FirstVertex, emission.FirstInstance);
                break;
            case GuestPrimitiveType.QuadListLegacy:
                if ((draw.Count & 0x3) != 0)
                {
                    throw _host.Fatal($"A legacy quad list count is not a multiple of four: count={draw.Count}.");
                }

                for (var i = 0u; i < draw.Count; i += 4)
                {
                    if (emission.Indexed)
                    {
                        _host.DrawIndexed(4, draw.InstanceCount, i, emission.VertexOffset, emission.FirstInstance);
                    }
                    else
                    {
                        _host.Draw(4, draw.InstanceCount, i + emission.FirstVertex, emission.FirstInstance);
                    }
                }

                break;
            default:
                throw _host.Fatal($"The primitive type is unknown: primitiveType={userConfig.PrimitiveType}.");
        }
    }
}
