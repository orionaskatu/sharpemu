// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
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

        Span<VertexBufferRange> ranges = stackalloc VertexBufferRange[VertexInputInfo.MaxBuffers];
        var rangeCount = 0;
        foreach (ref readonly var vertex in buffers.AsSpan())
        {
            var size = vertex.Size;
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
        Span<VertexBufferRange> merged = stackalloc VertexBufferRange[VertexInputInfo.MaxBuffers];
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
        for (var slot = 0; slot < buffers.Length; slot++)
        {
            ref readonly var vertex = ref buffers[slot];
            if (vertex.Size == 0)
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

    // Binds everything the draw needs inside one preparation scope, then records it.
    private static int _dbgGrassLogs; // TEMP
    private static int _dbgHogLogs; // TEMP
    private void RecordDraw(
        ulong submitId,
        RegisterBanks banks,
        in DrawCall draw,
        ref DrawState state,
        PrimitiveTopology topology,
        in DrawEmission emission,
        in IndexSource indexSource,
        bool primitiveRestart,
        bool setBindDebug,
        bool setAutoDebug)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DrawResourcePreparation);
        var context = banks.Context;
        var vertexInput = state.Programs.VertexInput;
        if (DbgSkipVertexHashes.Count != 0 && DbgSkipVertexHashes.Contains(vertexInput.Stage.Program?.Hash ?? 0)) // TEMP
        {
            return;
        }

        Diagnostics.DbgSequence.Tally("DR", vertexInput.Stage.Program?.Hash ?? 0); // TEMP
        if (DbgInstanceCheck.Count != 0 && (DbgInstanceCheck.Contains(0) || DbgInstanceCheck.Contains(vertexInput.Stage.Program?.Hash ?? 0)) && // TEMP
            vertexInput.Stage.Program is not null && System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds > 300 &&
            DbgVrefSeen.GetValueOrDefault(vertexInput.Stage.Program.Hash) is var seen && seen < 2 && (DbgVrefSeen[vertexInput.Stage.Program.Hash] = seen + 1) > 0)
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
            return;
        }

        if (DbgBisectSkip(vertexInput.Stage.Program?.Hash ?? 0, state.PixelActive)) // TEMP
        {
            return;
        }
        var pixelInput = state.Programs.PixelInput;
        if (state.PixelActive && SkippedPixelHashes.Count != 0 && !DbgNoSkipNow() && SkippedPixelHashes.Contains(pixelInput.Stage.Program?.Hash ?? 0))
        {
            return;
        }

        using var preparation = _host.BeginPreparation();
        IPreparedBindings vertexBindings;
        IPreparedBindings? pixelBindings;
        try
        {
            // The pixel program reads its position in host texels; the attachments say how
            // many of those one guest pixel covers.
            var attachmentScale = AttachmentRenderScale(in state);
            vertexBindings = PrepareBindings(vertexInput.Stage with { AttachmentRenderScale = attachmentScale });
            pixelBindings = state.PixelActive
                ? PrepareBindings(pixelInput.Stage with { AttachmentRenderScale = attachmentScale })
                : null;
        }
        catch (DrawImageTypeMismatchException rejection)
        {
            if (_strictDrawResources)
            {
                throw _host.Fatal(rejection.Message);
            }

            if (_reportedDrawImageTypeMismatches.Add(rejection.WarningKey))
            {
                Console.Error.WriteLine($"[GPU][WARN][DRAW_SKIPPED] {rejection.Message} " +
                    "The draw was not executed. Images and FPS can be incorrect. Set SHARPEMU_STRICT_COMPUTE=1 to stop on this failure.");
            }

            return;
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

        var vertexBuffers = AcquireVertexBuffers(vertexInput);
        var indexBuffer = AcquireIndexBuffer(in indexSource);
        var dbgArgsState = emission.IndirectArgumentsAddress != 0 && _host.DebugCapturing ? _host.DebugBufferState(emission.IndirectArgumentsAddress, emission.Indexed ? IndexedIndirectArgumentsSize : IndirectArgumentsSize) : ""; // TEMP
        var indirectArguments = emission.IndirectArgumentsAddress != 0
            ? _host.ObtainBuffer(emission.IndirectArgumentsAddress, emission.Indexed ? IndexedIndirectArgumentsSize : AutoIndirectArgumentsSize, isWritten: false)
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

        if (emission.IndirectArgumentsAddress != 0 && _host.DebugCapturing) _host.DebugNote($"argsaddr=0x{emission.IndirectArgumentsAddress:X} marker='{SharpEmu.Libs.Diagnostics.DbgSequence.Marker}' submit={submitId} vs=0x{vertexInput.Stage.Program?.Hash ?? 0:X} {dbgArgsState} bound=0x{indirectArguments.Handle:X}+0x{indirectArguments.Offset:X}"); // TEMP
        if (_host.DebugCapturing) // TEMP: let captures snapshot the vertex stage's buffers by slot
        {
            var dbgBuffers = new (ulong, ulong)[vertexInput.Stage.Resources.Buffers.Length];
            for (var index = 0; index < dbgBuffers.Length; index++)
            {
                if (vertexInput.Stage.Resources.Buffers[index].Length >= 4)
                {
                    var dbgDescriptor = BufferDescriptorWords.From(vertexInput.Stage.Resources.Buffers[index]);
                    dbgBuffers[index] = (dbgDescriptor.Address, dbgDescriptor.Footprint() ?? 0);
                }
            }

            _host.DebugSetDispatchBuffers(dbgBuffers);
            _host.DebugSetDrawHash(vertexInput.Stage.Program?.Hash ?? 0);
        }

        if (emission.IndirectArgumentsAddress != 0) _host.DebugCaptureIndirectArguments(indirectArguments); // TEMP
        if (emission.IndirectArgumentsAddress != 0) _host.DebugReportArgsWriter(vertexInput.Stage.Program?.Hash ?? 0, emission.IndirectArgumentsAddress); // TEMP
        _host.BindVertexBuffers(vertexBuffers, vertexInput);

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

        _host.PrepareGraphicsPipeline(in pipeline);
        _host.BeginRendering(in state.Rendering);
        _host.BindPipeline(PipelineBindPoint.Graphics, in pipeline);
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x500);
        }

        if (state.Programs.Tessellation is not null)
        {
            _host.Draw(draw.Count, 1, 0, 0);
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
        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x600);
        }

        var writeStages = PipelineStageFlags.None;
        if (HasBufferWrites(vertexInput.Stage))
        {
            writeStages |= vertexInput.Tessellation is null ? PipelineStageFlags.VertexShaderBit : PipelineStageFlags.TessellationEvaluationShaderBit;
        }

        if (state.PixelActive && HasBufferWrites(pixelInput.Stage))
        {
            writeStages |= PipelineStageFlags.FragmentShaderBit;
        }

        if (writeStages != PipelineStageFlags.None)
        {
            _host.EndRendering();
            _host.ShaderWriteBarrier(writeStages);
        }

        if (setAutoDebug)
        {
            SetDrawDebugPhase(submitId, in draw, 0x700);
        }
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
        // The skin subsurface composites (white skin). The grass pixel shader F23F7E7F... used to hang the GPU in an
        // EXEC waterfall that never ended in one-lane graphics waves; fixed in the translator, so it draws again.
        null => [0x722412AB9E88B57EUL, 0xD8CB0512A0ACB2A0UL, 0x8C2A2B9065B2CA33UL, 0xE2FC69DC66748B17UL, 0x3FABF91EFFC6254DUL, 0x4F17FC98EE8B5640UL],
        var text => new HashSet<ulong>(text.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Where(item => !item.Equals("off", StringComparison.OrdinalIgnoreCase))
            .Select(item => Convert.ToUInt64(item.Replace("0x", ""), 16))),
    };

    // The water (tessellated triangle patches, domain shader 4D314AF5..., pixel shader 459074A8...). Its pixel shader's 5 s per
    // frame (an EXEC waterfall that never ended in one-lane graphics waves) is fixed, but drawn it still covers the scene with
    // dark planes and smeared streaks, so it stays skipped by default; SHARPEMU_DBG_SKIP_VS=off draws it.
    private static readonly HashSet<ulong> DbgSkipVertexHashes = Environment.GetEnvironmentVariable("SHARPEMU_DBG_SKIP_VS") switch
    {
        null => [0x4D314AF535375953UL],
        var text => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Where(item => !item.Equals("off", StringComparison.OrdinalIgnoreCase))
            .Select(item => Convert.ToUInt64(item.Replace("0x", ""), 16)).ToHashSet(),
    };

    // TEMP: SHARPEMU_DBG_INST_CHECK=vsHash,... scans the 64-byte per-instance matrices the draw reads through the V# at
    // user-data pointer *(s[8:9] + 88) and logs entries that are not finite or are far outside the world.
    private static readonly HashSet<ulong> DbgInstanceCheck = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_INST_CHECK") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(item => item == "all" ? 0UL : Convert.ToUInt64(item.Replace("0x", ""), 16)).ToHashSet();
    private static readonly Dictionary<ulong, int> DbgVrefSeen = new();
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

                if (IsLegacyRectangleBatch(draw.Count))
                {
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
