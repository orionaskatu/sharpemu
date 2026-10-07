// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Silk.NET.Vulkan;
using ResourceSnapshot = SharpEmu.ShaderCompiler.Resources.ResourceSnapshot;

namespace SharpEmu.Libs.Gpu.Rendering;

public readonly record struct ComputeImageClear(BufferDescriptorWords Descriptor, uint PackedClear, ulong Size);

public sealed partial class RenderExecutor
{

    private const uint DispatchInitiatorUseThreadDimensions = 1u << 5;
    private const uint DispatchInitiatorBaseBits = 0x41;
    private const uint DispatchInitiatorModifierBits = 0xA038;
    private const uint DispatchInitiatorKnownMask = DispatchInitiatorBaseBits | DispatchInitiatorModifierBits;
    private const uint ImageClearDispatchInitiator = 0x61;
    private const uint ImageClearWaveSize = 64;
    private const uint Format32UInt = 20;
    private const uint ImageClearStride = 16;
    private const uint ImageClearUserDataCount = 8;

    // TEMP: SHARPEMU_DBG_SKIP_CS=hash,... drops those dispatches to bound their cost.
    // Ghost of Yotei culls its static world with a visibility feedback loop: compute 17444E6A... turns the
    // previous frame's visibility-buffer images into a per-triangle bitmask that the compaction passes read.
    // Those images (cleared with DCC constant-encoded fills) never receive visibility data in this emulation, so
    // the bitmask stays empty and every static mesh is culled. Until the loop is emulated the pass is replaced by
    // an all-ones fill (everything the frustum culler selected is drawn; occlusion culling is only an optimisation).
    // SHARPEMU_DBG_FILL_WRITTEN_CS=off keeps the real pass; a hash list overrides the default.
    private static readonly HashSet<ulong> DbgFillWrittenHashes = Environment.GetEnvironmentVariable("SHARPEMU_DBG_FILL_WRITTEN_CS") switch
    {
        null => [0x17444E6ABBF4F82CUL],
        var text => new HashSet<ulong>(text.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Where(item => !item.Equals("off", StringComparison.OrdinalIgnoreCase))
            .Select(item => Convert.ToUInt64(item.Replace("0x", ""), 16))),
    };
    // TEMP: SHARPEMU_DBG_BOOT_WINDOW=fromSeconds:lengthSeconds limits the replacement to that window of process time.
    private static readonly (double From, double Length)? DbgBootWindow = Environment.GetEnvironmentVariable("SHARPEMU_DBG_BOOT_WINDOW") is { Length: > 0 } dbgWindow && dbgWindow.Split(':') is { Length: 2 } dbgWindowParts
        ? (double.Parse(dbgWindowParts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(dbgWindowParts[1], System.Globalization.CultureInfo.InvariantCulture)) : null;
    private static readonly long DbgProcessStart = System.Diagnostics.Stopwatch.GetTimestamp();
    private static bool DbgInBootWindow()
    {
        if (DbgBootWindow is not { } window)
            return true;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds;
        return elapsed >= window.From && elapsed < window.From + window.Length;
    }

    private static readonly (ulong Hash, int Slot, int Words, ulong Offset)[] DbgPeek = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_PEEK") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
        .Select(item => item.Split(':')).Where(parts => parts.Length >= 3).Select(parts => (Convert.ToUInt64(parts[0].Replace("0x", ""), 16), Convert.ToInt32(parts[1], 16), Convert.ToInt32(parts[2]), parts.Length > 3 ? Convert.ToUInt64(parts[3].Replace("0x", ""), 16) : 0UL)).ToArray(); // TEMP
    private static readonly int DbgBootFrames = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_BOOT_FRAMES"), out var dbgBoot) ? dbgBoot : 0; // TEMP
    private static int _dbgBootCount; // TEMP
    private static readonly HashSet<ulong> DbgSkipHashes = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_SKIP_CS") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(h => Convert.ToUInt64(h.Replace("0x", ""), 16)).ToHashSet();

    private static readonly Dictionary<ulong, (int Count, long Groups, int Indirect)> _dbgDispatches = new(); // TEMP
    private static long _dbgDispatchLast = Environment.TickCount64;
    private static readonly bool _dbgDispatchStats = Environment.GetEnvironmentVariable("SHARPEMU_DBG_DISPATCHES") == "1";
    private static void DbgCountDispatch(ulong hash, long groups, bool indirect)
    {
        if (!_dbgDispatchStats) return;
        _dbgDispatches.TryGetValue(hash, out var e);
        _dbgDispatches[hash] = (e.Count + 1, e.Groups + groups, e.Indirect + (indirect ? 1 : 0));
        if (Environment.TickCount64 - _dbgDispatchLast < 10000) return;
        _dbgDispatchLast = Environment.TickCount64;
        Console.Error.WriteLine($"[DBG][DISPATCHES] total={_dbgDispatches.Values.Sum(v => v.Count)} distinct={_dbgDispatches.Count}");
        foreach (var (h, v) in _dbgDispatches.OrderByDescending(x => x.Value.Count).Take(25))
            Console.Error.WriteLine($"[DBG][DISPATCHES] n={v.Count} avg_groups={v.Groups / (double)v.Count:F1} indirect={v.Indirect} hash=0x{h:X16}");
        _dbgDispatches.Clear();
    }

    public void Dispatch(ulong submitId, RegisterBanks banks, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator, ulong indirectArgumentsAddress = 0)
    {
        if (!_host.IsRecording)
        {
            throw _host.Fatal("A dispatch has no recording command buffer.");
        }

        _host.RunPendingOperations();
        var compute = banks.Shader.Compute;
        _host.SetDebugInformation(RecordedOperation.DispatchDirect, submitId, groupsX, groupsY, groupsZ, dispatchInitiator, compute.Address);
        if (compute.Address == 0)
        {
            if (RenderTrace.Enabled && RenderTrace.NullComputeShader())
            {
                RenderTrace.Write($"Ignoring a dispatch with no compute shader: groups={groupsX}x{groupsY}x{groupsZ} initiator=0x{dispatchInitiator:X8}");
            }

            return;
        }

        var unknownBits = dispatchInitiator & ~DispatchInitiatorKnownMask;
        if (unknownBits != 0 && RenderTrace.Enabled && RenderTrace.UnknownInitiator())
        {
            RenderTrace.Write(
                $"The dispatch initiator has unknown bits: initiator=0x{dispatchInitiator:X8} unknown=0x{unknownBits:X8} shader=0x{compute.Address:X16} groups={groupsX}x{groupsY}x{groupsZ}");
        }

        var useThreadDimensions = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0;
        // A zero dimension executes no work. Check before materializing shaders: an indirect
        // buffer can be read here only when no pending GPU write can change its contents.
        var zeroDispatch = false;
        if (indirectArgumentsAddress == 0 || useThreadDimensions)
        {
            zeroDispatch = groupsX == 0 || groupsY == 0 || groupsZ == 0;
        }
        else
        {
            Span<byte> arguments = stackalloc byte[3 * sizeof(uint)];
            if (_host.TryReadCleanGuestBytes(indirectArgumentsAddress, arguments))
            {
                zeroDispatch = BinaryPrimitives.ReadUInt32LittleEndian(arguments) == 0 ||
                    BinaryPrimitives.ReadUInt32LittleEndian(arguments[sizeof(uint)..]) == 0 ||
                    BinaryPrimitives.ReadUInt32LittleEndian(arguments[(2 * sizeof(uint))..]) == 0;
            }
        }

        if (zeroDispatch)
        {
            if (RenderTrace.Enabled && RenderTrace.ZeroDispatch())
            {
                RenderTrace.Write($"Skipping a zero-sized dispatch: groups={groupsX}x{groupsY}x{groupsZ} " +
                    $"indirect=0x{indirectArgumentsAddress:X16} initiator=0x{dispatchInitiator:X8} shader=0x{compute.Address:X16}");
            }

            return;
        }

        var computeProgram = _pipelines.GetComputeProgram(compute, banks.Context.ShaderInterface, dispatchInitiator, groupsX, groupsY, groupsZ);
        if (computeProgram.Consumed)
        {
            return;
        }

        if (!computeProgram.Available)
        {
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"Skipping a dispatch without a program: shader=0x{compute.Address:X16} groups={groupsX}x{groupsY}x{groupsZ}");
            }

            return;
        }

        var input = computeProgram.Input;
        var program = input.Stage.Program ?? throw _host.Fatal($"The compute program is missing: shader=0x{compute.Address:X16}.");
        if (Diagnostics.DbgSequence.Active) // TEMP
        {
            var dbgWrites = new System.Text.StringBuilder();
            for (var index = 0; index < program.Buffers.Length && index < input.Stage.Resources.Buffers.Length; index++)
            {
                if (!program.Buffers[index].Written || input.Stage.Resources.Buffers[index].Length < 4)
                    continue;
                var dbgDescriptor = BufferDescriptorWords.From(input.Stage.Resources.Buffers[index]);
                dbgWrites.Append($" w{index}=0x{dbgDescriptor.Address:X}+0x{dbgDescriptor.Footprint() ?? 0:X}/stride{dbgDescriptor.Stride}/rec{dbgDescriptor.RecordCount}");
                if ((dbgDescriptor.Footprint() ?? 0) == 0)
                    dbgWrites.Append($"[raw={string.Join(",", input.Stage.Resources.Buffers[index].ToArray().Select(word => word.ToString("X8")))}]");
            }
            var dbgCb = banks.Context;
            Diagnostics.DbgSequence.Note($"  cbstate cb0[base=0x{dbgCb.ColorTargets[0].BaseAddress:X} info=0x{dbgCb.ColorTargets[0].Info:X8} dcc=0x{dbgCb.ColorTargets[0].DccAddress:X} clr0=0x{dbgCb.ColorTargets[0].ClearWord0:X8} clr1=0x{dbgCb.ColorClearWord1[0]:X8}] cb1[base=0x{dbgCb.ColorTargets[1].BaseAddress:X} info=0x{dbgCb.ColorTargets[1].Info:X8} dcc=0x{dbgCb.ColorTargets[1].DccAddress:X} clr0=0x{dbgCb.ColorTargets[1].ClearWord0:X8} clr1=0x{dbgCb.ColorClearWord1[1]:X8}]");
            var dbgUser = input.Stage.Resources.UserData;
            Diagnostics.DbgSequence.Note($"dispatch cs=0x{program.Hash:X16} groups={groupsX}x{groupsY}x{groupsZ} local={input.ThreadsX}{dbgWrites} ud4-7={(dbgUser.Length >= 8 && program.Hash == 0x2BD3CD129405F9A9UL ? $"{dbgUser[4]:X8},{dbgUser[5]:X8},{dbgUser[6]:X8},{dbgUser[7]:X8}" : "-")}");
        }
        if (DbgPeek.Length != 0) // TEMP: SHARPEMU_DBG_PEEK=hash:slot:dwords,... logs guest dwords of a bound buffer per dispatch
        {
            foreach (var (peekHash, peekSlot, peekWords, peekOffset) in DbgPeek)
            {
                if (peekHash != program.Hash)
                    continue;
                ulong peekAddress;
                if (peekSlot == 0xFE) // V# at ud pointer + offset: dump the first dwords of every record
                {
                    var peekUser2 = input.Stage.Resources.UserData;
                    var vsharp = new byte[16];
                    if (peekUser2.Length < 2 || !_host.TryReadGuest((peekUser2[0] | ((ulong)peekUser2[1] << 32)) + peekOffset, vsharp))
                        continue;
                    var baseAddress = BitConverter.ToUInt32(vsharp, 0) | ((ulong)(BitConverter.ToUInt32(vsharp, 4) & 0xFFFF) << 32);
                    var recordStride = (BitConverter.ToUInt32(vsharp, 4) >> 16) & 0x3FFF;
                    var recordCount = BitConverter.ToUInt32(vsharp, 8);
                    Console.Error.WriteLine($"[DBG][PEEK] V# base=0x{baseAddress:X} stride=0x{recordStride:X} records={recordCount} groups={groupsX}");
                    for (var record = 0u; record < Math.Min(recordCount, 64u); record++)
                    {
                        var recordBytes = new byte[peekWords * 4];
                        var recordRead = _host.TryReadGuest(baseAddress + record * recordStride, recordBytes);
                        Console.Error.WriteLine($"[DBG][PEEK]   rec {record}: " + (recordRead ? string.Join(" ", Enumerable.Range(0, peekWords).Select(index => BitConverter.ToUInt32(recordBytes, index * 4).ToString("X8"))) : "unreadable"));
                    }

                    continue;
                }

                if (peekSlot == 0xFF)
                {
                    var peekUser = input.Stage.Resources.UserData;
                    if (peekUser.Length < 2)
                        continue;
                    peekAddress = (peekUser[0] | ((ulong)peekUser[1] << 32)) + peekOffset;
                }
                else
                {
                    if (peekSlot >= input.Stage.Resources.Buffers.Length || input.Stage.Resources.Buffers[peekSlot].Length < 4)
                        continue;
                    peekAddress = BufferDescriptorWords.From(input.Stage.Resources.Buffers[peekSlot]).Address + peekOffset;
                }

                var peekBytes = new byte[peekWords * 4];
                var peekRead = _host.TryReadGuest(peekAddress, peekBytes);
                Console.Error.WriteLine($"[DBG][PEEK] t={System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds:F2} cs=0x{program.Hash:X} slot={peekSlot:X} addr=0x{peekAddress:X} groups={groupsX} " +
                    (peekRead ? string.Join(" ", Enumerable.Range(0, peekWords).Select(index => BitConverter.ToUInt32(peekBytes, index * 4).ToString("X8"))) : "unreadable"));
            }
        }

        if (DbgSkipHashes.Contains(program.Hash)) return; // TEMP
        if (DbgFillWrittenHashes.Contains(program.Hash) && DbgInBootWindow()) // TEMP: replace the dispatch by all-ones fills of its written buffers
        {
            for (var index = 0; index < program.Buffers.Length && index < input.Stage.Resources.Buffers.Length; index++)
            {
                if (!program.Buffers[index].Written || input.Stage.Resources.Buffers[index].Length < 4)
                    continue;
                var descriptor = BufferDescriptorWords.From(input.Stage.Resources.Buffers[index]);
                if (descriptor.Footprint() is { } footprint && descriptor.Address != 0 && footprint != 0)
                    _host.DebugFillBuffer(descriptor.Address & ~3UL, (footprint + 3) & ~3UL, 0xFFFFFFFF);
            }

            _host.ResetBindings();
            return;
        }
        DbgCountDispatch(program.Hash, groupsX * groupsY * groupsZ, indirectArgumentsAddress != 0); // TEMP
        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"Dispatch seq={RenderTrace.NextSequence()} submit={submitId} shader=0x{compute.Address:X16} hash=0x{program.Hash:X16} " +
                $"groups={groupsX}x{groupsY}x{groupsZ} local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ} wave={input.WaveSize} " +
                $"localDataShareDwords={input.LocalDataShareDwords} barriers={input.NeedsLocalDataShareBarriers} initiator=0x{dispatchInitiator:X8} " +
                $"buffers={program.Buffers.Length} images={program.Images.Length} samplers={program.SamplerCount}");
        }

        if (indirectArgumentsAddress == 0 && TryConsumeMetadataClear(input))
        {
            _host.ResetBindings();
            return;
        }

        if (indirectArgumentsAddress == 0 && TryConsumeConstantFill(input, groupsX, groupsY, groupsZ))
        {
            _host.ResetBindings();
            return;
        }

        if (indirectArgumentsAddress == 0 && TryConsumeImageClear(input, groupsX, groupsY, groupsZ, dispatchInitiator))
        {
            _host.ResetBindings();
            return;
        }

        if (useThreadDimensions)
        {
            // The indirect buffer carries thread counts in this mode, while Vulkan indirect
            // dispatch consumes workgroup counts. Use the CPU-resolved counts after conversion.
            indirectArgumentsAddress = 0;
            var threadsX = groupsX;
            var threadsY = groupsY;
            var threadsZ = groupsZ;
            groupsX = GroupsFromThreads(threadsX, compute.ThreadsX);
            groupsY = GroupsFromThreads(threadsY, compute.ThreadsY);
            groupsZ = GroupsFromThreads(threadsZ, compute.ThreadsZ);
            if (RenderTrace.Enabled && RenderTrace.ThreadDimensionConversion())
            {
                RenderTrace.Write(
                    $"Converted thread dimensions to groups: threads={threadsX}x{threadsY}x{threadsZ} " +
                    $"local={Math.Max(compute.ThreadsX, 1)}x{Math.Max(compute.ThreadsY, 1)}x{Math.Max(compute.ThreadsZ, 1)} groups={groupsX}x{groupsY}x{groupsZ}");
            }
        }

        _host.EndRendering();
        using (_host.BeginPreparation())
        {
            var pipeline = _pipelines.CreateComputePipeline(input, computeProgram.Program);
            var bindings = _host.PrepareBindings(input.Stage);
            if (program.UsesDeviceAddresses)
            {
                _host.PrepareDeviceAddresses();
            }

            _host.BindResources(bindings);
            Span<IPreparedBindings> stages = [bindings];
            _host.CommitBindings(PipelineBindPoint.Compute, in pipeline, stages);
            var hasStorageWrites = HasBufferWrites(input.Stage);
            foreach (var image in program.Images)
            {
                hasStorageWrites |= image.Written && image.Class == ImageResourceClass.Storage;
            }

            if (hasStorageWrites)
            {
                // Every earlier read of the written resources completes before this dispatch writes.
                _host.ShaderWriteHazardBarrier();
            }

            _host.BindPipeline(PipelineBindPoint.Compute, in pipeline);
            // The shader's local workgroup axes may have been remapped at compile time
            // (see Gen5SpirvTranslator.ComputeWorkgroupAxisOrder) so the largest NUM_THREAD
            // axis lands on a physical axis Vulkan actually allows it on. The dispatch group
            // counts must be permuted the same way, or vkCmdDispatch would hand group counts
            // for the wrong physical axis to the pipeline it built with the remapped sizes.
            var axisOrder = Gen5SpirvTranslator.ComputeWorkgroupAxisOrder(
                input.ThreadsX,
                input.ThreadsY,
                input.ThreadsZ);
            var logicalGroups = new[] { groupsX, groupsY, groupsZ };
            var physicalGroups = new uint[3];
            for (var logical = 0; logical < 3; logical++)
            {
                physicalGroups[axisOrder[logical]] = logicalGroups[logical];
            }

            if (_host.DebugCapturing) // TEMP: expose the resolved buffers so captures can name them by slot
            {
                var dbgBuffers = new (ulong, ulong)[input.Stage.Resources.Buffers.Length];
                for (var index = 0; index < dbgBuffers.Length; index++)
                {
                    if (input.Stage.Resources.Buffers[index].Length >= 4)
                    {
                        var dbgDescriptor = BufferDescriptorWords.From(input.Stage.Resources.Buffers[index]);
                        dbgBuffers[index] = (dbgDescriptor.Address, dbgDescriptor.Footprint() ?? 0);
                    }
                }

                _host.DebugSetDispatchBuffers(dbgBuffers);
            }

            if (indirectArgumentsAddress == 0 || !_host.TryDispatchIndirect(indirectArgumentsAddress))
            {
                _host.Dispatch(physicalGroups[0], physicalGroups[1], physicalGroups[2]);
            }
            _host.ShaderAccessBarrier();
            foreach (var (fillHash, fillSlot, fillValue) in DbgFillAfter) // TEMP: SHARPEMU_DBG_FILL_AFTER=hash:slot:value,... overwrites a bound buffer after the dispatch
            {
                if (fillHash != program.Hash || fillSlot >= input.Stage.Resources.Buffers.Length || input.Stage.Resources.Buffers[fillSlot].Length < 4)
                    continue;
                var fillDescriptor = BufferDescriptorWords.From(input.Stage.Resources.Buffers[fillSlot]);
                if (fillDescriptor.Footprint() is { } fillFootprint && fillDescriptor.Address != 0 && fillFootprint != 0)
                    _host.DebugFillBuffer(fillDescriptor.Address & ~3UL, (fillFootprint + 3) & ~3UL, fillValue);
            }
        }

        _host.ResetBindings();
    }

    private static readonly (ulong Hash, int Slot, uint Value)[] DbgFillAfter = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_FILL_AFTER") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
        .Select(item => item.Split(':')).Where(parts => parts.Length == 3).Select(parts => (Convert.ToUInt64(parts[0].Replace("0x", ""), 16), Convert.ToInt32(parts[1], 16), Convert.ToUInt32(parts[2].Replace("0x", ""), 16))).ToArray(); // TEMP

    // The dispatch counts threads; the host counts groups of the shader's thread size.
    public static uint GroupsFromThreads(uint threads, uint groupSize)
    {
        if (threads == 0)
        {
            return 0;
        }

        var size = Math.Max(groupSize, 1u);
        return (threads + size - 1) / size;
    }

    private BufferDescriptorWords DecodeBufferDescriptor(ResourceSnapshot resources, int index)
    {
        var words = resources.Buffers[index];
        if (words.Length < 4)
        {
            throw _host.Fatal($"A buffer descriptor is too short: index={index} words={words.Length}.");
        }

        return BufferDescriptorWords.From(words);
    }

    // The program's packed stride against the descriptor's. With runtime strides the
    // program carries only the flags and indexes with the descriptor's stride.
    private static bool StrideMatches(uint programStride, uint descriptorStride) =>
        (programStride & ~BufferSpecialization.StrideMask) == (descriptorStride & ~BufferSpecialization.StrideMask) &&
        ((programStride & BufferSpecialization.StrideMask) == 0 ||
         (programStride & BufferSpecialization.StrideMask) == (descriptorStride & BufferSpecialization.StrideMask));

    // A full overwrite of registered metadata by a compute shader becomes a tracked clear.
    private bool TryConsumeMetadataClear(ComputeInputInfo input)
    {
        var program = input.Stage.Program!;
        var resources = input.Stage.Resources;
        if (resources.Buffers.Length != program.Buffers.Length)
        {
            throw _host.Fatal($"The compute buffer count does not match the program: descriptors={resources.Buffers.Length} program={program.Buffers.Length}.");
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            var resource = program.Buffers[i];
            var descriptor = DecodeBufferDescriptor(resources, i);
            // Metadata that is also read is not a proven overwrite; the dispatch runs as written.
            if (_host.IsMetadata(descriptor.Address) && (!resource.Written || resource.Read))
            {
                return false;
            }
        }

        if (program.HasBitwiseExclusiveOr)
        {
            return false;
        }

        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written)
            {
                continue;
            }

            var descriptor = DecodeBufferDescriptor(resources, i);
            if (_host.ClearMetadata(descriptor.Address))
            {
                return true;
            }
        }

        return false;
    }

    // Recognizes a dispatch that fills one formatted buffer with a single value over every record.
    public ComputeImageClear? TryDecodeImageClear(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program ?? throw _host.Fatal("The compute stage has no program.");
        var resources = input.Stage.Resources;
        if (program.Buffers.Length != 1 || resources.Buffers.Length != 1 || program.Images.Length != 0 || program.SamplerCount != 0 ||
            program.UsesDeviceAddresses || resources.Images.Length != 0 || resources.Samplers.Length != 0)
        {
            return null;
        }

        var resource = program.Buffers[0];
        var words = resources.Buffers[0];
        if (words.Length != 4)
        {
            return null;
        }

        var descriptor = BufferDescriptorWords.From(words);
        if (!resource.Formatted || !resource.Written || resource.Read || resource.Atomic || resource.Scalar || resource.MaxByteExtent != ImageClearStride ||
            descriptor.Stride != ImageClearStride || descriptor.Format != BufferDescriptorWords.Format32x4UInt || descriptor.SwizzleEnabled ||
            descriptor.IndexStride != 0 || descriptor.AddThreadId || !StrideMatches(resource.PackedStride, descriptor.PackedStride) ||
            program.UserDataBase != 0 || resources.UserData.Length != ImageClearUserDataCount)
        {
            return null;
        }

        for (var i = 0; i < words.Length; i++)
        {
            if (words[i] != resources.UserData[i])
            {
                return null;
            }
        }

        var clear = resources.UserData[4];
        if (resources.UserData[5] != clear || resources.UserData[6] != clear || resources.UserData[7] != clear)
        {
            return null;
        }

        var fullDispatch =
            input.DispatchThreadDimensions && input.ThreadsX == ImageClearWaveSize && input.ThreadsY == 1 && input.ThreadsZ == 1 &&
            groupsX != 0 && groupsY == 1 && groupsZ == 1 &&
            input.DispatchThreadsX == groupsX && input.DispatchThreadsY == 1 && input.DispatchThreadsZ == 1 &&
            input.GroupIdX && !input.GroupIdY && !input.GroupIdZ &&
            input.ThreadIdCount == 1 && input.WaveSize == ImageClearWaveSize && !input.ThreadGroupSizeEnabled &&
            dispatchInitiator == ImageClearDispatchInitiator && groupsX % input.ThreadsX == 0 && descriptor.RecordCount == groupsX;
        var size = descriptor.Footprint() ?? throw _host.Fatal($"The compute buffer footprint overflows: stride={descriptor.Stride} records={descriptor.RecordCount}.");
        if (!fullDispatch || size == 0)
        {
            return null;
        }

        return new ComputeImageClear(descriptor, clear, size);
    }

    // A constant fill that covers a whole image or DCC metadata becomes a clear. Anything
    // else, including a fill of plain buffer memory, runs as the guest wrote it.
    private bool TryConsumeConstantFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program!;
        if (program.ConstantFill is not { } fill || program.UserDataBase != 0)
        {
            return false;
        }

        var userData = input.Stage.Resources.UserData;
        if (fill.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || groupsY != 1 || groupsZ != 1 ||
            fill.DestinationScalarResource + 4 > userData.Length || fill.SourceScalarResource + 4 > userData.Length)
        {
            return false;
        }

        var destination = BufferDescriptorWords.From(userData.AsSpan((int)fill.DestinationScalarResource, 4).ToArray());
        var source = BufferDescriptorWords.From(userData.AsSpan((int)fill.SourceScalarResource, 4).ToArray());
        Span<byte> valueBytes = stackalloc byte[sizeof(uint)];
        if (destination.Format != Format32UInt || destination.Stride != sizeof(uint) || destination.SwizzleEnabled || destination.AddThreadId ||
            (ulong)groupsX * input.ThreadsX != destination.RecordCount || source.RecordCount == 0 ||
            !_host.TryReadGuest(source.Address, valueBytes))
        {
            return false;
        }

        var value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(valueBytes);
        var size = (ulong)destination.RecordCount * sizeof(uint);
        var consumed = _host.TryClearImageFromBuffer(destination.Address, size, value) ||
                       _host.TryAbsorbDccFill(destination.Address, size, value);
        if (RenderTrace.Enabled && RenderTrace.MetadataClear())
        {
            RenderTrace.Write($"Constant fill: shader=0x{program.Hash:X16} address=0x{destination.Address:X16} size=0x{size:X} value=0x{value:X8} consumed={consumed}");
        }

        return consumed;
    }

    private bool TryConsumeImageClear(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        if (TryDecodeImageClear(input, groupsX, groupsY, groupsZ, dispatchInitiator) is not { } clear)
        {
            return false;
        }

        var address = clear.Descriptor.Address;
        var hash = input.Stage.Program!.Hash;
        if (!_host.TryClearImageFromBuffer(address, clear.Size, clear.PackedClear))
        {
            // A metadata fill may run before its target is bound; the store keeps it pending and the dispatch runs.
            var registered = _host.TryAbsorbDccFill(address, clear.Size, clear.PackedClear);
            if (RenderTrace.Enabled && RenderTrace.MetadataClear())
            {
                RenderTrace.Write(
                    $"{(registered ? "Tracked" : "Deferred")} a metadata clear: shader=0x{hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
            }

            return registered;
        }

        if (RenderTrace.Enabled && RenderTrace.ImageClear())
        {
            RenderTrace.Write($"Consumed a compute image clear: shader=0x{hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
        }

        return true;
    }
}
