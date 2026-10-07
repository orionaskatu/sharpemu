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
        null => [], // the real visibility pass: forcing every triangle visible draws all instances and garbage geometry
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
    private static readonly HashSet<ulong> DbgRunOnly = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_RUN_CS") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(h => Convert.ToUInt64(h.Replace("0x", ""), 16)).ToHashSet(); // TEMP
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

        var dbgT0 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
        var computeProgram = _pipelines.GetComputeProgram(compute, banks.Context.ShaderInterface, dispatchInitiator, groupsX, groupsY, groupsZ);
        var dbgT1 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
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
        if (Diagnostics.DccWriterTrace.Enabled)
        {
            Diagnostics.DccWriterTrace.CurrentProgram = program.Hash;
        }

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

        if (DbgSkipHashes.Contains(program.Hash) || Environment.GetEnvironmentVariable("SHARPEMU_DBG_SKIP_ALL_CS") == "1" || (DbgRunOnly.Count != 0 && !DbgRunOnly.Contains(program.Hash))) return; // TEMP
        DbgVcullWatch(program, input); // TEMP
        if (Environment.GetEnvironmentVariable("SHARPEMU_DBG_STRUCTS") == "1" && program.Hash is 0x25FCDA2A90D50DD4UL or 0x525A55D3242304C9UL or 0xB7DF200E29FEE750UL && System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds > 70 && DbgStructQuota(program.Hash)) // TEMP
        {
            var dbgUd = input.Stage.Resources.UserData;
            var dbgStruct = new byte[256];
            var dbgOk = dbgUd.Length >= 2 && _host.TryReadGuest(dbgUd[0] | ((ulong)dbgUd[1] << 32), dbgStruct);
            var dbgDw = Enumerable.Range(0, 64).Select(i => BitConverter.ToUInt32(dbgStruct, i * 4)).ToArray();
            var dbgBufs = string.Join(" ", Enumerable.Range(0, input.Stage.Resources.Buffers.Length).Select(index =>
            {
                var words = input.Stage.Resources.Buffers[index];
                if (words.Length < 4) return "-";
                var descriptor = BufferDescriptorWords.From(words);
                return $"{descriptor.Address:X}+{descriptor.Footprint() ?? 0:X}{(program.Buffers[index].Written ? "w" : "")}";
            }));
            Console.Error.WriteLine($"[DBG][STRUCT] cs={program.Hash:X16} g={groupsX} read={dbgOk} struct={string.Join(",", dbgDw.Select(w => w.ToString("X")))} bufs={dbgBufs}");
        }

        if (program.Hash == 0x17444E6ABBF4F82CUL && Environment.GetEnvironmentVariable("SHARPEMU_DBG_VCULL_TABLES") == "1" && Interlocked.Increment(ref _dbgVcullTableLogs) <= 3) // TEMP
        {
            for (var index = 0; index < program.Buffers.Length && index < input.Stage.Resources.Buffers.Length; index++)
            {
                var words = input.Stage.Resources.Buffers[index];
                if (words.Length < 4) continue;
                var descriptor = BufferDescriptorWords.From(words);
                var footprint = descriptor.Footprint() ?? 0;
                var bytes = new byte[Math.Min(footprint, 4096UL * 4)];
                var ok = descriptor.Address != 0 && _host.TryReadGuest(descriptor.Address, bytes);
                var dwords = Enumerable.Range(0, bytes.Length / 4).Select(i => BitConverter.ToUInt32(bytes, i * 4)).ToArray();
                Console.Error.WriteLine($"[DBG][VTAB] buf[{index}] addr=0x{descriptor.Address:X} bytes=0x{footprint:X} written={program.Buffers[index].Written} read={ok} nonzero={dwords.Count(x => x != 0)} first={string.Join(",", dwords.Take(48).Select(x => x.ToString("X")))}");
            }

            var ud = input.Stage.Resources.UserData;
            Console.Error.WriteLine($"[DBG][VTAB] groups={groupsX}x{groupsY}x{groupsZ} ud={string.Join(",", ud.Take(16).Select(x => x.ToString("X")))}");
            var rect = new byte[16];
            if (ud.Length >= 2 && _host.TryReadGuest(ud[0] | ((ulong)ud[1] << 32), rect))
                Console.Error.WriteLine($"[DBG][VTAB] ud-struct rect={string.Join(",", Enumerable.Range(0, 4).Select(i => BitConverter.ToUInt32(rect, i * 4).ToString("X")))}");
        }
        if ((DbgFillWrittenHashes.Contains(program.Hash) || (program.Hash == 0x17444E6ABBF4F82CUL && DbgInForceWindow())) && DbgInBootWindow() && !DbgInRealVcullWindow()) // TEMP: replace the dispatch by all-ones fills of its written buffers
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
        if (Environment.GetEnvironmentVariable("SHARPEMU_DBG_DISPLOG") == "2" && System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds is var dbgNow && dbgNow >= double.Parse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_DISPLOG_FROM") ?? "0") && dbgNow < double.Parse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_DISPLOG_FROM") ?? "0") + 8) // TEMP: per-dispatch inputs
        {
            var dbgUser = input.Stage.Resources.UserData;
            var dbgBuffers = string.Join(" ", Enumerable.Range(0, input.Stage.Resources.Buffers.Length).Select(index =>
            {
                var words = input.Stage.Resources.Buffers[index];
                if (words.Length < 4) return "-";
                var descriptor = BufferDescriptorWords.From(words);
                return $"{descriptor.Address:X}+{descriptor.Footprint() ?? 0:X}{(program.Buffers[index].Written ? "w" : "")}";
            }));
            Console.Error.WriteLine($"[DBG][DISPIN] t={System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds:F3} cs={program.Hash:X16} g={groupsX}x{groupsY}x{groupsZ} ind={(indirectArgumentsAddress != 0 ? 1 : 0)} ud=" + string.Join(",", dbgUser.Take(16).Select(word => word.ToString("X"))) + " buf=" + dbgBuffers);
        }
        if (Environment.GetEnvironmentVariable("SHARPEMU_DBG_DISPLOG") == "1") Console.Error.WriteLine($"[DBG][DISPLOG] cs=0x{program.Hash:X16} groups={groupsX}x{groupsY}x{groupsZ} indirect={indirectArgumentsAddress != 0} local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ} wave={input.WaveSize} buffers={program.Buffers.Length} images={program.Images.Length}"); // TEMP
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

        if (indirectArgumentsAddress == 0 && !SharpEmu.ShaderCompiler.DbgFlags.Disabled("boundedfill") && TryConsumeBoundedFill(input, groupsX, groupsY, groupsZ, dispatchInitiator))
        {
            _host.ResetBindings();
            return;
        }

        if (indirectArgumentsAddress == 0 && !SharpEmu.ShaderCompiler.DbgFlags.Disabled("boundedfill") && TryConsumeBoundedCopy(input, groupsX, groupsY, groupsZ, dispatchInitiator))
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
        var dbgT2 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
        long dbgT3 = 0, dbgT4 = 0, dbgT5 = 0;
        var dbgTb0 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
        using (_host.BeginPreparation())
        {
            DbgBeginPrep += System.Diagnostics.Stopwatch.GetTimestamp() - dbgTb0; // TEMP
            // The host may compile this program off the command-stream thread and report that
            // the pipeline is not ready yet. The dispatch waits for it rather than being
            // dropped: a guest that does not replay a one-shot dispatch deadlocks, because
            // the dispatch can feed a label a later packet waits on. Asking again keeps the
            // wait outside the pipeline cache's lock, so other queues can create their own
            // pipelines while this program compiles.
            PipelineHandle pipeline;
            while (!_pipelines.TryCreateComputePipeline(input, computeProgram.Program, out pipeline))
            {
                Thread.Sleep(1);
            }

            dbgT3 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
            var bindings = PrepareBindings(input.Stage);
            dbgT4 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
            if (program.UsesDeviceAddresses)
            {
                _host.PrepareDeviceAddresses();
            }

            _host.BindResources(bindings);
            dbgT5 = System.Diagnostics.Stopwatch.GetTimestamp(); // TEMP
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

        DbgDispStat(program.Hash, dbgT0, dbgT1, dbgT2, dbgT3, dbgT4, dbgT5, System.Diagnostics.Stopwatch.GetTimestamp(), program.Buffers.Length, program.Images.Length); // TEMP
        _host.ResetBindings();
    }

    // TEMP: SHARPEMU_DBG_DISP_STATS=1 prints per-program dispatch phase costs every 10 s.
    private static readonly bool _dbgDispStats = Environment.GetEnvironmentVariable("SHARPEMU_DBG_DISP_STATS") == "1";
    // TEMP: SHARPEMU_DBG_REAL_VCULL=from:len runs the real visibility-feedback pass in that window of process time
    // and logs which draws and dispatches touch its two visibility images (T# base addresses learned at the pass).
    private static readonly (double From, double Length)? DbgRealVcull = Environment.GetEnvironmentVariable("SHARPEMU_DBG_REAL_VCULL") is { Length: > 0 } dbgRv && dbgRv.Split(':') is { Length: 2 } dbgRvParts
        ? (double.Parse(dbgRvParts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(dbgRvParts[1], System.Globalization.CultureInfo.InvariantCulture)) : null;
    private static bool DbgInRealVcullWindow()
    {
        if (DbgRealVcull is not { } window) return false;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds;
        return elapsed >= window.From && elapsed < window.From + window.Length;
    }

    // TEMP: SHARPEMU_DBG_FORCE_WINDOW=from:len forces every triangle visible (all-ones bitmask) in that window of process time.
    private static readonly (double From, double Length)? DbgForceWindow = Environment.GetEnvironmentVariable("SHARPEMU_DBG_FORCE_WINDOW") is { Length: > 0 } dbgFw && dbgFw.Split(':') is { Length: 2 } dbgFwParts
        ? (double.Parse(dbgFwParts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(dbgFwParts[1], System.Globalization.CultureInfo.InvariantCulture)) : null;
    private static bool DbgInForceWindow()
    {
        if (DbgForceWindow is not { } window) return false;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(DbgProcessStart).TotalSeconds;
        return elapsed >= window.From && elapsed < window.From + window.Length;
    }

    private static readonly HashSet<ulong> DbgVcullAddresses = new();
    private static int _dbgVcullLogs;
    private static void DbgVcullWatch(ShaderProgramInfo program, ComputeInputInfo input)
    {
        if (DbgRealVcull is null || !DbgInRealVcullWindow() || input.Stage.Program is not { } stageProgram) return;
        var images = input.Stage.Resources.Images;
        if (program.Hash == 0x17444E6ABBF4F82CUL || stageProgram.Hash == 0x17444E6ABBF4F82CUL)
        {
            for (var index = 0; index < images.Length; index++)
            {
                if (images[index].Length < 8) continue;
                var address = ((ulong)images[index][0] | ((ulong)(images[index][1] & 0xFFu) << 32)) << 8;
                if (DbgVcullAddresses.Add(address))
                {
                    Diagnostics.DbgTargetWatch.Addresses.Add(address);
                    Console.Error.WriteLine($"[DBG][VCULL] image[{index}] addr=0x{address:X} t#={string.Join(",", images[index].Take(8).Select(word => word.ToString("X8")))}");
                }
            }

            return;
        }

        for (var index = 0; index < images.Length; index++)
        {
            if (images[index].Length < 8) continue;
            var address = ((ulong)images[index][0] | ((ulong)(images[index][1] & 0xFFu) << 32)) << 8;
            if (DbgVcullAddresses.Contains(address) && Interlocked.Increment(ref _dbgVcullLogs) <= 400)
                Console.Error.WriteLine($"[DBG][VCULL] dispatch cs=0x{stageProgram.Hash:X16} binds watched image[{index}] addr=0x{address:X} t#={string.Join(",", images[index].Take(8).Select(word => word.ToString("X8")))}");
        }
    }
    private static int _dbgVcullTableLogs, _dbgStructLogs;
    private static readonly Dictionary<ulong, int> _dbgStructQuota = new();
    private static bool DbgStructQuota(ulong hash)
    {
        lock (_dbgStructQuota)
        {
            var count = _dbgStructQuota.GetValueOrDefault(hash);
            _dbgStructQuota[hash] = count + 1;
            return count < 14;
        }
    }
    private static long DbgBeginPrep; // TEMP: ticks spent in BeginPreparation (flush + capacity wait)
    private static long DbgEndRendering; // TEMP
    private static readonly Dictionary<ulong, long[]> _dbgDispPhases = new();
    private static long _dbgDispPhasesLast = Environment.TickCount64;
    private static void DbgDispStat(ulong hash, long t0, long t1, long t2, long t3, long t4, long t5, long t6, int buffers, int images)
    {
        if (!_dbgDispStats) return;
        if (!_dbgDispPhases.TryGetValue(hash, out var a)) _dbgDispPhases[hash] = a = new long[8];
        a[0]++; a[1] += t1 - t0; a[2] += t3 - t2; a[3] += t4 - t3; a[4] += t5 - t4; a[5] += t6 - t5; a[6] = buffers; a[7] = images;
        if (Environment.TickCount64 - _dbgDispPhasesLast < 10000) return;
        _dbgDispPhasesLast = Environment.TickCount64;
        var f = System.Diagnostics.Stopwatch.Frequency / 1e6;
        Console.Error.WriteLine($"[DBG][DISPPH] beginprep_ms={DbgBeginPrep / (System.Diagnostics.Stopwatch.Frequency / 1000.0):F0}"); DbgBeginPrep = 0;
        Console.Error.WriteLine($"[DBG][DISPPH] programs={_dbgDispPhases.Count} total_ms={_dbgDispPhases.Values.Sum(v => v[1] + v[2] + v[3] + v[4] + v[5]) / f / 1000:F0}");
        foreach (var (h, v) in _dbgDispPhases.OrderByDescending(x => x.Value[1] + x.Value[2] + x.Value[3] + x.Value[4] + x.Value[5]).Take(14))
            Console.Error.WriteLine($"[DBG][DISPPH]   0x{h:X16} n={v[0]} us/op: program={v[1] / f / v[0]:F0} pipeline={v[2] / f / v[0]:F0} prepare={v[3] / f / v[0]:F0} bind={v[4] / f / v[0]:F0} record={v[5] / f / v[0]:F0} bufs={v[6]} imgs={v[7]}");
        _dbgDispPhases.Clear();
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
    public ComputeImageClear? TryDecodeImageClear(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator) =>
        TryDecodeUserDataFill(input, groupsX, groupsY, groupsZ, dispatchInitiator) ??
        TryDecodeLoadedValueFill(input, groupsX, groupsY, groupsZ) ??
        TryDecodeConstantFill(input, groupsX, groupsY, groupsZ);

    // AGC's constant fill kernel: one formatted dword buffer written at every thread index with a
    // value the program moves from a constant (a DCC or CMask clear code).
    private ComputeImageClear? TryDecodeConstantFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program ?? throw _host.Fatal("The compute stage has no program.");
        var resources = input.Stage.Resources;
        if (program.ConstantStoreValue is not { } value ||
            program.Buffers.Length != 1 || resources.Buffers.Length != 1 || resources.Buffers[0].Length != 4 ||
            program.Images.Length != 0 || program.SamplerCount != 0 || program.UsesDeviceAddresses)
        {
            return null;
        }

        var info = program.Buffers[0];
        var descriptor = BufferDescriptorWords.From(resources.Buffers[0]);
        var threads = (ulong)groupsX * input.ThreadsX;
        if (!info.Formatted || !info.Written || info.Read || info.Atomic || info.Scalar || info.MaxByteExtent != sizeof(uint) ||
            descriptor.Stride != sizeof(uint) || descriptor.Format != BufferDescriptorWords.Format32UInt || descriptor.SwizzleEnabled ||
            descriptor.IndexStride != 0 || descriptor.AddThreadId || descriptor.RecordCount == 0 ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || input.WaveSize != ImageClearWaveSize ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1 || threads != descriptor.RecordCount ||
            !input.GroupIdX || input.GroupIdY || input.GroupIdZ || input.ThreadIdCount != 1 ||
            (input.DispatchThreadDimensions && input.DispatchThreadsX != threads))
        {
            return null;
        }

        var size = descriptor.Footprint() ?? throw _host.Fatal($"The compute buffer footprint overflows: stride={descriptor.Stride} records={descriptor.RecordCount}.");
        return new ComputeImageClear(descriptor, value, size);
    }

    // The fill shader AGC titles use for DCC and CMask clears: each thread stores one dword it loaded
    // with s_buffer_load_dword from a one-record constant buffer, at its global thread index.
    private ComputeImageClear? TryDecodeLoadedValueFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ)
    {
        var program = input.Stage.Program ?? throw _host.Fatal("The compute stage has no program.");
        var resources = input.Stage.Resources;
        if (program.Buffers.Length != 2 || resources.Buffers.Length != 2 || program.Images.Length != 0 || program.SamplerCount != 0 ||
            program.UsesDeviceAddresses || resources.Images.Length != 0 || resources.Samplers.Length != 0 ||
            resources.Buffers[0].Length != 4 || resources.Buffers[1].Length != 4)
        {
            return null;
        }

        int source = -1, target = -1;
        for (var index = 0; index < 2; index++)
        {
            var info = program.Buffers[index];
            if (info.Scalar && info.Read && !info.Written && !info.Atomic && info.MaxByteExtent == sizeof(uint))
            {
                source = index;
            }
            else if (info.Formatted && info.Written && !info.Read && !info.Atomic && !info.Scalar && info.MaxByteExtent == sizeof(uint))
            {
                target = index;
            }
        }

        if (source < 0 || target < 0)
        {
            return null;
        }

        var descriptor = BufferDescriptorWords.From(resources.Buffers[target]);
        var valueDescriptor = BufferDescriptorWords.From(resources.Buffers[source]);
        var threads = (ulong)groupsX * input.ThreadsX;
        if (descriptor.Stride != sizeof(uint) || descriptor.Format != BufferDescriptorWords.Format32UInt || descriptor.SwizzleEnabled ||
            descriptor.IndexStride != 0 || descriptor.AddThreadId || descriptor.RecordCount == 0 ||
            input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 || input.WaveSize != ImageClearWaveSize ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1 || threads != descriptor.RecordCount ||
            !input.GroupIdX || input.GroupIdY || input.GroupIdZ || input.ThreadIdCount != 1 ||
            (input.DispatchThreadDimensions && input.DispatchThreadsX != threads) ||
            valueDescriptor.Address == 0)
        {
            return null;
        }

        Span<byte> value = stackalloc byte[sizeof(uint)];
        if (!_host.TryReadGuest(valueDescriptor.Address, value))
        {
            return null;
        }

        var size = descriptor.Footprint() ?? throw _host.Fatal($"The compute buffer footprint overflows: stride={descriptor.Stride} records={descriptor.RecordCount}.");
        return new ComputeImageClear(descriptor, System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(value), size);
    }

    private ComputeImageClear? TryDecodeUserDataFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
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

    private const uint Format32SInt = 21;
    private const uint Format32Float = 22;
    private const ulong DescriptorAddressMask = 0x0000_FFFF_FFFF_FFFFul;

    private bool TryConsumeBoundedFill(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program!;
        if (program.BoundedFill is not { } fill)
        {
            return false;
        }

        if (program.UserDataBase != 0 || fill.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadIdCount < 1 || input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1)
        {
            return RefuseBoundedFill(program,
                $"shape group=s{fill.GroupScalarRegister}/s{input.WorkgroupRegister} ids={input.GroupIdX}{input.GroupIdY}{input.GroupIdZ} tid={input.ThreadIdCount} " +
                $"local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ} dispatch={groupsX}x{groupsY}x{groupsZ} base={program.UserDataBase}");
        }

        var userData = input.Stage.Resources.UserData;
        Span<uint> destinationWords = stackalloc uint[4];
        for (var word = 0; word < destinationWords.Length; word++)
        {
            if (!TryResolveFillWord(fill.Destination[word], userData, out destinationWords[word]))
            {
                return RefuseBoundedFill(program, $"destination word {word} unreadable");
            }
        }

        var destination = BufferDescriptorWords.From(destinationWords);
        uint start = 0;
        uint value;
        if (!TryResolveFillWord(fill.Count, userData, out var count) ||
            (fill.Start is { } startWord && !TryResolveFillWord(startWord, userData, out start)))
        {
            return RefuseBoundedFill(program, "range unreadable");
        }

        if (fill.ConstantValue is { } constant)
        {
            value = constant;
        }
        else if (fill.Value is not { } valueWord || !TryResolveFillWord(valueWord, userData, out value))
        {
            return RefuseBoundedFill(program, "value unreadable");
        }

        if (fill.PatternLength is { } lengthWord)
        {
            if (fill.Pattern is not { } pattern || !TryResolveFillWord(lengthWord, userData, out var length) || length == 0 || length > pattern.Length)
            {
                return RefuseBoundedFill(program, "pattern length");
            }

            for (var word = 1; word < length; word++)
            {
                if (!TryResolveFillWord(pattern[word], userData, out var repeated) || repeated != value)
                {
                    return RefuseBoundedFill(program, $"pattern length={length} varies");
                }
            }
        }

        if (destination.Stride != sizeof(uint) || destination.SwizzleEnabled || destination.AddThreadId || destination.OutOfBounds != 0 ||
            destination.Type != 0 || (destination.Address & 3) != 0 ||
            (fill.Formatted && destination.Format is not (Format32UInt or Format32SInt or Format32Float)))
        {
            return RefuseBoundedFill(program,
                $"descriptor stride={destination.Stride} swizzle={destination.SwizzleEnabled} tid={destination.AddThreadId} oob={destination.OutOfBounds} " +
                $"type={destination.Type} format={destination.Format} address=0x{destination.Address:X}");
        }

        var threads = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0 ? groupsX : (ulong)groupsX * input.ThreadsX;
        var written = Math.Min(count, threads);
        if (threads > uint.MaxValue || (ulong)start + written > uint.MaxValue)
        {
            return RefuseBoundedFill(program, $"overflow threads={threads} start={start} count={count}");
        }

        var end = Math.Min((ulong)start + written, destination.RecordCount);
        if (end <= start)
        {
            return false;
        }

        var address = destination.Address + (ulong)start * sizeof(uint);
        var size = (end - start) * sizeof(uint);
        var consumed = _host.TryFillDccMetadata(address, size, value);
        if (RenderTrace.Enabled && RenderTrace.MetadataClear())
        {
            RenderTrace.Write($"Bounded fill: shader=0x{program.Hash:X16} address=0x{address:X16} size=0x{size:X} value=0x{value:X8} consumed={consumed}");
        }

        return consumed;
    }

    private bool TryConsumeBoundedCopy(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        var program = input.Stage.Program!;
        if (program.BoundedCopy is not { } copy || !BoundedCopyEnabled)
        {
            return false;
        }

        if (program.UserDataBase != 0 || copy.GroupScalarRegister != (uint)input.WorkgroupRegister || !input.GroupIdX || input.GroupIdY || input.GroupIdZ ||
            input.ThreadIdCount < 1 || input.ThreadsX != ImageClearWaveSize || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            groupsX == 0 || groupsY != 1 || groupsZ != 1)
        {
            return RefuseBoundedFill(program, $"copy shape dispatch={groupsX}x{groupsY}x{groupsZ} local={input.ThreadsX}x{input.ThreadsY}x{input.ThreadsZ}");
        }

        var userData = input.Stage.Resources.UserData;
        Span<uint> sourceWords = stackalloc uint[4];
        Span<uint> destinationWords = stackalloc uint[4];
        for (var word = 0; word < 4; word++)
        {
            if (!TryResolveFillWord(copy.Source[word], userData, out sourceWords[word]) ||
                !TryResolveFillWord(copy.Destination[word], userData, out destinationWords[word]))
            {
                return RefuseBoundedFill(program, $"copy descriptor word {word} unreadable");
            }
        }

        if (!TryResolveFillWord(copy.Count, userData, out var count) || !TryResolveFillWord(copy.Modulus, userData, out var modulus))
        {
            return RefuseBoundedFill(program, "copy range unreadable");
        }

        var source = BufferDescriptorWords.From(sourceWords);
        var destination = BufferDescriptorWords.From(destinationWords);
        if (!IsPlainWordBuffer(source) || !IsPlainWordBuffer(destination) || source.Format != destination.Format ||
            modulus == 0 || modulus > source.RecordCount)
        {
            return RefuseBoundedFill(program,
                $"copy src=0x{source.Address:X}/{source.Format}/{source.Stride}/{source.OutOfBounds} dst=0x{destination.Address:X}/{destination.Format}/{destination.Stride}/{destination.OutOfBounds} " +
                $"modulus={modulus} records={source.RecordCount}");
        }

        var threads = (dispatchInitiator & DispatchInitiatorUseThreadDimensions) != 0 ? groupsX : (ulong)groupsX * input.ThreadsX;
        var words = Math.Min(Math.Min((ulong)count, threads), destination.RecordCount);
        if (words == 0)
        {
            return false;
        }

        var consumed = _host.TryCopyWordsOnHost(destination.Address, source.Address, modulus, words);
        if (RenderTrace.Enabled)
        {
            RenderTrace.Write(
                $"Bounded copy: shader=0x{program.Hash:X16} src=0x{source.Address:X16} dst=0x{destination.Address:X16} words={words} modulus={modulus} consumed={consumed}");
        }

        return consumed || RefuseBoundedFill(program, $"copy refused by the host dst=0x{destination.Address:X} words={words} modulus={modulus}");
    }

    private static readonly bool BoundedCopyEnabled = Environment.GetEnvironmentVariable("SHARPEMU_HOST_BOUNDED_COPY") != "0";

    private static bool IsPlainWordBuffer(BufferDescriptorWords descriptor) =>
        descriptor.Stride == sizeof(uint) && !descriptor.SwizzleEnabled && !descriptor.AddThreadId && descriptor.OutOfBounds == 0 &&
        descriptor.Type == 0 && (descriptor.Address & 3) == 0 && descriptor.Format is Format32UInt or Format32SInt or Format32Float;

    private static bool RefuseBoundedFill(ShaderProgramInfo program, string reason)
    {
        Diagnostics.DccWriterTrace.Refuse(program.Hash, reason);
        return false;
    }

    private bool TryResolveFillWord(Pipelines.FillWord word, ReadOnlySpan<uint> userData, out uint value)
    {
        value = 0;
        switch (word.Source)
        {
            case Pipelines.FillWordSource.UserData:
                if (word.Register >= userData.Length)
                {
                    return false;
                }

                value = userData[(int)word.Register];
                return true;
            case Pipelines.FillWordSource.BufferResource:
            {
                if (word.Register + 4 > userData.Length)
                {
                    return false;
                }

                var resource = BufferDescriptorWords.From(userData.Slice((int)word.Register, 4));
                var size = resource.Stride == 0 ? resource.RecordCount : (ulong)resource.Stride * resource.RecordCount;
                var offset = (ulong)word.Offset & ~3ul;
                if (offset > size || size - offset < sizeof(uint))
                {
                    return true;
                }

                return TryReadFillWord((resource.Address & ~3ul) + offset, out value);
            }
            case Pipelines.FillWordSource.Pointer:
                return TryReadPointer(word.Register, userData, out var pointer) && TryReadFillWord(pointer + (ulong)word.Offset, out value);
            case Pipelines.FillWordSource.IndirectPointer:
            {
                if (!TryReadPointer(word.Register, userData, out var outer) ||
                    !TryReadFillWord(outer + (ulong)word.PointerOffset, out var low) ||
                    !TryReadFillWord(outer + (ulong)word.PointerOffset + sizeof(uint), out var high))
                {
                    return false;
                }

                var inner = ((low | ((ulong)high << 32)) & DescriptorAddressMask) & ~3ul;
                return TryReadFillWord(inner + (ulong)word.Offset, out value);
            }
            default:
                return false;
        }
    }

    private static bool TryReadPointer(uint register, ReadOnlySpan<uint> userData, out ulong pointer)
    {
        pointer = 0;
        if (register + 2 > userData.Length)
        {
            return false;
        }

        pointer = ((userData[(int)register] | ((ulong)userData[(int)register + 1] << 32)) & DescriptorAddressMask) & ~3ul;
        return pointer != 0;
    }

    private bool TryReadFillWord(ulong address, out uint value)
    {
        value = 0;
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (address == 0 || !_host.TryReadGuest(address, bytes))
        {
            return false;
        }

        value = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }
    // TEMP: SHARPEMU_DBG_CLEARS=1 reports how clear/fill dispatches were handled.
    private static readonly bool _dbgClears = Environment.GetEnvironmentVariable("SHARPEMU_DBG_CLEARS") == "1";
    private static readonly Dictionary<string, int> _dbgClearCounts = new();
    private static long _dbgClearLast = Environment.TickCount64;
    private static void DbgClear(string outcome, ulong hash, ulong address, ulong size, uint value)
    {
        if (!_dbgClears || outcome == "nodecode") return;
        var key = $"{outcome} hash=0x{hash:X} addr=0x{address:X} size=0x{size:X} value=0x{value:X8}";
        _dbgClearCounts.TryGetValue(key, out var count);
        _dbgClearCounts[key] = count + 1;
        if (Environment.TickCount64 - _dbgClearLast < 10000) return;
        _dbgClearLast = Environment.TickCount64;
        foreach (var (k, v) in _dbgClearCounts.OrderByDescending(x => x.Value).Take(14))
            Console.Error.WriteLine($"[DBG][CLEARS] {v} {k}");
        _dbgClearCounts.Clear();
    }

    private static bool IsDccFillPattern(uint value) =>
        value == (value & 0xFFu) * 0x01010101u && value is 0x00000000u or 0x20202020u or 0x40404040u or 0x80808080u or 0xC0C0C0C0u;

    private bool TryConsumeImageClear(ComputeInputInfo input, uint groupsX, uint groupsY, uint groupsZ, uint dispatchInitiator)
    {
        if (TryDecodeImageClear(input, groupsX, groupsY, groupsZ, dispatchInitiator) is not { } clear)
        {
            DbgClear("nodecode", input.Stage.Program!.Hash, 0, 0, 0); // TEMP
            return false;
        }

        var address = clear.Descriptor.Address;
        var hash = input.Stage.Program!.Hash;
        if (!_host.TryClearImageFromBuffer(address, clear.Size, clear.PackedClear))
        {
            // A metadata fill may run before its target is bound; the store keeps it pending and the dispatch runs.
            var registered = _host.TryAbsorbDccFill(address, clear.Size, clear.PackedClear);
            // A DCC-code fill of metadata that is still pending its target is applied on the CPU: the
            // dispatch would leave the bytes GPU-dirty and the target's discovery would wait for them.
            if (!registered && IsDccFillPattern(clear.PackedClear) && _host.TryFillGuestMemoryOnCpu(address, clear.Size, clear.PackedClear))
            {
                DbgClear("cpu-fill", hash, address, clear.Size, clear.PackedClear); // TEMP
                return true;
            }

            DbgClear(registered ? "absorbed" : "pending-run", hash, address, clear.Size, clear.PackedClear); // TEMP
            if (RenderTrace.Enabled && RenderTrace.MetadataClear())
            {
                RenderTrace.Write(
                    $"{(registered ? "Tracked" : "Deferred")} a metadata clear: shader=0x{hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
            }

            return registered;
        }

        DbgClear("cleared", hash, address, clear.Size, clear.PackedClear); // TEMP
        if (RenderTrace.Enabled && RenderTrace.ImageClear())
        {
            RenderTrace.Write($"Consumed a compute image clear: shader=0x{hash:X16} address=0x{address:X16} size=0x{clear.Size:X16} value=0x{clear.PackedClear:X8}");
        }

        return true;
    }
}
