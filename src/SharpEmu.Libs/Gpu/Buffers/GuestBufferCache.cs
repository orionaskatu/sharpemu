// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Kernel;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Libs.Gpu.Vulkan;
using Silk.NET.Vulkan;

namespace SharpEmu.Libs.Gpu.Buffers;

public readonly record struct DownloadPiece(GpuBuffer Buffer, ulong SourceOffset, ulong Address, ulong Size);

public readonly record struct OverlapSpan(int First, int Last, ulong Begin, ulong End, bool HasStreamLeap);

// Guest memory mirrored in device buffers: upload on use, download on CPU fault, page table for BDA.
public sealed unsafe class GuestBufferCache : IGuestBufferStore, IDisposable
{
    public const int CachingPageBits = 14;
    public const ulong CachingPageSize = 1UL << CachingPageBits;
    public const ulong CachingPageCount = 1UL << (40 - CachingPageBits);
    public const ulong BdaPageTableSize = CachingPageCount * sizeof(ulong);
    public static readonly ResourceSlotIdentifier NullBufferId = new(0, 1);

    private const ulong MiB = 1024 * 1024;
    private const ulong GdsBufferSize = 64 * 1024;
    private readonly record struct PlannedDownload(GpuBuffer Buffer, ulong Address, BufferDownloadPlacement Placement);

    private enum ShutdownOutcome
    {
        Pending,
        Drained,
        Failed,
    }

    private readonly GpuDeviceInfo _device;
    private readonly SubmissionScheduler _scheduler;
    private readonly IGpuQueueRelay _relay;
    private readonly GuestBufferUploader _uploader;
    private readonly IGuestBackedSpace _backing;
    private readonly ICpuMemory _guest;

    // False for addresses the guest never mapped. A shader that follows a bad pointer there
    // reads zeros, as on the console; caching that memory only fills VRAM with junk buffers.
    internal bool IsGuestMemoryMapped(ulong guestAddress, ulong size) => _guest.CanRead(guestAddress, size);
    private readonly BdaFaultProcessor _faults;
    private readonly GpuBuffer _gds;
    private readonly GpuBuffer _bdaPageTable;
    private readonly GuestBufferRegistry<GpuBuffer> _registry = new(CachingPageSize, PageOwnerTable.AddressSpaceSize);
    private readonly SpanSet _gpuModifiedRanges = new();
    private readonly GuestPageTracker _tracker;
    private readonly GpuRingBuffer _staging;
    private readonly GpuRingBuffer _stream;
    private readonly GpuRingBuffer _download;
    private readonly GpuRingBuffer _deviceRing;
    private readonly object _shutdownGate = new();
    private readonly BufferRetirementPolicy _retirementPolicy = new();
    private ShutdownOutcome _outcome;
    private bool _faultProcessPending;
    private bool _disposed;

    public GuestBufferCache(
        GpuDeviceInfo device,
        SubmissionScheduler scheduler,
        IGpuQueueRelay relay,
        PageGuard pages,
        ICpuMemory guest,
        IGuestBackedSpace backing)
    {
        _device = device;
        _scheduler = scheduler;
        _relay = relay;
        _backing = backing;
        _guest = guest;
        _faults = new BdaFaultProcessor(device, scheduler, this, CachingPageBits, CachingPageCount);
        _gds = new GpuBuffer(device, scheduler, GpuBufferUsage.Stream, 0, GpuBuffer.AllFlags, GdsBufferSize);
        _bdaPageTable = new GpuBuffer(device, scheduler, GpuBufferUsage.DeviceLocal, 0, GpuBuffer.AllFlags, BdaPageTableSize);
        _tracker = new GuestPageTracker(pages);
        _staging = new GpuRingBuffer(device, scheduler, GpuBufferUsage.Upload, 512 * MiB);
        _uploader = new GuestBufferUploader(device, scheduler, guest, _staging);
        _stream = new GpuRingBuffer(device, scheduler, GpuBufferUsage.Stream, 64 * MiB);
        _download = new GpuRingBuffer(device, scheduler, GpuBufferUsage.Download, 32 * MiB);
        _deviceRing = new GpuRingBuffer(device, scheduler, GpuBufferUsage.DeviceLocal, 128 * MiB);
        StreamOffsetAlignment = device.MinUniformBufferOffsetAlignment;
        _gds.Mapped.Clear();
        _gds.Flush(0, _gds.Size);
        var nullId = _registry.AllocateBuffer(new GpuBuffer(device, scheduler, GpuBufferUsage.DeviceLocal, 0, GpuBuffer.AllFlags, 16), 0, 16);
        if (nullId != NullBufferId)
        {
            throw SubmissionScheduler.Fatal("The null buffer occupies the wrong slot.");
        }

        _gpuMappingCreated = (address, size) => _pendingGpuMappings.Enqueue(new GuestSpan(address, size));
        GuestGpuMemoryHook.GpuMappingCreated += _gpuMappingCreated;
    }

    // Small GPU-accessible mappings announced by the kernel, mapped before the next device-address use.
    private readonly System.Collections.Concurrent.ConcurrentQueue<GuestSpan> _pendingGpuMappings = new();
    private readonly Action<ulong, ulong> _gpuMappingCreated;

    // Maps the announced mappings as a fault would, before the GPU first reaches them.
    private void MapPendingGpuMappings()
    {
        while (_pendingGpuMappings.TryDequeue(out var span))
        {
            if (!KernelMemoryCompatExports.TryGetMappedRange(span.Address, out var mappingStart, out var mappingLength) ||
                span.Address + span.Size > mappingStart + mappingLength)
            {
                continue;
            }

            NoteDeviceAddressFault(span.Address, span.Size, insideGuestMapping: true);
            _ = FindBuffer(span.Address, span.Size);
        }
    }

    public IGuestImageCache? ImageCache { get; set; }

    public GpuBuffer GdsBuffer => _gds;

    public GpuBuffer BdaPageTableBuffer => _bdaPageTable;

    public GpuBuffer FaultBuffer => _faults.FaultBuffer;

    public ulong TotalUsedMemory => _registry.RegisteredBytes + _registry.RetiredBytes;

    public bool RetirementOverBudget => _registry.RetiredBytes > 256UL * MiB;

    public int BufferCount => _registry.RegisteredCount;

    // The stream fast path aligns to this; the presenter raises it to its descriptor alignment.
    public ulong StreamOffsetAlignment { get; set; }

    public void ForEachBuffer(Action<GpuBuffer> visit)
    {
        for (var index = 0; index < _registry.RegisteredCount; index++)
        {
            visit(_registry.GetBuffer(_registry.GetRegisteredIdentifier(index)));
        }
    }

    public GpuBuffer GetBuffer(ResourceSlotIdentifier bufferIdentifier) => _registry.GetBuffer(bufferIdentifier);

    public GpuRingBuffer GetUtilityBuffer(GpuBufferUsage usage) => usage switch
    {
        GpuBufferUsage.Upload => _staging,
        GpuBufferUsage.Stream => _stream,
        GpuBufferUsage.Download => _download,
        GpuBufferUsage.DeviceLocal => _deviceRing,
        _ => throw SubmissionScheduler.Fatal("The utility buffer usage is invalid."),
    };

    // A CPU write fault: true when the range is tracked and any GPU data reached guest memory.
    bool IGuestBufferStore.MarkCpuWrite(ulong address, ulong size)
    {
        var tracked = _tracker.InvalidateRegion(address, size, out var needsGpuFlush);
        var completed = !needsGpuFlush || ReadMemoryOrAwaitShutdown(address, size, isWrite: true,
            GuestMemoryProfile.ReadbackSource.CpuWriteInvalidation);
        if (GuestGpuMemoryHook.Traces(address, size))
            GuestGpuMemoryHook.Trace(address, size, $"buffer-write tracked={tracked} completed={completed}");
        return tracked && completed;
    }

    private const int MaxWriteThroughBytes = 64;

    // A small command-processor write (a label, WRITE_DATA) into a page the GPU also wrote
    // would download the whole page first. When no buffer write covered these bytes, guest
    // memory and the cached buffer can both take the new value instead: the copy is ordered
    // after the earlier GPU work, and the page's other GPU bytes stay where they are.
    public bool TryWriteThrough(ulong address, ReadOnlySpan<byte> data)
    {
        var size = (ulong)data.Length;
        if (size == 0 || size > MaxWriteThroughBytes || (size & 3) != 0 || (address & 3) != 0 ||
            !IsValidRange(address, size) || _scheduler.Current.IsInvalid)
        {
            return false;
        }

        if (!_tracker.MayHaveGpuDirtyPages(address, size) || !_tracker.HasGpuDirtyPages(address, size) ||
            _gpuModifiedRanges.Overlaps(address, size))
        {
            return false;
        }

        var owner = _registry.FindContainingBuffer(address, size);
        if (!owner.IsValid)
        {
            return false;
        }

        var buffer = _registry.GetBuffer(owner);
        if (!_backing.TryWriteBacking(address, data))
        {
            return false;
        }

        var command = _scheduler.Current;
        command.EndRendering();
        var native = new CommandBuffer(command.Handle);
        var vk = _device.Vk;
        var barrier = new BufferMemoryBarrier2
        {
            SType = StructureType.BufferMemoryBarrier2,
            SrcAccessMask = AccessFlags2.MemoryReadBit | AccessFlags2.MemoryWriteBit,
            DstAccessMask = AccessFlags2.TransferWriteBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = buffer.Handle,
            Offset = buffer.Offset(address),
            Size = size,
        };
        VulkanSynchronization.PipelineBarrier(vk, native, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit,
            DependencyFlags.None, 0, null, 1, &barrier, 0, null);
        fixed (byte* pointer = data)
        {
            vk.CmdUpdateBuffer(native, buffer.Handle, buffer.Offset(address), size, pointer);
        }

        var after = barrier;
        after.SrcAccessMask = AccessFlags2.TransferWriteBit;
        after.DstAccessMask = AccessFlags2.MemoryReadBit | AccessFlags2.MemoryWriteBit;
        VulkanSynchronization.PipelineBarrier(vk, native, PipelineStageFlags.TransferBit, PipelineStageFlags.AllCommandsBit,
            DependencyFlags.None, 0, null, 1, &after, 0, null);
        return true;
    }

    // A dword only a recorded GPU write holds: the buffer that owns it is newer than guest memory.
    public bool IsGpuOwnedWord(ulong address) =>
        (address & 3) == 0 && IsValidRange(address, sizeof(uint)) &&
        _tracker.MayHaveGpuDirtyPages(address, sizeof(uint)) && _tracker.HasGpuDirtyPages(address, sizeof(uint)) &&
        _gpuModifiedRanges.Overlaps(address, sizeof(uint)) && !_tracker.HasCpuDirtyPages(address, sizeof(uint)) &&
        _registry.FindContainingBuffer(address, sizeof(uint)).IsValid;

    // Copies GPU-owned guest bytes into another buffer in queue order, so work recorded after
    // this sees the value earlier work wrote without the CPU waiting for it.
    public bool TryCopyGpuOwned(ulong address, ulong size, ulong destination, ulong destinationOffset)
    {
        if (size == 0 || _scheduler.Current.IsInvalid || !IsValidRange(address, size) ||
            !_gpuModifiedRanges.Overlaps(address, size) || _tracker.HasCpuDirtyPages(address, size))
        {
            return false;
        }

        var owner = _registry.FindContainingBuffer(address, size);
        if (!owner.IsValid)
        {
            return false;
        }

        var buffer = _registry.GetBuffer(owner);
        var command = _scheduler.Current;
        command.EndRendering();
        var native = new CommandBuffer(command.Handle);
        var vk = _device.Vk;
        var barriers = stackalloc BufferMemoryBarrier2[2];
        barriers[0] = new BufferMemoryBarrier2
        {
            SType = StructureType.BufferMemoryBarrier2,
            SrcAccessMask = AccessFlags2.MemoryWriteBit,
            DstAccessMask = AccessFlags2.TransferReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = buffer.Handle,
            Offset = buffer.Offset(address),
            Size = size,
        };
        barriers[1] = barriers[0] with
        {
            SrcAccessMask = AccessFlags2.MemoryReadBit | AccessFlags2.MemoryWriteBit,
            DstAccessMask = AccessFlags2.TransferWriteBit,
            Buffer = new Silk.NET.Vulkan.Buffer(destination),
            Offset = destinationOffset,
        };
        VulkanSynchronization.PipelineBarrier(vk, native, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit,
            DependencyFlags.None, 0, null, 2, barriers, 0, null);
        var copy = new BufferCopy(buffer.Offset(address), destinationOffset, size);
        vk.CmdCopyBuffer(native, buffer.Handle, new Silk.NET.Vulkan.Buffer(destination), 1, &copy);
        var after = barriers[1] with
        {
            SrcAccessMask = AccessFlags2.TransferWriteBit,
            DstAccessMask = AccessFlags2.MemoryReadBit,
        };
        VulkanSynchronization.PipelineBarrier(vk, native, PipelineStageFlags.TransferBit, PipelineStageFlags.AllCommandsBit,
            DependencyFlags.None, 0, null, 1, &after, 0, null);
        return true;
    }

    // TEMP: reads a dword straight from the GPU buffer that contains it, ignoring dirty tracking.
    public bool DbgReadGpuWord(ulong address, out uint word, out ulong bufferBase, out bool gpuWritten)
    {
        word = 0;
        bufferBase = 0;
        gpuWritten = false;
        var owner = _registry.FindContainingBuffer(address, sizeof(uint));
        if (!owner.IsValid || AsyncReadback is not { } readback)
            return false;
        var buffer = _registry.GetBuffer(owner);
        bufferBase = buffer.CpuAddress;
        gpuWritten = buffer.LastGpuWriteTick != 0;
        if (!gpuWritten)
            return false;
        if (buffer.LastGpuWriteTick >= _scheduler.CurrentTick)
            _scheduler.Flush();
        uint result = 0;
        readback.Read([new Vulkan.ReadbackPiece(buffer, buffer.Offset(address), sizeof(uint))], buffer.LastGpuWriteTick,
            (_, bytes) => result = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes));
        word = result;
        return true;
    }

    public bool TrySynchronizeCpuRead(ulong address, ulong size) =>
        TrySynchronizeCpuRead(address, size, GuestMemoryProfile.ReadbackSource.CpuReadSynchronization);

    public bool TrySynchronizeCpuRead(ulong address, ulong size, GuestMemoryProfile.ReadbackSource source) =>
        !_tracker.MayHaveGpuDirtyPages(address, size) ||
        !_tracker.HasGpuDirtyPages(address, size) ||
        ReadMemoryOrAwaitShutdown(address, size, isWrite: false, source);

    // Lock-free on the GPU queue thread; see GuestPageTracker.MayHaveGpuDirtyPages.
    public bool MayHaveGpuDirtyPages(ulong address, ulong size) => _tracker.MayHaveGpuDirtyPages(address, size);

    // A CPU read fault: GPU-dirty pages download through the worker first.
    // TEMP
    private static readonly bool _dbgFaults = Environment.GetEnvironmentVariable("SHARPEMU_DBG_FAULTS") == "1";
    private static readonly Dictionary<ulong, (ulong Address, ulong Size, long Tick)> _dbgLastWrite = new();
    private static readonly Dictionary<string, (int Count, double AgeMs)> _dbgFaultStats = new();
    private static long _dbgFaultLast = Environment.TickCount64;
    internal void DbgFault(ulong address, string tag = "fault")
    {
        if (!_dbgFaults) return;
        string key;
        double age = -1;
        lock (_dbgLastWrite)
        {
            // Find the most recent write range that covers this address.
            (ulong Address, ulong Size, long Tick) best = default;
            foreach (var (_, w) in _dbgLastWrite)
                if (address >= w.Address && address < w.Address + w.Size && w.Tick > best.Tick) best = w;
            if (best.Tick != 0) age = System.Diagnostics.Stopwatch.GetElapsedTime(best.Tick).TotalMilliseconds;
            key = tag + " " + (best.Tick == 0 ? $"unknown page=0x{address >> 12:X}" : $"write=0x{best.Address:X}+0x{best.Size:X} off=0x{address - best.Address:X}");
        }
        lock (_dbgFaultStats)
        {
            _dbgFaultStats.TryGetValue(key, out var e);
            _dbgFaultStats[key] = (e.Count + 1, e.AgeMs + age);
            if (Environment.TickCount64 - _dbgFaultLast < 10000) return;
            _dbgFaultLast = Environment.TickCount64;
            Console.Error.WriteLine($"[DBG][FAULT] distinct={_dbgFaultStats.Count} total={_dbgFaultStats.Values.Sum(v => v.Count)}");
            foreach (var (k, v) in _dbgFaultStats.OrderByDescending(x => x.Value.Count).Take(25))
                Console.Error.WriteLine($"[DBG][FAULT] n={v.Count} avg_age_ms={v.AgeMs / v.Count:F1} {k}");
            _dbgFaultStats.Clear();
        }
    }

    private void DbgFaultKind(string kind, ulong address, ulong size) // TEMP
    {
        var inRange = _gpuModifiedRanges.Overlaps(address, size);
        var pageInRange = _gpuModifiedRanges.Overlaps(address & ~0xFFFul, 0x1000);
        var owner = _registry.FindContainingBuffer(address, size);
        var complete = owner.IsValid && _scheduler.IsTickComplete(_registry.GetBuffer(owner).LastGpuWriteTick);
        var key = $"{kind} bytes_gpu_written={inRange} page_has_gpu_bytes={pageInRange} writer_done={complete}";
        DbgStacks.Record(key, 0, size, depth: 0);
    }

    public bool DownloadToCpu(ulong address, ulong size)
    {
        var tracked = _tracker.HasRegion(address, size);
        var dirty = tracked && _tracker.HasGpuDirtyPages(address, size);

        var completed = tracked && (!dirty || ReadMemoryOrAwaitShutdown(address, size, isWrite: false,
            GuestMemoryProfile.ReadbackSource.StoreDownload));
        if (GuestGpuMemoryHook.Traces(address, size))
            GuestGpuMemoryHook.Trace(address, size, $"buffer-read tracked={tracked} gpu_dirty={dirty} completed={completed}");
        return completed;
    }

    public void InvalidateMemory(ulong guestAddress, ulong size)
    {
        if (!IsValidRange(guestAddress, size))
        {
            throw SubmissionScheduler.Fatal("The memory invalidation range is invalid.");
        }

        _tracker.InvalidateRegion(guestAddress, size, () => ReadMemory(guestAddress, size, isWrite: true));
    }

    public void ReadMemory(ulong guestAddress, ulong size, bool isWrite = false)
    {
        if (!ReadMemoryOrAwaitShutdown(guestAddress, size, isWrite))
        {
            throw SubmissionScheduler.Fatal($"Cannot download buffer data after a failed shutdown: addr=0x{guestAddress:X16} size=0x{size:X16}");
        }
    }

    public ResourceSlotIdentifier FindBuffer(ulong guestAddress, ulong size)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.BufferCacheLookup);
        if (guestAddress == 0)
        {
            return NullBufferId;
        }

        if (!IsValidRange(guestAddress, size))
        {
            throw SubmissionScheduler.Fatal("The buffer lookup range is invalid.");
        }

        var owner = _registry.FindContainingBuffer(guestAddress, size);
        if (owner.IsValid)
        {
            return owner;
        }

        return CreateBuffer(guestAddress, size);
    }

    // TEMP: SHARPEMU_DBG_DIRTY_RANGE=start:size prints who makes the range GPU-written.
    private static readonly (ulong Begin, ulong End) _dbgDirtyRange = ParseDbgDirtyRange();
    private static readonly Dictionary<string, int> _dbgDirtyStacks = new();

    private static (ulong, ulong) ParseDbgDirtyRange()
    {
        var value = Environment.GetEnvironmentVariable("SHARPEMU_DBG_DIRTY_RANGE");
        if (string.IsNullOrWhiteSpace(value)) return (0, 0);
        var parts = value.Split(':');
        var begin = Convert.ToUInt64(parts[0].Replace("0x", ""), 16);
        return (begin, begin + Convert.ToUInt64(parts[1].Replace("0x", ""), 16));
    }

    internal static void DbgNoteWrite(ulong address, ulong size)
    {
        if (_dbgDirtyRange.End == 0 || address >= _dbgDirtyRange.End || address + size <= _dbgDirtyRange.Begin) return;
        var frames = new System.Diagnostics.StackTrace(1, false).GetFrames();
        var key = string.Join(" < ", frames.Take(7).Select(frame => frame.GetMethod()?.Name ?? "?"));
        lock (_dbgDirtyStacks)
        {
            _dbgDirtyStacks.TryGetValue(key, out var count);
            _dbgDirtyStacks[key] = count + 1;
            if ((count & (count + 1)) == 0)
                Console.Error.WriteLine($"[DBG][DIRTY] n={count + 1} 0x{address:X}+0x{size:X} {key}");
        }
    }

    public (GpuBuffer Buffer, ulong Offset) ObtainBuffer(ulong guestAddress, ulong size, bool isWritten, bool isTexelBuffer = false, ResourceSlotIdentifier bufferIdentifier = default)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.BufferAcquisitionChecks);
        var command = _scheduler.Current;
        if (command.IsInvalid || !IsValidRange(guestAddress, size))
        {
            throw SubmissionScheduler.Fatal("A buffer request requires a command buffer that is recording.");
        }

        if (!isWritten &&
            !_tracker.HasGpuDirtyPages(guestAddress, size) &&
            _tracker.HasCpuDirtyPages(guestAddress, size) &&
            (size <= CachingPageSize || (!isTexelBuffer && _tracker.IsCpuWriteHotRange(guestAddress, size))))
        {
            using var streamProfile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.BufferStreamUpload);
            if (_stream.TryMap(size, out var streamOffset, StreamOffsetAlignment, allowWait: false) &&
                _backing.TryReadBacking(guestAddress, _stream.Mapped.Slice((int)streamOffset, (int)size)))
            {
                _stream.Commit();
                return (_stream, streamOffset);
            }
        }

        // A GPU write into memory without backing could never download later; refuse it now.
        if (isWritten && !_backing.IsBackedRange(guestAddress, size))
        {
            throw SubmissionScheduler.Fatal($"Could not write the required direct backing: addr=0x{guestAddress:X16} size=0x{size:X16}");
        }

        var buffer = _registry.TryGetRegisteredBuffer(bufferIdentifier);
        if (buffer == null || !buffer.IsInBounds(guestAddress, size))
        {
            bufferIdentifier = FindBuffer(guestAddress, size);
            buffer = _registry.GetBuffer(bufferIdentifier);
        }

        TouchBuffer(bufferIdentifier);
        // A persistent upload must track the next CPU write instead of uploading unchanged hot pages.
        // Streaming keeps hot pages writable; the fallback must restore protection before copying.
        _ = SynchronizeBuffer(buffer, guestAddress, size, isWritten, isTexelBuffer, preserveCpuWriteHotPages: false);
        if (isWritten)
        {
            buffer.NoteGpuWrite();
            _gpuModifiedRanges.Add(guestAddress, size);
            DbgNoteWrite(guestAddress, size); // TEMP
            if (_dbgFaults) lock (_dbgLastWrite) _dbgLastWrite[guestAddress >> 12] = (guestAddress, size, System.Diagnostics.Stopwatch.GetTimestamp());
        }

        return (buffer, buffer.Offset(guestAddress));
    }

    // A texture streamed into its address range maps only its resident mips: the mip tail
    // and small levels sit at the start of the chain and level 0 at its end. The GPU samples
    // only resident levels, so the unmapped pages of a whole-chain upload read as zero.
    private bool TryReadResidentImagePages(ulong guestAddress, Span<byte> destination)
    {
        const ulong page = 1UL << 12;
        var size = (ulong)destination.Length;
        if (size == 0 || !_backing.IsBackedRange(guestAddress, 1))
        {
            return false;
        }

        // Only a resident prefix qualifies: a hole between backed pages is not a mip chain.
        var resident = true;
        for (ulong offset = 0; offset < size;)
        {
            var chunk = Math.Min(page - ((guestAddress + offset) & (page - 1)), size - offset);
            var target = destination.Slice((int)offset, (int)chunk);
            var backed = _backing.IsBackedRange(guestAddress + offset, chunk);
            if (backed && !resident)
            {
                return false;
            }

            resident = backed && _backing.TryReadBacking(guestAddress + offset, target);
            if (!resident)
            {
                target.Clear();
            }

            offset += chunk;
        }

        return true;
    }

    public (GpuBuffer Buffer, ulong Offset) ObtainBufferForImage(ulong guestAddress, ulong size)
    {
        if (!IsValidRange(guestAddress, size))
        {
            throw SubmissionScheduler.Fatal("The image source range is invalid.");
        }

        var cpuModified = _tracker.HasCpuDirtyPages(guestAddress, size);
        var gpuModified = _tracker.HasGpuDirtyPages(guestAddress, size);
        var hasDirtyBufferSource = _gpuModifiedRanges.Overlaps(guestAddress, size);
        _tracker.ValidateGpuDirtyOwnership(_gpuModifiedRanges, guestAddress, size, "image source");

        var owner = FindOwner(guestAddress, size);
        if (hasDirtyBufferSource && owner == null)
        {
            if (!IsRegionRegistered(guestAddress, size))
            {
                throw SubmissionScheduler.Fatal("The GPU-dirty image source has no device buffer.");
            }

            owner = _registry.GetBuffer(FindBuffer(guestAddress, size));
        }

        if (owner != null && (!gpuModified || hasDirtyBufferSource))
        {
            TouchBuffer(owner);
            if (cpuModified)
            {
                // Tracking clears complete pages, even for a subpage image. The regular
                // uploader stages those complete runs; staging only the image would copy
                // unprepared bytes over neighbouring textures and mark them clean.
                _ = SynchronizeBuffer(owner, guestAddress, size, isWritten: false, isTexelBuffer: true,
                    preserveCpuWriteHotPages: false, readImageBacking: true);
            }
            return (owner, owner.Offset(guestAddress));
        }

        if (hasDirtyBufferSource && owner == null)
        {
            throw SubmissionScheduler.Fatal("Cannot find the device buffer that owns the GPU-dirty image source.");
        }

        if (!_staging.TryMap(size, out var stageOffset, 16))
        {
            throw SubmissionScheduler.Fatal(
                $"Cannot reserve image staging space: address=0x{guestAddress:X16} size=0x{size:X16} capacity=0x{_staging.Size:X16} tick={_scheduler.CurrentTick}.");
        }
        if (!_backing.TryReadBacking(guestAddress, _staging.Mapped.Slice((int)stageOffset, (int)size)) &&
            !KernelMemoryCompatExports.TryReadPrtBacking(_backing, guestAddress,
                _staging.Mapped.Slice((int)stageOffset, (int)size)) &&
            !TryReadResidentImagePages(guestAddress, _staging.Mapped.Slice((int)stageOffset, (int)size)))
        {
            throw SubmissionScheduler.Fatal(
                $"Could not read the mapped guest image backing: address=0x{guestAddress:X16} size=0x{size:X16} range_backed={_backing.IsBackedRange(guestAddress, size)} first_byte_backed={_backing.IsBackedRange(guestAddress, 1)} last_byte_backed={_backing.IsBackedRange(guestAddress + size - 1, 1)} tick={_scheduler.CurrentTick}.");
        }

        _staging.Commit();
        hasDirtyBufferSource = _gpuModifiedRanges.Overlaps(guestAddress, size);
        owner = FindOwner(guestAddress, size);
        if (hasDirtyBufferSource && owner == null)
        {
            throw SubmissionScheduler.Fatal("The GPU-dirty image source lost its device buffer owner.");
        }

        if (owner == null || (_tracker.HasGpuDirtyPages(guestAddress, size) && !hasDirtyBufferSource))
        {
            return (_staging, stageOffset);
        }

        TouchBuffer(owner);
        _ = SynchronizeBuffer(owner, guestAddress, size, isWritten: false, isTexelBuffer: true,
            preserveCpuWriteHotPages: false, readImageBacking: true);
        return (owner, owner.Offset(guestAddress));
    }

    // Image acquisition already holds the image-cache lock. Read CPU-owned backing
    // directly instead of faulting through guest memory and reentering that cache.
    private bool TryReadImageSource(ulong address, Span<byte> destination) =>
        _backing.TryReadBacking(address, destination) ||
        KernelMemoryCompatExports.TryReadPrtBacking(_backing, address, destination) ||
        TryReadResidentImagePages(address, destination);

    public void WriteHostMemory(ulong guestAddress, ReadOnlySpan<byte> data)
    {
        if (guestAddress == 0 || data.IsEmpty || (ulong)data.Length > ulong.MaxValue - guestAddress)
        {
            throw SubmissionScheduler.Fatal("The host DMA write range is invalid.");
        }

        if (!_backing.TryWriteBacking(guestAddress, data))
        {
            throw SubmissionScheduler.Fatal($"Could not write the required direct backing: addr=0x{guestAddress:X16} size=0x{data.Length:X16}");
        }

        var end = guestAddress + (ulong)data.Length;
        for (var index = 0; index < _registry.RegisteredCount; index++)
        {
            var address = _registry.GetRegisteredAddress(index);
            var bufferIdentifier = _registry.GetRegisteredIdentifier(index);
            var buffer = _registry.GetBuffer(bufferIdentifier);
            var begin = Math.Max(guestAddress, address);
            var rangeEnd = Math.Min(end, address + buffer.Size);
            if (begin >= rangeEnd)
            {
                continue;
            }

            WriteDataBuffer(buffer, begin, data.Slice((int)(begin - guestAddress), (int)(rangeEnd - begin)));
            TouchBuffer(bufferIdentifier);
        }
    }

    private static readonly bool DbgAllFills = Environment.GetEnvironmentVariable("SHARPEMU_DBG_ALL_FILLS") == "1"; // TEMP

    public void FillBuffer(ulong guestAddress, ulong size, uint value, bool isGds)
    {
        if ((guestAddress & 3) != 0 || size == 0 || (size & 3) != 0 || size > ulong.MaxValue - guestAddress)
        {
            throw SubmissionScheduler.Fatal("The fill range must be aligned to four bytes.");
        }

        if (isGds)
        {
            if (guestAddress > _gds.Size || size > _gds.Size - guestAddress)
            {
                throw SubmissionScheduler.Fatal("The GDS fill range is outside the buffer.");
            }

            _gds.Fill(guestAddress, size, value);
            return;
        }

        if (guestAddress == 0)
        {
            throw SubmissionScheduler.Fatal("The fill memory address is invalid.");
        }

        var images = RequireImageCache();
        if (DbgAllFills && size >= 0x1000) // TEMP
            Diagnostics.DbgTargetWatch.Log($"anyfill {guestAddress >> 16:X} {size:X} {value:X}", () => $"anyfill address=0x{guestAddress:X} size=0x{size:X} value=0x{value:X8}");
        if (Diagnostics.DbgTargetWatch.Enabled) // TEMP
        {
            bool watched; lock (Diagnostics.DbgTargetWatch.MetadataAddresses) watched = Diagnostics.DbgTargetWatch.MetadataAddresses.Contains(guestAddress);
            if (watched || Diagnostics.DbgTargetWatch.Addresses.Any(watch => guestAddress < watch + (32UL << 20) && watch < guestAddress + size))
                Diagnostics.DbgTargetWatch.Log($"dmafill {guestAddress:X} {size:X} {value:X}", () => $"dmafill address=0x{guestAddress:X} size=0x{size:X} value=0x{value:X8}");
        }
        _ = images.ClearMetadata(guestAddress);
        var region = images.QueryRegion(guestAddress, size);
        if (!HasGpuDirtyBytes(guestAddress, size) && !region.GpuImageBytes)
        {
            if (region.ImageBytes)
            {
                images.InvalidateMemory(guestAddress, size);
            }

            var values = new uint[4096];
            Array.Fill(values, value);
            var bytes = MemoryMarshal.AsBytes<uint>(values);
            for (ulong offset = 0; offset < size;)
            {
                var chunk = (int)Math.Min(size - offset, (ulong)bytes.Length);
                WriteHostMemory(guestAddress + offset, bytes[..chunk]);
                offset += (ulong)chunk;
            }

            return;
        }

        images.InvalidateMemoryFromGpu(guestAddress, size);
        var bufferIdentifier = FindBuffer(guestAddress, size);
        var (destination, destinationOffset) = ObtainBuffer(guestAddress, size, true, true, bufferIdentifier);
        destination.Fill(destinationOffset, size, value);
    }

    public void CopyBuffer(ulong dstVaddr, ulong srcVaddr, ulong size, bool dstGds, bool srcGds)
    {
        var dstMemory = !dstGds;
        var srcMemory = !srcGds;
        if ((dstMemory && dstVaddr == 0) || (srcMemory && srcVaddr == 0) || size == 0 ||
            ((dstGds || srcGds) && ((dstVaddr | srcVaddr | size) & 3) != 0) ||
            size > ulong.MaxValue - dstVaddr || size > ulong.MaxValue - srcVaddr || (dstGds && srcGds) ||
            (dstGds && (dstVaddr > _gds.Size || size > _gds.Size - dstVaddr)) ||
            (srcGds && (srcVaddr > _gds.Size || size > _gds.Size - srcVaddr)))
        {
            throw SubmissionScheduler.Fatal(
                $"The buffer copy range is invalid: src=0x{srcVaddr:X16} dst=0x{dstVaddr:X16} size=0x{size:X16} src_gds={(srcGds ? 1 : 0)} dst_gds={(dstGds ? 1 : 0)}");
        }

        var images = RequireImageCache();
        if (Diagnostics.DbgTargetWatch.Enabled) // TEMP
        {
            foreach (var watch in Diagnostics.DbgTargetWatch.Addresses)
            {
                if ((dstMemory && dstVaddr < watch + (32UL << 20) && watch < dstVaddr + size) || (srcMemory && srcVaddr < watch + (32UL << 20) && watch < srcVaddr + size))
                    Diagnostics.DbgTargetWatch.Log($"dmacopy {dstVaddr:X} {srcVaddr:X} {size:X}", () => $"dmacopy dst=0x{dstVaddr:X} src=0x{srcVaddr:X} size=0x{size:X} srcGpuDirty={HasGpuDirtyBytes(srcVaddr, size)} srcGpuImage={images.QueryRegion(srcVaddr, size).GpuImageBytes}");
            }
        }
        var srcRegion = srcMemory ? images.QueryRegion(srcVaddr, size) : default;
        var dstRegion = dstMemory ? images.QueryRegion(dstVaddr, size) : default;
        if (srcMemory && dstMemory && !HasGpuDirtyBytes(srcVaddr, size) && !HasGpuDirtyBytes(dstVaddr, size) &&
            !srcRegion.GpuImageBytes && !dstRegion.GpuImageBytes)
        {
            if (dstRegion.ImageBytes)
            {
                images.InvalidateMemory(dstVaddr, size);
            }

            var bytes = new byte[64 * 1024];
            for (ulong offset = 0; offset < size;)
            {
                var chunk = (int)Math.Min(size - offset, (ulong)bytes.Length);
                if (!_backing.TryReadBacking(srcVaddr + offset, bytes.AsSpan(0, chunk)))
                {
                    throw SubmissionScheduler.Fatal("The host DMA source has no direct backing.");
                }

                WriteHostMemory(dstVaddr + offset, bytes.AsSpan(0, chunk));
                offset += (ulong)chunk;
            }

            return;
        }

        var command = _scheduler.Current;
        if (dstMemory)
        {
            images.InvalidateMemoryFromGpu(dstVaddr, size);
        }

        var srcId = srcMemory ? FindBuffer(srcVaddr, size) : default;
        var dstId = dstMemory ? FindBuffer(dstVaddr, size) : default;
        var (src, srcOffset) = srcMemory ? ObtainBuffer(srcVaddr, size, false, true, srcId) : (_gds, srcVaddr);
        var (dst, dstOffset) = dstMemory ? ObtainBuffer(dstVaddr, size, true, true, dstId) : (_gds, dstVaddr);
        if (ReferenceEquals(src, dst) && srcOffset < dstOffset + size && dstOffset < srcOffset + size)
        {
            throw SubmissionScheduler.Fatal("The resolved Vulkan copy ranges overlap.");
        }

        dst.CopyFrom(command, src, srcOffset, dstOffset, size);
    }

    // Buffers are ordered and disjoint: the last one starting before the query end is the only candidate.
    public bool IsRegionRegistered(ulong guestAddress, ulong size)
    {
        if (!IsValidRange(guestAddress, size))
        {
            throw SubmissionScheduler.Fatal("The registered-region query is invalid.");
        }

        return _registry.HasOverlap(guestAddress, size);
    }

    public bool HasGpuDirtyPages(ulong guestAddress, ulong size) => _tracker.HasGpuDirtyPages(guestAddress, size);

    public bool HasGpuDirtyBytes(ulong guestAddress, ulong size) => _gpuModifiedRanges.Overlaps(guestAddress, size);

    public bool HasCpuDirtyPages(ulong guestAddress, ulong size) => _tracker.HasCpuDirtyPages(guestAddress, size);

    internal bool DbgIsCpuWriteHot(ulong guestAddress, ulong size) => _tracker.IsCpuWriteHotRange(guestAddress, size); // TEMP

    public void ProcessFaultBuffer() => _faults.ProcessFaultBuffer();

    // Uploads every mapped range before a BDA draw; the fault pass runs at the next collection.
    // Every device-address program prepares all GPU-mapped memory; visiting each registered
    // buffer per dispatch cost ~17 % of the Demon's Souls render thread. The same work is done
    // in two cheaper parts: the recency touch, a no-op after the first one in a retirement tick,
    // runs once per tick (and when the mapping changes; new buffers register with the current
    // tick), and uploads only visit buffers in blocks the tracker does not know to be clean.
    // Hot pages stay dirty and writable, so every preparation still re-uploads them.
    private ulong _bdaTouchTick = ulong.MaxValue;
    private ulong _bdaTouchMapping;

    // Ranges shaders reached through device addresses outside the GPU mappings: pointers
    // into ordinary guest memory, which the GPU reads too. Only a fault reveals them, so
    // they are kept and touched like the mappings. Otherwise their buffers age out while
    // still in use, fault again, and the cache collects and re-creates them every frame.
    private readonly SpanSet _deviceAddressFaultSpans = new();

    // Fault ranges inside a known guest mapping; they are forgotten once it is unmapped.
    private readonly SpanSet _mappedDeviceAddressFaultSpans = new();

    // A page a shader wrote through a device address after the submission ran: the buffer
    // that holds it becomes GPU-written there, so the guest CPU downloads the result when it
    // reads, as it does for the writes planned before the draw.
    internal void NoteDeviceAddressWrite(ulong pageAddress, ulong pageSize)
    {
        var owner = _registry.FindContainingBuffer(pageAddress, pageSize);
        if (!owner.IsValid)
        {
            return;
        }

        var buffer = _registry.GetBuffer(owner);
        buffer.NoteGpuWrite();
        // A tracker page the guest CPU wrote since keeps its newer CPU data.
        _tracker.MarkGpuDirtyPagesWhereCpuClean(pageAddress, pageSize, _noteGpuModifiedRange ??= (address, size) => _gpuModifiedRanges.Add(address, size));
        if (Diagnostics.DbgTargetWatch.Enabled && Diagnostics.DbgTargetWatch.Addresses.Any(watch => pageAddress >= watch && pageAddress < watch + 0x800000)) // TEMP
            Diagnostics.DbgTargetWatch.Log($"devwrite {pageAddress >> 20:X}", () => $"devwrite page=0x{pageAddress:X} tick={_scheduler.CurrentTick}");
        if (DbgWriteLog && Interlocked.Increment(ref _dbgWrittenPages) <= 200) // TEMP
            Console.Error.WriteLine($"[DBG][DEVWRITTEN] page=0x{pageAddress:X}");
    }

    private Action<ulong, ulong>? _noteGpuModifiedRange;

    private static readonly bool DbgWriteLog = Environment.GetEnvironmentVariable("SHARPEMU_DBG_DEVICE_WRITES") == "1"; // TEMP
    private static long _dbgWrittenPages; // TEMP

    internal void NoteDeviceAddressFault(ulong guestAddress, ulong size, bool insideGuestMapping = false)
    {
        _deviceAddressFaultSpans.Add(guestAddress, size);
        if (insideGuestMapping)
            _mappedDeviceAddressFaultSpans.Add(guestAddress, size);
    }

    // A guest buffer the GPU reads through pointers (a per-frame ring, for one) is reached a
    // page at a time, and every first touch reads zeros for that frame. A fault therefore
    // brings in the aligned window around it, clipped to the guest mapping that holds it.
    internal const ulong DeviceAddressFaultWindow = 2UL << 20;

    internal static GuestSpan DeviceAddressFaultSpan(ulong pageAddress, ulong pageSize, ulong mappingStart, ulong mappingLength)
    {
        if (mappingLength == 0 || pageAddress < mappingStart || pageAddress - mappingStart >= mappingLength)
        {
            return new GuestSpan(pageAddress, pageSize);
        }

        var mappingEnd = mappingStart + mappingLength;
        var windowStart = Math.Max(pageAddress & ~(DeviceAddressFaultWindow - 1), mappingStart);
        var windowEnd = Math.Min((pageAddress & ~(DeviceAddressFaultWindow - 1)) + DeviceAddressFaultWindow, mappingEnd);
        return new GuestSpan(windowStart, windowEnd - windowStart);
    }

    // Forgets fault ranges whose guest mapping is gone, so their buffers can age out again.
    private void PruneUnmappedDeviceAddressFaults()
    {
        List<GuestSpan>? unmapped = null;
        _mappedDeviceAddressFaultSpans.ForEach((start, size) =>
        {
            if (!KernelMemoryCompatExports.TryGetMappedRange(start, out var mappingStart, out var mappingLength) ||
                start + size > mappingStart + mappingLength)
                (unmapped ??= []).Add(new GuestSpan(start, size));
        });
        if (unmapped is null) return;
        foreach (var span in unmapped)
        {
            _mappedDeviceAddressFaultSpans.Remove(span.Address, span.Size);
            _deviceAddressFaultSpans.Remove(span.Address, span.Size);
        }
    }

    // Device-address programs may read any mapped byte, so every CPU-dirty buffer page is
    // uploaded first. Only the dirty blocks are visited; the mapped spans (tens of thousands
    // with texture streaming) are walked only when the mapping or the retirement tick moved.
    // A fixed span list has no mapping version, so its touch pass always runs.
    public void PrepareBda(IReadOnlyCollection<GuestSpan> mapped)
    {
        _bdaTouchTick = ulong.MaxValue;
        PrepareBda(0, () => mapped, (address, size, overlapping) =>
        {
            foreach (var span in mapped)
            {
                var start = Math.Max(span.Address, address);
                var end = Math.Min(span.Address + span.Size, address + size);
                if (start < end)
                    overlapping.Add(new GuestSpan(start, end - start));
            }
        });
    }

    // The mapped parts of one dirty block, reused across the blocks of a sweep.
    private readonly List<GuestSpan> _mappedWithinScratch = [];
    private readonly List<GuestSpan> _cpuDirtyRunScratch = [];
    private ulong _bdaSweepMapping = ulong.MaxValue;

    public void PrepareBda(long mappingVersion, Func<IReadOnlyCollection<GuestSpan>> mapped, Action<ulong, ulong, List<GuestSpan>> mappedWithin)
    {
        MapPendingGpuMappings();
        var traceAddress = GuestGpuMemoryHook.TraceAddress;
        var traceCovered = false;
        if (traceAddress != 0)
        {
            foreach (var span in mapped())
            {
                if (traceAddress != 0 && GuestGpuMemoryHook.Traces(span.Address, span.Size))
                    traceCovered = true;
                SynchronizeBuffersInRange(span.Address, span.Size);
            }
        }
        else
        {
            if (_retirementPolicy.CurrentTick != _bdaTouchTick || (ulong)mappingVersion != _bdaTouchMapping)
            {
                foreach (var span in mapped())
                    TouchBuffersInRange(span.Address, span.Size);
                PruneUnmappedDeviceAddressFaults();
                _deviceAddressFaultSpans.ForEach(_touchBuffersInRange ??= TouchBuffersInRange);
                _bdaTouchTick = _retirementPolicy.CurrentTick;
                _bdaTouchMapping = (ulong)mappingVersion;
            }

            if (SharpEmu.ShaderCompiler.DbgFlags.Disabled("bda")) // TEMP
            {
                foreach (var span in mapped())
                    _tracker.ForEachPossiblyCpuDirtyRange(span.Address, span.Size, UploadDirtyBuffersInRange);
            }
            else
            {
                // Only blocks that gained dirty pages since the previous sweep are revisited: CPU-dirty
                // pages no buffer covers stay dirty, and rescanning them and the thousands of small
                // buffers around them for every device-address draw dominated the frame. A mapping
                // change can expose dirty pages the earlier sweeps skipped, so it sweeps every block.
                var pendingOnly = _bdaSweepMapping == (ulong)mappingVersion;
                _bdaSweepMapping = (ulong)mappingVersion;
                _tracker.ForEachCpuDirtyRun(_cpuDirtyRunScratch, pendingOnly, (address, size) =>
                {
                    _mappedWithinScratch.Clear();
                    mappedWithin(address, size, _mappedWithinScratch);
                    foreach (var span in _mappedWithinScratch)
                        UploadDirtyBuffersInRange(span.Address, span.Size);
                });
            }
        }

        if (traceAddress != 0)
            GuestGpuMemoryHook.Trace(traceAddress, 1,
                $"device-address-preparation covered={traceCovered} registered={IsRegionRegistered(traceAddress, 1)} submission_tick={_scheduler.CurrentTick} collection_tick={_retirementPolicy.CurrentTick}");
        _faultProcessPending = true;
    }

    private Action<ulong, ulong>? _touchBuffersInRange;

    private void TouchBuffersInRange(ulong guestAddress, ulong size)
    {
        var end = guestAddress + size;
        var index = _registry.FindFirstOverlappingIndex(guestAddress);
        for (; index < _registry.RegisteredCount && _registry.GetRegisteredAddress(index) < end; index++)
            TouchBuffer(_registry.GetRegisteredIdentifier(index));
    }

    // The upload half of SynchronizeBuffersInRange, for a range that may hold CPU-dirty pages.
    private void UploadDirtyBuffersInRange(ulong guestAddress, ulong size)
    {
        var end = guestAddress + size;
        var index = _registry.FindFirstOverlappingIndex(guestAddress);
        for (; index < _registry.RegisteredCount && _registry.GetRegisteredAddress(index) < end; index++)
        {
            var buffer = _registry.GetBuffer(_registry.GetRegisteredIdentifier(index));
            var start = Math.Max(buffer.CpuAddress, guestAddress);
            var finish = Math.Min(buffer.CpuAddress + buffer.Size, end);
            if (start < finish && _tracker.HasCpuDirtyPages(start, finish - start))
                _ = SynchronizeBuffer(buffer, start, finish - start, false, false, preserveCpuWriteHotPages: false);
        }
    }

    public void SynchronizeBuffersInRange(ulong guestAddress, ulong size)
    {
        var end = guestAddress + size;
        var index = _registry.FindFirstOverlappingIndex(guestAddress);
        for (; index < _registry.RegisteredCount && _registry.GetRegisteredAddress(index) < end; index++)
        {
            var identifier = _registry.GetRegisteredIdentifier(index);
            var buffer = _registry.GetBuffer(identifier);
            var start = Math.Max(buffer.CpuAddress, guestAddress);
            var finish = Math.Min(buffer.CpuAddress + buffer.Size, end);
            if (start < finish)
            {
                if (GuestGpuMemoryHook.Traces(start, finish - start))
                    GuestGpuMemoryHook.Trace(start, finish - start,
                        $"device-address-touch buffer={identifier} submission_tick={_scheduler.CurrentTick} collection_tick={_retirementPolicy.CurrentTick}");
                // Clean buffers remain in use through their device addresses.
                TouchBuffer(identifier);
                // Device-address reads reuse persistent buffers; track writes after each upload.
                // A range without CPU-dirty pages has nothing to upload (the same early exit
                // SynchronizeBuffer takes for this read-only call), and the block summary
                // answers that without a lock for the common all-clean case.
                if (_tracker.HasCpuDirtyPages(start, finish - start))
                    _ = SynchronizeBuffer(buffer, start, finish - start, false, false, preserveCpuWriteHotPages: false);
            }
        }
    }

    // Tests use lower thresholds to check collection without large allocations.
    internal void SetCollectionThresholds(ulong collectionThreshold, ulong criticalThreshold)
    {
        _retirementPolicy.SetThresholds(collectionThreshold, criticalThreshold);
    }

    // Runs before the image readback flush and both collectors; the order matches the render loop.
    internal bool DbgFaultProcessPending => _faultProcessPending; // TEMP

    public void ProcessPendingFaultBuffer()
    {
        if (_faultProcessPending)
        {
            _faultProcessPending = false;
            ProcessFaultBuffer();
        }
    }

    public void RunGarbageCollector()
    {
        using var foreignRead = _device.Slabs.BeginForeignRead();
        ProcessPendingFaultBuffer();
        if (!_retirementPolicy.TryBeginCollection(TotalUsedMemory, out var retirement))
        {
            return;
        }


        var dirtyBuffers = new List<ResourceSlotIdentifier>();
        var copies = new List<DownloadPiece>();
        var retireCount = 0;
        _registry.VisitRetirementCandidates(retirement.LatestEligibleTick, bufferIdentifier =>
        {
            var buffer = _registry.TryGetRegisteredBuffer(bufferIdentifier);
            if (buffer == null)
            {
                throw SubmissionScheduler.Fatal("The recency queue contains a deleted buffer.");
            }

            _tracker.ValidateGpuDirtyOwnership(_gpuModifiedRanges, buffer.CpuAddress, buffer.Size, "garbage collection");
            var dirty = _tracker.HasGpuDirtyPages(buffer.CpuAddress, buffer.Size);
            if (dirty && !retirement.DownloadDirtyBuffers)
            {
                return false;
            }

            if (GuestGpuMemoryHook.Traces(buffer.CpuAddress, buffer.Size))
                GuestGpuMemoryHook.Trace(buffer.CpuAddress, buffer.Size,
                    $"device-address-collection buffer={bufferIdentifier} dirty={dirty} aggressive={retirement.DownloadDirtyBuffers} cutoff_tick={retirement.LatestEligibleTick} collection_tick={_retirementPolicy.CurrentTick - 1} used_bytes={_registry.RegisteredBytes}");
            if (dirty)
            {
                CollectDirtyPieces(buffer, copies, "garbage collection");
                dirtyBuffers.Add(bufferIdentifier);
            }
            else
            {
                _tracker.UntrackMemory(buffer.CpuAddress, buffer.Size);
                DeleteBuffer(bufferIdentifier);
            }

            return ++retireCount == retirement.MaximumBufferCount;
        });
        if (dirtyBuffers.Count == 0)
        {
            return;
        }

        if (copies.Count == 0)
        {
            throw SubmissionScheduler.Fatal("Dirty buffers have no download ranges.");
        }

        DownloadBufferMemory(copies);
        foreach (var bufferIdentifier in dirtyBuffers)
        {
            ReleaseDownloaded(bufferIdentifier);
        }
    }

    // Teardown: every GPU result reaches guest memory and every page returns to its guest protection.
    public void Shutdown()
    {
        var drained = false;
        try
        {
            if (_scheduler.Active)
            {
                _scheduler.Finish();
            }

            using var foreignRead = _device.Slabs.BeginForeignRead();
            var copies = new List<DownloadPiece>();
            var dirtyBuffers = new List<ResourceSlotIdentifier>();
            foreach (var bufferIdentifier in _registry.SnapshotRegisteredIdentifiers())
            {
                var buffer = _registry.GetBuffer(bufferIdentifier);
                if (_tracker.HasGpuDirtyPages(buffer.CpuAddress, buffer.Size))
                {
                    CollectDirtyPieces(buffer, copies, "shutdown");
                    dirtyBuffers.Add(bufferIdentifier);
                }
            }

            if (copies.Count != 0)
            {
                DownloadBufferMemory(copies);
            }

            foreach (var bufferIdentifier in dirtyBuffers)
            {
                ReleaseDownloaded(bufferIdentifier);
            }

            foreach (var bufferIdentifier in _registry.SnapshotRegisteredIdentifiers())
            {
                var buffer = _registry.GetBuffer(bufferIdentifier);
                _tracker.UntrackMemory(buffer.CpuAddress, buffer.Size);
                Unregister(bufferIdentifier);
                _registry.CompleteRetirement(bufferIdentifier);
            }

            if (_scheduler.Active)
            {
                _scheduler.Finish();
            }

            Dispose();
            drained = true;
        }
        finally
        {
            lock (_shutdownGate)
            {
                _outcome = drained ? ShutdownOutcome.Drained : ShutdownOutcome.Failed;
                Monitor.PulseAll(_shutdownGate);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        GuestGpuMemoryHook.GpuMappingCreated -= _gpuMappingCreated;
        _registry.Dispose();
        _deviceRing.Dispose();
        _download.Dispose();
        _stream.Dispose();
        _staging.Dispose();
        _bdaPageTable.Dispose();
        _gds.Dispose();
        _faults.Dispose();
    }

    // False only when the store closed and its drain failed; the caller then declines the fault.
    private bool ReadMemoryOrAwaitShutdown(ulong guestAddress, ulong size, bool isWrite,
        GuestMemoryProfile.ReadbackSource source = GuestMemoryProfile.ReadbackSource.ExplicitReadback)
    {
        if (!_relay.IsGpuQueueThread && SubmissionScheduler.InDeferredOperation)
        {
            throw SubmissionScheduler.Fatal(
                $"unsupported buffer readback from an asynchronous GPU completion, addr=0x{guestAddress:X16} size=0x{size:X16}");
        }

        // A guest thread waits for the GPU itself: the GPU queue thread only submits the copy
        // and clears the tracking afterwards, so it keeps recording while the GPU catches up.
        if (!_relay.IsGpuQueueThread && AsyncReadback is not null && AsyncGuestReadback)
        {
            PendingGuestRead? pending = null;
            if (_relay.TryRunOnGpuQueue(() => pending = BeginGuestRead(guestAddress, size, isWrite, source)))
            {
                if (pending is null)
                {
                    return true;
                }

                AsyncReadback.Wait(pending.Ticket);
                if (_relay.TryRunOnGpuQueue(() => FinishGuestRead(pending)))
                {
                    return true;
                }
            }

            lock (_shutdownGate)
            {
                while (_outcome == ShutdownOutcome.Pending)
                {
                    Monitor.Wait(_shutdownGate);
                }

                return _outcome == ShutdownOutcome.Drained;
            }
        }

        var dbgStart = DbgStacks.Start(); // TEMP
        if (_relay.TryRunOnGpuQueue(() => ReadMemoryOnGpu(guestAddress, size, isWrite, source)))
        {
            DbgStacks.Record($"readback {source} w={isWrite}", dbgStart, size); // TEMP
            return true;
        }

        lock (_shutdownGate)
        {
            while (_outcome == ShutdownOutcome.Pending)
            {
                Monitor.Wait(_shutdownGate);
            }

            return _outcome == ShutdownOutcome.Drained;
        }
    }

    private static readonly bool AsyncGuestReadback = Environment.GetEnvironmentVariable("SHARPEMU_ASYNC_GUEST_READBACK") == "1";

    private sealed record PendingGuestRead(VulkanAsyncReadback.Ticket Ticket, List<DownloadPiece> Copies, ulong WindowBegin, ulong WindowEnd, ulong WaitTick, bool IsWrite, ulong Address, ulong Size);

    // The GPU-queue half of a guest read: submits the download when it can run without this
    // thread waiting, otherwise performs the whole read here and returns null.
    private PendingGuestRead? BeginGuestRead(ulong guestAddress, ulong size, bool isWrite, GuestMemoryProfile.ReadbackSource source)
    {
        if (isWrite && !IsRegionRegistered(guestAddress, size))
        {
            return null;
        }

        var copies = CollectReadbackWindow(guestAddress, size, out var windowBegin, out var windowEnd);
        var waitTick = 0UL;
        var total = 0UL;
        var asyncPossible = copies.Count != 0;
        foreach (var copy in copies)
        {
            var written = copy.Buffer.LastGpuWriteTick;
            asyncPossible &= written != 0;
            waitTick = Math.Max(waitTick, written);
            total += copy.Size;
        }

        if (!asyncPossible || total > AsyncReadbackLimit)
        {
            ReadMemoryOnGpu(guestAddress, size, isWrite, source);
            return null;
        }

        if (waitTick >= _scheduler.CurrentTick)
        {
            _scheduler.Flush();
        }

        var pieces = new ReadbackPiece[copies.Count];
        for (var index = 0; index < pieces.Length; index++)
        {
            pieces[index] = new ReadbackPiece(copies[index].Buffer, copies[index].SourceOffset, copies[index].Size);
        }

        if (AsyncReadback!.TrySubmit(pieces, waitTick) is not { } ticket)
        {
            ReadMemoryOnGpu(guestAddress, size, isWrite, source);
            return null;
        }

        return new PendingGuestRead(ticket, copies, windowBegin, windowEnd, waitTick, isWrite, guestAddress, size);
    }

    // Runs after the guest thread wrote the downloaded bytes. A newer GPU write to the same
    // buffers keeps the pages GPU-dirty, so the next guest access downloads again.
    // The guest bytes are written here, on the GPU queue thread: a range another download
    // already brought back (and possibly newer) is no longer GPU-modified and is skipped.
    private void FinishGuestRead(PendingGuestRead pending)
    {
        var readback = AsyncReadback!;
        try
        {
            foreach (var copy in pending.Copies)
            {
                if (copy.Buffer.LastGpuWriteTick > pending.WaitTick)
                {
                    return;
                }
            }

            for (var index = 0; index < pending.Copies.Count; index++)
            {
                var copy = pending.Copies[index];
                if (!_gpuModifiedRanges.Overlaps(copy.Address, copy.Size))
                {
                    continue;
                }

                readback.Consume(pending.Ticket, index, (_, bytes) =>
                {
                    if (!_backing.TryWriteBacking(copy.Address, bytes))
                    {
                        throw SubmissionScheduler.Fatal($"Could not write the required direct backing: addr=0x{copy.Address:X16} size=0x{(ulong)bytes.Length:X16}");
                    }
                });
                _gpuModifiedRanges.Remove(copy.Address, copy.Size);
            }
        }
        finally
        {
            readback.Release(pending.Ticket);
        }

        _tracker.ClearGpuDirtyPages(pending.WindowBegin, pending.WindowEnd - pending.WindowBegin);
        if (pending.IsWrite)
        {
            _tracker.MarkCpuDirtyPages(pending.Address, pending.Size);
        }
    }

    private void ReadMemoryOnGpu(ulong guestAddress, ulong size, bool isWrite, GuestMemoryProfile.ReadbackSource source)
    {
        using var readbackScope = GuestMemoryProfile.Measure(GuestMemoryProfile.Operation.BufferReadback);
        using var foreignRead = _device.Slabs.BeginForeignRead();
        var readbackStarted = GuestMemoryProfile.ReadbackDetailsEnabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (isWrite && !IsRegionRegistered(guestAddress, size))
        {
            return;
        }

        if (_dbgFaults) DbgFaultKind(source.ToString(), guestAddress, size); // TEMP
        var dbgStart = DbgStacks.Start(); // TEMP
        var copies = CollectReadbackWindow(guestAddress, size, out var windowBegin, out var windowEnd);
        var dbgDone = copies.All(c => _scheduler.IsTickComplete(c.Buffer.LastGpuWriteTick)); // TEMP
        var dbgCurrent = copies.Any(c => c.Buffer.LastGpuWriteTick >= _scheduler.CurrentTick); // TEMP
        if (copies.Count != 0)
        {
            DownloadBufferMemory(copies);
            _tracker.ClearGpuDirtyPages(windowBegin, windowEnd - windowBegin);
        }
        ulong dbgBytes = 0; foreach (var c in copies) dbgBytes += c.Size; // TEMP
        if (!dbgDone && Environment.GetEnvironmentVariable("SHARPEMU_DBG_SLOWRB") == "1") // TEMP
        {
            var ms = System.Diagnostics.Stopwatch.GetElapsedTime(dbgStart).TotalMilliseconds;
            var desc = string.Join(",", copies.Take(4).Select(c => $"0x{c.Address:X}+0x{c.Size:X}@buf0x{c.Buffer.CpuAddress:X}+0x{c.Buffer.Size:X}/t{c.Buffer.LastGpuWriteTick}"));
            Console.Error.WriteLine($"[DBG][SLOWRB] {source} addr=0x{guestAddress:X}+{size} ms={ms:F1} cur={_scheduler.CurrentTick} done={_scheduler.Timeline.CompletedTick} n={copies.Count} {desc}");
        }
        DbgStacks.Record($"onGpu {source} done={dbgDone} in_current={dbgCurrent} copies={Math.Min(copies.Count, 3)}", dbgStart, dbgBytes, depth: 0); // TEMP

        if (isWrite)
        {
            _tracker.MarkCpuDirtyPages(guestAddress, size);
        }
        if (GuestMemoryProfile.ReadbackDetailsEnabled)
        {
            var downloadedBytes = 0UL;
            foreach (var copy in copies)
                downloadedBytes += copy.Size;
            GuestMemoryProfile.RecordBufferReadback(windowBegin, windowEnd - windowBegin, isWrite, downloadedBytes,
                System.Diagnostics.Stopwatch.GetTimestamp() - readbackStarted, source);
        }
    }

    // The GPU-modified ranges to download for a read of the range, widened to a window so
    // nearby CPU reads share one GPU drain.
    private List<DownloadPiece> CollectReadbackWindow(ulong guestAddress, ulong size, out ulong windowBegin, out ulong windowEnd)
    {
        var buffer = _registry.GetBuffer(FindBuffer(guestAddress, size));
        const ulong windowSize = 512 * 1024;
        var bufferEnd = buffer.CpuAddress + buffer.Size;
        windowBegin = Math.Max(guestAddress & ~(windowSize - 1), buffer.CpuAddress);
        windowEnd = Math.Min(Math.Max(windowBegin + windowSize, guestAddress + size), bufferEnd);

        var copies = new List<DownloadPiece>();
        _tracker.ForEachDownloadRange(
            windowBegin,
            windowEnd - windowBegin,
            clear: false,
            (address, bytes) => _tracker.ValidateGpuDirtyPages(_gpuModifiedRanges, address, bytes, "memory invalidation"),
            (address, bytes) =>
            {
                foreach (var range in _gpuModifiedRanges.GetOverlappingRanges(address, bytes))
                {
                    copies.Add(new DownloadPiece(buffer, buffer.Offset(range.Address), range.Address, range.Size));
                }
            });
        return copies;
    }

    private void CollectDirtyPieces(GpuBuffer buffer, List<DownloadPiece> copies, string operation)
    {
        _tracker.ForEachDownloadRange(
            buffer.CpuAddress,
            buffer.Size,
            clear: false,
            (address, bytes) => _tracker.ValidateGpuDirtyPages(_gpuModifiedRanges, address, bytes, operation),
            (address, bytes) =>
            {
                foreach (var range in _gpuModifiedRanges.GetOverlappingRanges(address, bytes))
                {
                    copies.Add(new DownloadPiece(buffer, range.Address - buffer.CpuAddress, range.Address, range.Size));
                }
            });
    }

    private void ReleaseDownloaded(ResourceSlotIdentifier bufferIdentifier)
    {
        var buffer = _registry.GetBuffer(bufferIdentifier);
        _tracker.ClearGpuDirtyPages(buffer.CpuAddress, buffer.Size);
        if (_tracker.HasGpuDirtyPages(buffer.CpuAddress, buffer.Size) || _gpuModifiedRanges.Overlaps(buffer.CpuAddress, buffer.Size))
        {
            throw SubmissionScheduler.Fatal("Buffer collection left GPU-owned memory.");
        }

        _tracker.UntrackMemory(buffer.CpuAddress, buffer.Size);
        Unregister(bufferIdentifier);
        CompleteRetirementAfterSubmittedWork(bufferIdentifier);
    }

    private void WriteDataBuffer(GpuBuffer buffer, ulong address, ReadOnlySpan<byte> source)
    {
        while (!source.IsEmpty)
        {
            var chunk = (int)Math.Min((ulong)source.Length, _staging.Size);
            var offset = _staging.Copy(source[..chunk], 4);
            buffer.CopyFrom(_scheduler.Current, _staging, offset, buffer.Offset(address), (ulong)chunk, AccessFlags.HostWriteBit);
            source = source[chunk..];
            address += (ulong)chunk;
        }
    }

    private void Register(ResourceSlotIdentifier bufferIdentifier) => UpdateRegistration(bufferIdentifier, insert: true);

    private void Unregister(ResourceSlotIdentifier bufferIdentifier) => UpdateRegistration(bufferIdentifier, insert: false);

    private void UpdateRegistration(ResourceSlotIdentifier bufferIdentifier, bool insert)
    {
        var buffer = _registry.GetBuffer(bufferIdentifier);
        if (!PageOwnerTable.TryGetPageRange(buffer.CpuAddress, buffer.Size, out var first, out var lastExclusive))
        {
            throw SubmissionScheduler.Fatal("The buffer is outside the page table.");
        }

        var sizePages = lastExclusive - first;
        if (GuestGpuMemoryHook.Traces(buffer.CpuAddress, buffer.Size))
            GuestGpuMemoryHook.Trace(buffer.CpuAddress, buffer.Size,
                $"device-address-registration insert={insert} buffer={bufferIdentifier} submission_tick={_scheduler.CurrentTick} collection_tick={_retirementPolicy.CurrentTick}");
        if (insert)
        {
            _registry.RegisterBuffer(bufferIdentifier, _retirementPolicy.CurrentTick);
            _tracker.MarkCpuSweepPending(buffer.CpuAddress, buffer.Size);
            var addresses = new ulong[sizePages];
            for (ulong page = 0; page < sizePages; page++)
            {
                addresses[page] = buffer.DeviceAddress + (page << CachingPageBits);
            }

            WriteDataBuffer(_bdaPageTable, first * sizeof(ulong), MemoryMarshal.AsBytes<ulong>(addresses));
        }
        else
        {
            _registry.BeginRetirement(bufferIdentifier);
            _bdaPageTable.Fill(first * sizeof(ulong), sizePages * sizeof(ulong), 0);
        }
    }

    private void TouchBuffer(GpuBuffer buffer)
    {
        var identifier = _registry.FindContainingBuffer(buffer.CpuAddress, buffer.Size);
        if (identifier.IsValid && ReferenceEquals(_registry.GetBuffer(identifier), buffer))
        {
            TouchBuffer(identifier);
        }
    }

    private void TouchBuffer(ResourceSlotIdentifier identifier) =>
        _registry.MarkBufferUsed(identifier, _retirementPolicy.CurrentTick);

    private void DeleteBuffer(ResourceSlotIdentifier bufferIdentifier)
    {
        if (_registry.TryGetRegisteredBuffer(bufferIdentifier) == null)
        {
            return;
        }

        Unregister(bufferIdentifier);
        CompleteRetirementAfterSubmittedWork(bufferIdentifier);
    }

    // Unregistering clears the buffer's page-table entries only for work recorded from now
    // on. Work already recorded or in flight can still reach it through its device address
    // (an async readback waits only for its last recorded writer), so it is destroyed once
    // that work completes.
    private void CompleteRetirementAfterSubmittedWork(ResourceSlotIdentifier bufferIdentifier)
    {
        if (_scheduler.Active)
        {
            _scheduler.QueueCompletionAction(() => _registry.CompleteRetirement(bufferIdentifier));
        }
        else
        {
            _registry.CompleteRetirement(bufferIdentifier);
        }
    }

    // Set when a second queue can copy readbacks; null keeps every readback on the main queue.
    internal IBufferReadback? AsyncReadback { get; set; }

    private const ulong AsyncReadbackLimit = 64UL << 20;

    // Packs pieces into the download ring, waits for the copy, then writes each through the backing alias.
    private void DownloadBufferMemory(List<DownloadPiece> copies)
    {
        if (TryDownloadAsync(copies))
        {
            foreach (var copy in copies)
            {
                _gpuModifiedRanges.Remove(copy.Address, copy.Size);
            }

            return;
        }

        var batch = new List<PlannedDownload>();
        var planner = new BufferDownloadBatchPlanner(_download.Size);
        foreach (var piece in copies)
        {
            var copy = piece;
            while (copy.Size != 0)
            {
                var placement = planner.Append(copy.SourceOffset, copy.Size, copy.Buffer.Size);
                batch.Add(new PlannedDownload(copy.Buffer, copy.Address, placement));
                copy = copy with
                {
                    SourceOffset = copy.SourceOffset + placement.DataSize,
                    Address = copy.Address + placement.DataSize,
                    Size = copy.Size - placement.DataSize,
                };
                if (planner.IsFull)
                {
                    FlushDownloads(batch, planner.PackedSize);
                    planner.Reset();
                }
            }
        }

        if (batch.Count != 0)
        {
            FlushDownloads(batch, planner.PackedSize);
        }

        foreach (var copy in copies)
        {
            _gpuModifiedRanges.Remove(copy.Address, copy.Size);
        }
    }

    // Copies the pieces on the readback queue after only the tick that last wrote their
    // buffers, instead of appending the copy to the main queue and draining all of it.
    private bool TryDownloadAsync(List<DownloadPiece> copies)
    {
        if (AsyncReadback is not { } readback || copies.Count == 0)
        {
            return false;
        }

        var waitTick = 0UL;
        var total = 0UL;
        foreach (var copy in copies)
        {
            var written = copy.Buffer.LastGpuWriteTick;
            // A GPU-modified range with no recorded writer: keep the conservative path.
            if (written == 0)
            {
                return false;
            }

            waitTick = Math.Max(waitTick, written);
            total += copy.Size;
        }

        if (total > AsyncReadbackLimit)
        {
            return false;
        }

        // The last writer is still in the buffer being recorded; submit it (without
        // waiting) so the readback queue has a signal to wait for.
        if (waitTick >= _scheduler.CurrentTick)
        {
            _scheduler.Flush();
        }

        var pieces = new ReadbackPiece[copies.Count];
        for (var index = 0; index < pieces.Length; index++)
        {
            pieces[index] = new ReadbackPiece(copies[index].Buffer, copies[index].SourceOffset, copies[index].Size);
        }

        readback.Read(pieces, waitTick, (index, bytes) =>
        {
            if (!_backing.TryWriteBacking(copies[index].Address, bytes))
            {
                throw SubmissionScheduler.Fatal($"Could not write the required direct backing: addr=0x{copies[index].Address:X16} size=0x{(ulong)bytes.Length:X16}");
            }
        });
        return true;
    }

    private void FlushDownloads(List<PlannedDownload> batch, ulong packedSize)
    {
        if (!_download.TryMap(packedSize, out var baseOffset, BufferDownloadBatchPlanner.Alignment))
        {
            throw SubmissionScheduler.Fatal("The download ring could not map the batch.");
        }

        foreach (var copy in batch)
        {
            var placement = copy.Placement;
            _download.CopyFrom(
                _scheduler.Current, copy.Buffer, placement.SourceOffset, baseOffset + placement.DestinationOffset, placement.TransferSize,
                AccessFlags.MemoryWriteBit, AccessFlags.None, AccessFlags.MemoryReadBit | AccessFlags.MemoryWriteBit, AccessFlags.HostReadBit);
        }

        _download.Commit();
        var completionTick = _scheduler.CurrentTick;
        _scheduler.Finish();
        _scheduler.WaitForPriorityOperations(completionTick);
        foreach (var copy in batch)
        {
            var placement = copy.Placement;
            var offset = baseOffset + placement.DataOffset;
            _download.Invalidate(offset, placement.DataSize);
            if (!_backing.TryWriteBacking(copy.Address, _download.Mapped.Slice((int)offset, (int)placement.DataSize)))
            {
                throw SubmissionScheduler.Fatal($"Could not write the required direct backing: addr=0x{copy.Address:X16} size=0x{placement.DataSize:X16}");
            }
        }

        batch.Clear();
    }

    private OverlapSpan ResolveOverlaps(ulong guestAddress, ulong size)
    {
        var range = new BufferMergeRange(guestAddress, guestAddress + size);
        var first = _registry.FindFirstOverlappingIndex(range.Begin);
        var last = first;
        for (; last < _registry.RegisteredCount && _registry.GetRegisteredAddress(last) < range.End; last++)
        {
            var buffer = _registry.GetBuffer(_registry.GetRegisteredIdentifier(last));
            if (range.IncludeBuffer(buffer.CpuAddress, buffer.CpuAddress + buffer.Size, buffer.StreamScore))
            {
                first = _registry.FindFirstOverlappingIndex(range.Begin);
                if (first < _registry.RegisteredCount)
                {
                    range.IncludeEarlierBuffer(_registry.GetRegisteredAddress(first));
                }
            }
        }

        return new OverlapSpan(first, last, range.Begin, range.End, range.HasStreamExpansion);
    }

    private void MergeOverlappingBuffer(ResourceSlotIdentifier newBufferIdentifier, ResourceSlotIdentifier overlappingBufferIdentifier, bool accumulateStreamScore)
    {
        var newBuffer = _registry.GetBuffer(newBufferIdentifier);
        var overlap = _registry.GetBuffer(overlappingBufferIdentifier);
        if (GuestGpuMemoryHook.Traces(overlap.CpuAddress, overlap.Size))
            GuestGpuMemoryHook.Trace(overlap.CpuAddress, overlap.Size,
                $"device-address-merge old={overlappingBufferIdentifier} replacement={newBufferIdentifier} submission_tick={_scheduler.CurrentTick}");
        if (accumulateStreamScore)
        {
            newBuffer.AddStreamScore(overlap.StreamScore + 1);
        }

        newBuffer.CopyFrom(_scheduler.Current, overlap, 0, overlap.CpuAddress - newBuffer.CpuAddress, overlap.Size);
        DeleteBuffer(overlappingBufferIdentifier);
    }

    private ResourceSlotIdentifier CreateBuffer(ulong guestAddress, ulong size)
    {
        if (_scheduler.Current.IsInvalid)
        {
            throw SubmissionScheduler.Fatal("Buffer creation requires a command buffer that is recording.");
        }

        var end = (guestAddress + size + CachingPageSize - 1) & ~(CachingPageSize - 1);
        guestAddress &= ~(CachingPageSize - 1);
        size = end - guestAddress;
        var overlap = ResolveOverlaps(guestAddress, size);
        var overlapping = new List<ResourceSlotIdentifier>();
        for (var index = overlap.First; index < overlap.Last; index++)
        {
            overlapping.Add(_registry.GetRegisteredIdentifier(index));
        }

        var bufferIdentifier = _registry.AllocateBuffer(new GpuBuffer(
            _device, _scheduler, GpuBufferUsage.DeviceLocal, overlap.Begin,
            GpuBuffer.AllFlags | BufferUsageFlags.ShaderDeviceAddressBit, overlap.End - overlap.Begin, allowSlab: true), overlap.Begin, overlap.End - overlap.Begin);
        foreach (var oldId in overlapping)
        {
            MergeOverlappingBuffer(bufferIdentifier, oldId, !overlap.HasStreamLeap);
        }

        Register(bufferIdentifier);
        return bufferIdentifier;
    }

    private bool SynchronizeBuffer(GpuBuffer buffer, ulong guestAddress, ulong size, bool isWritten, bool isTexelBuffer,
        bool preserveCpuWriteHotPages = true, bool readImageBacking = false)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.BufferDirtySynchronization);
        var startedAt = BufferUploadProfile.Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        // The locked query observes completed writes; a later write remains dirty for the next obtain.
        if (!preserveCpuWriteHotPages && !isWritten && !isTexelBuffer && !_tracker.HasCpuDirtyPages(guestAddress, size))
        {
            if (BufferUploadProfile.Enabled)
                BufferUploadProfile.Record(guestAddress, size, 0, 0, 0, System.Diagnostics.Stopwatch.GetTimestamp() - startedAt);
            return false;
        }
        var copies = new List<BufferCopy>();
        var totalSize = 0UL;
        GpuBuffer? source = null;
        _tracker.ForEachUploadRange(
            guestAddress,
            size,
            isWritten,
            (address, bytes) =>
            {
                copies.Add(new BufferCopy(totalSize, buffer.Offset(address), bytes));
                totalSize += bytes;
            },
            () => source = _uploader.PrepareSource(buffer.CpuAddress, CollectionsMarshal.AsSpan(copies), totalSize, guestAddress, size,
                readImageBacking ? TryReadImageSource : null),
            preserveCpuWriteHotPages);
        if (source != null)
        {
            buffer.NoteGpuWrite();
            var command = _scheduler.Current;
            command.EndRendering();
            var native = new CommandBuffer(command.Handle);
            var vk = _device.Vk;
            var before = new BufferMemoryBarrier2
            {
                SType = StructureType.BufferMemoryBarrier2,
                SrcAccessMask = AccessFlags2.MemoryReadBit | AccessFlags2.MemoryWriteBit | AccessFlags2.TransferReadBit | AccessFlags2.TransferWriteBit,
                DstAccessMask = AccessFlags2.TransferWriteBit,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                Buffer = buffer.Handle,
                Offset = 0,
                Size = buffer.Size,
            };
            VulkanSynchronization.PipelineBarrier(vk,
                native, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit, DependencyFlags.ByRegionBit,
                0, null, 1, &before, 0, null);
            var regions = CollectionsMarshal.AsSpan(copies);
            fixed (BufferCopy* pointer = regions)
            {
                vk.CmdCopyBuffer(native, source.Handle, buffer.Handle, (uint)regions.Length, pointer);
            }

            var after = before;
            after.SrcAccessMask = AccessFlags2.TransferWriteBit;
            after.DstAccessMask = AccessFlags2.MemoryReadBit | AccessFlags2.MemoryWriteBit;
            VulkanSynchronization.PipelineBarrier(vk,
                native, PipelineStageFlags.TransferBit, PipelineStageFlags.AllCommandsBit, DependencyFlags.ByRegionBit,
                0, null, 1, &after, 0, null);
        }

        if (BufferUploadProfile.Enabled)
        {
            var elapsedTicks = System.Diagnostics.Stopwatch.GetTimestamp() - startedAt;
            ulong hotBytes = 0;
            foreach (var copy in copies)
                hotBytes += _tracker.CountCpuWriteHotBytes(buffer.CpuAddress + copy.DstOffset, copy.Size);
            BufferUploadProfile.Record(guestAddress, size, copies.Count, totalSize, hotBytes, elapsedTicks);
        }

        if (isTexelBuffer && !readImageBacking)
        {
            var copiedFromImage = RequireImageCache().TrySynchronizeBufferFromImage(buffer, guestAddress, size);
            if (copiedFromImage)
            {
                buffer.NoteGpuWrite();
            }

            return copiedFromImage;
        }

        return false;
    }

    private GpuBuffer? FindOwner(ulong guestAddress, ulong size)
    {
        return _registry.TryGetRegisteredBuffer(_registry.FindContainingBuffer(guestAddress, size));
    }

    private IGuestImageCache RequireImageCache() =>
        ImageCache ?? throw SubmissionScheduler.Fatal("The image cache is not connected.");

    private static bool IsValidRange(ulong guestAddress, ulong size) => guestAddress != 0 && size != 0 && new GuestSpan(guestAddress, size).IsValid;

}
