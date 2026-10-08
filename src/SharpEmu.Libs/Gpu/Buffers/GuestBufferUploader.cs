// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using Silk.NET.Vulkan;

namespace SharpEmu.Libs.Gpu.Buffers;

internal delegate bool GuestBufferSourceReader(ulong address, Span<byte> destination);

// The cache owns the staging ring; temporary sources retire after their submission.
internal sealed class GuestBufferUploader(
    GpuDeviceInfo device, SubmissionScheduler scheduler, ICpuMemory guest, GpuRingBuffer staging)
{
    private static readonly bool DbgUploadStacks = Environment.GetEnvironmentVariable("SHARPEMU_DBG_UPLOAD_STACKS") == "1"; // TEMP
    private static long _dbgUploadCalls; // TEMP

    public GpuBuffer? PrepareSource(ulong bufferAddress, Span<BufferCopy> regions,
        ulong totalSize, ulong requestedAddress, ulong requestedSize, GuestBufferSourceReader? readSource = null)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.BufferStagingUpload);
        if (regions.IsEmpty)
        {
            return null;
        }

        if (DbgUploadStacks && Interlocked.Increment(ref _dbgUploadCalls) % 997 == 0) // TEMP
        {
            var frames = new System.Diagnostics.StackTrace(1, false).GetFrames().Skip(1).Take(7).Select(frame => frame.GetMethod()?.Name ?? "?");
            Console.Error.WriteLine($"[DBG][UPLOAD] size=0x{totalSize:X} regions={regions.Length} first=0x{regions[0].Size:X} req=0x{requestedSize:X} {string.Join(" < ", frames)}");
        }

        if (staging.TryMap(totalSize, out var baseOffset, 4))
        {
            foreach (ref var copy in regions)
            {
                ReadGuestBytes(bufferAddress + copy.DstOffset,
                    staging.Mapped.Slice((int)(baseOffset + copy.SrcOffset), (int)copy.Size), requestedAddress, requestedSize, readSource);
                copy.SrcOffset += baseOffset;
            }

            staging.Commit();
            return staging;
        }

        var temporary = new GpuBuffer(device, scheduler, GpuBufferUsage.Upload, 0, BufferUsageFlags.TransferSrcBit, totalSize);
        foreach (ref readonly var copy in regions)
        {
            ReadGuestBytes(bufferAddress + copy.DstOffset,
                temporary.Mapped.Slice((int)copy.SrcOffset, (int)copy.Size), requestedAddress, requestedSize, readSource);
        }

        temporary.Flush(0, totalSize);
        scheduler.QueueCompletionAction(temporary.Dispose);
        return temporary;
    }

    private void ReadGuestBytes(ulong address, Span<byte> destination, ulong requestedAddress, ulong requestedSize, GuestBufferSourceReader? readSource)
    {
        if (!(readSource is null ? guest.TryRead(address, destination) : readSource(address, destination)))
        {
            Console.Error.WriteLine($"[GPU][READ_FAILURE] addr=0x{address:X16} size=0x{destination.Length:X}");
            Console.Error.WriteLine($"[GPU][READ_FAILURE] binding=0x{requestedAddress:X16} size=0x{requestedSize:X}");
            try
            {
                Console.Error.WriteLine(guest.DescribeReadRange(address, (ulong)destination.Length));
            }
            catch (Exception error)
            {
                Console.Error.WriteLine($"[GPU][READ_FAILURE] mapping query failed: {error.GetType().Name}");
            }
            throw SubmissionScheduler.Fatal($"Could not read guest memory: addr=0x{address:X16} size=0x{destination.Length:X}");
        }
    }
}
