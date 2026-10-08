// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Intrinsics.X86;

namespace SharpEmu.Libs.Gpu.Buffers;

// Reads from host-visible device memory (resizable BAR: VRAM across PCIe, mapped write-combined) are uncached, so a
// plain memcpy moves one small load per round trip (~50 MB/s measured). Non-temporal loads pull a whole 64-byte line
// per request from write-combined memory.
internal static unsafe class WriteCombinedCopy
{
    private const int Line = 64;
    private static readonly bool Enabled =
        Sse41.IsSupported && Environment.GetEnvironmentVariable("SHARPEMU_WC_STREAM_COPY") != "0";

    public static void Read(byte* source, byte* destination, int length)
    {
        if (!Enabled || length < 4 * Line)
        {
            Buffer.MemoryCopy(source, destination, length, length);
            return;
        }

        var offset = 0;
        var head = (int)((Line - ((nuint)source & (Line - 1))) & (Line - 1));
        if (head != 0)
        {
            Buffer.MemoryCopy(source, destination, head, head);
            offset = head;
        }

        for (; offset + Line <= length; offset += Line)
        {
            var line = source + offset;
            var a = Sse41.LoadAlignedVector128NonTemporal(line);
            var b = Sse41.LoadAlignedVector128NonTemporal(line + 16);
            var c = Sse41.LoadAlignedVector128NonTemporal(line + 32);
            var d = Sse41.LoadAlignedVector128NonTemporal(line + 48);
            var target = destination + offset;
            Sse2.Store(target, a);
            Sse2.Store(target + 16, b);
            Sse2.Store(target + 32, c);
            Sse2.Store(target + 48, d);
        }

        if (offset < length)
        {
            Buffer.MemoryCopy(source + offset, destination + offset, length - offset, length - offset);
        }
    }
}
