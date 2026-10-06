// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.GpuMemory;

// TEMP: logs the first guest CPU access to each page of a GPU-written image (SHARPEMU_DBG_IMAGE_READS=1).
// Every page is watched at most once, so overlapping images cannot overflow the watch counts.
public static class DbgImageReadProbe
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_DBG_IMAGE_READS") == "1";
    private const ulong Page = TrackerLayout.PageBytes;
    private static readonly object Gate = new();
    private static readonly Dictionary<ulong, string> Active = new();
    private static readonly HashSet<ulong> Probed = new();
    private static readonly HashSet<string> Reported = new();
    private static PageGuard? _pages;

    [ThreadStatic] public static bool Adding;
    [ThreadStatic] public static bool Skipped;

    // A real watcher takes a page the probe holds: the page keeps its read watch.
    public static bool Release(ulong address)
    {
        if (!Enabled)
        {
            return false;
        }

        lock (Gate)
        {
            return Active.Remove(address & ~(Page - 1));
        }
    }

    public static void Probe(PageGuard pages, ulong address, ulong size, string info)
    {
        if (!Enabled || size == 0)
        {
            return;
        }

        var text = $"image=0x{address:X}+0x{size:X} {info}";
        var end = (address + size + Page - 1) & ~(Page - 1);
        for (var page = address & ~(Page - 1); page < end; page += Page)
        {
            lock (Gate)
            {
                if (!Probed.Add(page))
                {
                    continue;
                }

                _pages = pages;
                Active[page] = text;
            }

            Adding = true;
            Skipped = false;
            try
            {
                pages.AddWatch(page, Page, blockReads: true);
            }
            finally
            {
                Adding = false;
            }

            if (Skipped)
            {
                lock (Gate)
                {
                    Active.Remove(page);
                }
            }
        }
    }

    public static bool TryHit(FaultKind kind, ulong address)
    {
        if (!Enabled)
        {
            return false;
        }

        var page = address & ~(Page - 1);
        string? info;
        bool first;
        lock (Gate)
        {
            if (!Active.Remove(page, out info))
            {
                return false;
            }

            first = Reported.Add(info);
        }

        if (first)
            Console.Error.WriteLine($"[DBG][IMGREAD] kind={kind} addr=0x{address:X} {info} tid={Environment.CurrentManagedThreadId}");
        _pages!.RemoveWatch(page, Page, blockReads: true);
        return true;
    }
}
