// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Diagnostics;

// TEMP: SHARPEMU_DBG_TARGET_WATCH=addr,... logs how those color targets are bound and cleared
// (target registers, DCC/metadata fills, image clears), each distinct event a few times.
internal static class DbgTargetWatch
{
    public static readonly HashSet<ulong> Addresses = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_TARGET_WATCH") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(text => Convert.ToUInt64(text.Replace("0x", ""), 16)).ToHashSet();
    public static readonly HashSet<ulong> MetadataAddresses = new();
    private static readonly Dictionary<string, int> Seen = new();

    private static readonly bool EnabledFlag = Addresses.Count != 0 || PhysicalWatchEnabled() || Environment.GetEnvironmentVariable("SHARPEMU_DBG_GFX_STORES") == "1" || Environment.GetEnvironmentVariable("SHARPEMU_DBG_ALL_FILLS") == "1";
    public static bool Enabled => EnabledFlag;
    private static bool PhysicalWatchEnabled() => !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("SHARPEMU_DBG_PHYS_WATCH"));

    // TEMP: SHARPEMU_DBG_RAW_WATCH=hexdword,... logs any command packet carrying one of those dwords.
    public static readonly uint[] RawWatch = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_RAW_WATCH") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(text => Convert.ToUInt32(text.Replace("0x", ""), 16)).ToArray();

    // TEMP: SHARPEMU_DBG_PHYS_WATCH=va,... reports any GPU write whose physical memory overlaps
    // the 32 MiB at those virtual addresses' physical addresses, whatever virtual address it uses.
    public static readonly ulong[] PhysicalWatch = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_PHYS_WATCH") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(text => Convert.ToUInt64(text.Replace("0x", ""), 16)).ToArray();
    private static ulong[]? _physicalBases;

    public static void CheckPhysical(string kind, ulong address, ulong size, Func<string> detail)
    {
        if (PhysicalWatch.Length == 0 || size == 0)
            return;
        if (_physicalBases is null)
        {
            var bases = new ulong[PhysicalWatch.Length];
            for (var index = 0; index < bases.Length; index++)
            {
                if (!Kernel.KernelMemoryCompatExports.DbgTryGetPhysical(PhysicalWatch[index], out bases[index], out _, out _))
                    return;
            }
            _physicalBases = bases;
            Console.Error.WriteLine($"[DBG][PHYSWATCH] watching {string.Join(",", PhysicalWatch.Select((va, i) => $"va=0x{va:X}->pa=0x{bases[i]:X}"))}");
        }

        if (!Kernel.KernelMemoryCompatExports.DbgTryGetPhysical(address, out var physical, out var regionStart, out var regionLength))
            return;
        for (var index = 0; index < _physicalBases.Length; index++)
        {
            var watched = _physicalBases[index];
            if (physical < watched + 0x7F8000 && watched < physical + size)
                Log($"phys {kind} {address:X} {detail()}", () => $"phys {kind} va=0x{address:X} pa=0x{physical:X} size=0x{size:X} region=0x{regionStart:X}+0x{regionLength:X} watchedVa=0x{PhysicalWatch[index]:X} watchedPa=0x{watched:X} {detail()}");
        }
    }

    public static void Log(string key, Func<string> message)
    {
        lock (Seen)
        {
            var count = Seen.GetValueOrDefault(key);
            Seen[key] = count + 1;
            if ((count & (count - 1)) != 0 || count > 4096)
                return;
            Console.Error.WriteLine($"[DBG][TGTWATCH] n={count + 1} {message()}");
        }
    }
}
