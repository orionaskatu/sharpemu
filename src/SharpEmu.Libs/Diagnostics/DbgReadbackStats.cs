// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.Diagnostics;

// TEMP: SHARPEMU_DBG_RB_STATS=1 attributes shader-resource readbacks to the shader whose
// resources were being materialized and to the 64 KiB region read, reported every 5 seconds.
internal static class DbgReadbackStats
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_DBG_RB_STATS") == "1";

    [ThreadStatic] public static ulong CurrentShader;
    [ThreadStatic] public static string? CurrentLabel;

    private static readonly Dictionary<(ulong Shader, string Label, ulong Region, long Read), (int Count, double Ms)> _stats = new();
    private static long _windowStart = Stopwatch.GetTimestamp();

    public static void Record(ulong address, double ms)
    {
        var key = (CurrentShader, CurrentLabel ?? "?", address, SharpEmu.ShaderCompiler.DbgFlags.CurrentRawRead);
        lock (_stats)
        {
            var value = _stats.GetValueOrDefault(key);
            _stats[key] = (value.Count + 1, value.Ms + ms);
            if (Stopwatch.GetElapsedTime(_windowStart).TotalSeconds < 5)
                return;

            _windowStart = Stopwatch.GetTimestamp();
            var total = _stats.Values.Sum(v => v.Ms);
            var count = _stats.Values.Sum(v => v.Count);
            Console.Error.WriteLine($"[DBG][RBSTATS] readbacks={count} ms={total:F1}");
            foreach (var (k, v) in _stats.OrderByDescending(p => p.Value.Ms).Take(25))
                Console.Error.WriteLine($"[DBG][RBSTATS]   shader=0x{k.Shader:X16} {k.Label} addr=0x{k.Region:X} memory={k.Read} n={v.Count} ms={v.Ms:F1}");
            _stats.Clear();
        }
    }
}
