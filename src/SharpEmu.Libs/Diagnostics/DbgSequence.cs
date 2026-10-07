// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.Diagnostics;

// TEMP: SHARPEMU_DBG_SEQ_AT=seconds records an ordered event sequence for SHARPEMU_DBG_SEQ_FOR
// seconds (default 3), collapsing consecutive identical events into one line with a count.
internal static class DbgSequence
{
    private static readonly double At = double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_SEQ_AT"), out var at) ? at : -1;
    private static readonly double For = double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_SEQ_FOR"), out var span) ? span : 3;
    private static readonly long Start = Stopwatch.GetTimestamp();
    private static string? _last;
    private static int _repeat;

    // The fog and haze layers (volumetric fog, water mist, deferred and translucent particles composited into the light buffer)
    // are skipped by default: they hazed the whole frame and cost GPU time. SHARPEMU_FOG=1 draws them again.
    private static readonly bool FogOff = Environment.GetEnvironmentVariable("SHARPEMU_FOG") != "1";
    private static readonly string[] FogMarkers = ["Volfog", "Water Particles", "Deferred Particles", "Composite particles", "Translucent Particles"];

    // TEMP: SHARPEMU_DBG_SKIP_MARKER_SCHED="from:len:name|name;from:len:name" skips draws and dispatches recorded under a
    // marker whose name contains one of the names, in that window of process time (needs SHARPEMU_DBG_SEQ_AT set to track markers).
    private static readonly (double From, double Length, string[] Names)[] SkipSchedule =
        (Environment.GetEnvironmentVariable("SHARPEMU_DBG_SKIP_MARKER_SCHED") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Select(item => item.Split(':', 3)).Where(parts => parts.Length == 3)
            .Select(parts => (double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture), parts[2].Split('|'))).ToArray();

    // Innermost PUSH marker name of the current command stream (tracked only while capturing or sequencing).
    public static readonly bool TrackMarkers = Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_FRAME_DIR") is { Length: > 0 } || At >= 0 || FogOff || SkipSchedule.Length != 0;

    [ThreadStatic] private static List<string>? _markers;
    public static string Marker => _markers is { Count: > 0 } ? string.Join("/", _markers.Skip(Math.Max(0, _markers.Count - 2))) : "";
    public static void PushMarker(string text) => (_markers ??= []).Add(text);
    public static void PopMarker() { if (_markers is { Count: > 0 }) _markers.RemoveAt(_markers.Count - 1); }

    public static bool SkipByMarker
    {
        get
        {
            if (_markers is not { Count: > 0 }) return false;
            if (FogOff)
            {
                foreach (var marker in _markers)
                    foreach (var name in FogMarkers)
                        if (marker.Contains(name, StringComparison.Ordinal)) return true;
            }

            if (SkipSchedule.Length == 0) return false;
            var elapsed = Stopwatch.GetElapsedTime(Start).TotalSeconds;
            foreach (var (from, length, names) in SkipSchedule)
            {
                if (elapsed < from || elapsed >= from + length) continue;
                foreach (var marker in _markers)
                    foreach (var name in names)
                        if (marker.Contains(name, StringComparison.Ordinal)) return true;
            }

            return false;
        }
    }

    public static bool Active
    {
        get
        {
            if (At < 0)
                return false;
            var elapsed = Stopwatch.GetElapsedTime(Start).TotalSeconds;
            return elapsed >= At && elapsed < At + For;
        }
    }

    public static void Note(string line)
    {
        lock (typeof(DbgSequence))
        {
            if (line == _last)
            {
                _repeat++;
                return;
            }

            if (_last is not null && _repeat > 0)
                Console.Error.WriteLine($"[DBG][SEQ]   x{_repeat + 1}");
            Console.Error.WriteLine($"[DBG][SEQ] {line}");
            _last = line;
            _repeat = 0;
        }
    }
}
