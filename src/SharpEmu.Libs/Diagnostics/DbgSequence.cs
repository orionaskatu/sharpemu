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

    // Innermost PUSH marker name of the current command stream (tracked only while capturing or sequencing).
    public static readonly bool TrackMarkers = Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_FRAME_DIR") is { Length: > 0 } || At >= 0;
    [ThreadStatic] private static List<string>? _markers;
    public static string Marker => _markers is { Count: > 0 } ? string.Join("/", _markers.Skip(Math.Max(0, _markers.Count - 2))) : "";
    public static void PushMarker(string text) => (_markers ??= []).Add(text);
    public static void PopMarker() { if (_markers is { Count: > 0 }) _markers.RemoveAt(_markers.Count - 1); }

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
