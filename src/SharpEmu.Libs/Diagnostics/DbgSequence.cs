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
    // are drawn by default; SHARPEMU_FOG=0 skips them.
    private static readonly bool FogOff = Environment.GetEnvironmentVariable("SHARPEMU_FOG") == "0";
    // SHARPEMU_SHADOWS=0 skips every shadow pass: the cascade and particle casters, the slice resolves, the screen-space and
    // occlusion shadows and the combine. Measured: the frame rate does not change (shadows are cheap here) and the lighting
    // then reads stale, never-written shadow data, so the picture is worse. A debugging switch only.
    private static readonly bool ShadowsOff = Environment.GetEnvironmentVariable("SHARPEMU_SHADOWS") == "0";
    private static readonly string[] ShadowMarkers = ["Cascaded shadow map", "Particle Shadow", "Resolve Shadow Map Slice", "Directional Shadow", "Enqueue Shadow Resolve",
        "Screen Space Shadows", "Dirshadow occlusion", "Shadow Wait"];
    private static readonly string[] FogMarkers = ["Volfog", "Water Particles", "Deferred Particles", "Composite particles", "Translucent Particles"];

    // TEMP: SHARPEMU_DBG_SKIP_MARKER_SCHED="from:len:name|name;from:len:name" skips draws and dispatches recorded under a
    // marker whose name contains one of the names, in that window of process time (needs SHARPEMU_DBG_SEQ_AT set to track markers).
    private static readonly (double From, double Length, string[] Names)[] SkipSchedule =
        (Environment.GetEnvironmentVariable("SHARPEMU_DBG_SKIP_MARKER_SCHED") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Select(item => item.Split(':', 3)).Where(parts => parts.Length == 3)
            .Select(parts => (double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture), parts[2].Split('|'))).ToArray();

    // Innermost PUSH marker name of the current command stream (tracked only while capturing or sequencing).
    public static readonly bool TrackMarkers = Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_FRAME_DIR") is { Length: > 0 } || Environment.GetEnvironmentVariable("SHARPEMU_DBG_TALLY") == "1" || Environment.GetEnvironmentVariable("SHARPEMU_DBG_PROBE") == "1" || At >= 0 || FogOff || ShadowsOff || SkipSchedule.Length != 0;

    [ThreadStatic] private static List<string>? _markers;
    public static string Marker => _markers is { Count: > 0 } ? string.Join("/", _markers.Skip(Math.Max(0, _markers.Count - 2))) : "";
    // Particle simulation (emit, update, sort, ribbons, wind) runs on one frame out of SHARPEMU_PARTICLE_EVERY (default 4; 1 = every
    // frame): the many tiny dispatches cost far more than their GPU work. Skipped frames reuse the previous particle buffers.
    private static readonly int ParticleEvery = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_PARTICLE_EVERY"), out var every) ? Math.Max(1, every) : 4;
    private static readonly string[] ParticleSimMarkers = ["Particle Update", "Particle Emit", "Particle No Deps", "Particle Sort", "Particle Ribbon",
        "Particle Post-Update", "ParticleWind", "Particle Wait For Velocity", "Particle CS Frame"];
    private static int _frames;

    public static int Frames => Volatile.Read(ref _frames);
    public static event Action? FrameStarted;

    public static void PushMarker(string text)
    {
        (_markers ??= []).Add(text);
        if (text == "Frame") { Interlocked.Increment(ref _frames); FrameStarted?.Invoke(); }
        if (TallyOn) { lock (Tallies) { var key = "P " + text; Tallies.TryGetValue(key, out var entry); Tallies[key] = (entry.Count + 1, entry.Ticks); } }
    }
    public static void PopMarker()
    {
        if (TallyOn) { lock (Tallies) { var key = "P- " + (_markers is { Count: > 0 } ? _markers[^1] : "(empty)"); Tallies.TryGetValue(key, out var entry); Tallies[key] = (entry.Count + 1, entry.Ticks); } }
        if (_markers is { Count: > 0 }) _markers.RemoveAt(_markers.Count - 1);
    }

    // TEMP: SHARPEMU_DBG_TALLY=1 counts draws and dispatches per shader (and per innermost marker) and attributes the wall time
    // between consecutive ops to the earlier one; printed every 10 s.
    public static readonly bool TallyOn = Environment.GetEnvironmentVariable("SHARPEMU_DBG_TALLY") == "1";
    private static readonly Dictionary<string, (long Count, long Ticks)> Tallies = [];
    private static long _tallyNext;
    [ThreadStatic] private static string? _tallyPrev;
    [ThreadStatic] private static long _tallyPrevAt;

    public static void Tally(string kind, ulong hash)
    {
        if (!TallyOn) return;
        var now = Stopwatch.GetTimestamp();
        var marker = _markers is { Count: > 0 } ? _markers[^1] : "";
        lock (Tallies)
        {
            if (_tallyPrev is not null && now - _tallyPrevAt < Stopwatch.Frequency / 4)
                foreach (var key in _tallyPrev.Split('|'))
                {
                    Tallies.TryGetValue(key, out var entry);
                    Tallies[key] = (entry.Count, entry.Ticks + now - _tallyPrevAt);
                }

            _tallyPrev = $"{kind} {hash:X16}|M {marker}|X {kind} {hash:X16} {marker}";
            _tallyPrevAt = now;
            foreach (var key in _tallyPrev.Split('|'))
            {
                Tallies.TryGetValue(key, out var entry);
                Tallies[key] = (entry.Count + 1, entry.Ticks);
            }

            if (now < _tallyNext) return;
            var first = _tallyNext == 0;
            _tallyNext = now + 10 * Stopwatch.Frequency;
            if (!first)
            {
                Console.Error.WriteLine($"[DBG][TALLY] t={Stopwatch.GetElapsedTime(Start).TotalSeconds:F0} by wall time (ms, count)");
                foreach (var pair in Tallies.OrderByDescending(pair => pair.Key.StartsWith("P") ? long.MaxValue : pair.Value.Ticks).Take(700))
                    Console.Error.WriteLine($"[DBG][TALLY]   {pair.Value.Ticks * 1000 / Stopwatch.Frequency,7}ms {pair.Value.Count,7} {pair.Key}");
            }

            Tallies.Clear();
        }
    }

    public static bool SkipByMarker
    {
        get
        {
            if (_markers is not { Count: > 0 }) return false;
            if (ParticleEvery > 1 && (Volatile.Read(ref _frames) % ParticleEvery) != 0 && Array.IndexOf(ParticleSimMarkers, _markers[^1]) >= 0) return true;
            if (ShadowsOff)
            {
                foreach (var marker in _markers)
                    foreach (var name in ShadowMarkers)
                        if (marker.Contains(name, StringComparison.Ordinal)) return true;
            }

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
