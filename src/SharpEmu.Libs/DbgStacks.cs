// TEMP: aggregates call stacks by tag for profiling; remove before commit.
namespace SharpEmu.Libs;

internal static class DbgStacks
{
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_DBG_STACKS") == "1";
    private static readonly Dictionary<string, (int Count, double Ms, ulong Bytes)> Stacks = new();
    private static long _last = Environment.TickCount64;

    public static long Start() => Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

    public static void Record(string tag, long start, ulong bytes = 0, int depth = 10)
    {
        if (!Enabled) return;
        var ms = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        var frames = new System.Diagnostics.StackTrace(1, false).GetFrames();
        var key = tag + " " + string.Join(" < ", frames.Take(depth).Select(f => f.GetMethod() is { } m ? $"{m.DeclaringType?.Name}.{m.Name}" : "?"));
        lock (Stacks)
        {
            Stacks.TryGetValue(key, out var e);
            Stacks[key] = (e.Count + 1, e.Ms + ms, e.Bytes + bytes);
            if (Environment.TickCount64 - _last < 10000) return;
            _last = Environment.TickCount64;
            foreach (var (k, v) in Stacks.OrderByDescending(x => x.Value.Ms).Take(15))
                Console.Error.WriteLine($"[DBG][STACK] n={v.Count} ms={v.Ms:F0} bytes={v.Bytes} {k}");
            Console.Error.WriteLine("[DBG][STACK] ----");
            Stacks.Clear();
        }
    }
}
