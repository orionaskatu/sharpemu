// TEMPORARY PROFILING - do not commit.
namespace SharpEmu.Libs.Gpu.Rendering;

internal static class DbgDrawStats
{
    private static readonly Dictionary<(uint, string), long> Counts = new();
    private static long _last = System.Diagnostics.Stopwatch.GetTimestamp();

    public static void Record(uint stages, string outcome)
    {
        lock (Counts)
        {
            Counts.TryGetValue((stages, outcome), out var count);
            Counts[(stages, outcome)] = count + 1;
            var now = System.Diagnostics.Stopwatch.GetTimestamp();
            if (now - _last < System.Diagnostics.Stopwatch.Frequency * 10)
                return;
            _last = now;
            Console.Error.WriteLine("[DBG][DRAWS] " + string.Join(" ", Counts.OrderByDescending(entry => entry.Value)
                .Select(entry => $"stages=0x{entry.Key.Item1:X8}:{entry.Key.Item2}={entry.Value}")));
            Counts.Clear();
        }
    }
}
