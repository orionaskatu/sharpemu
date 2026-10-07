// TEMP: bisection switches; remove before commit.
namespace SharpEmu.ShaderCompiler;

public static class DbgFlags
{
    private static readonly string Off = Environment.GetEnvironmentVariable("SHARPEMU_DBG_OFF") ?? "";
    private static readonly HashSet<string> OffNames = Off.Split(',', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
    public static bool Disabled(string name) => OffNames.Count != 0 && (Off == "all" || OffNames.Contains(name));

    // TEMP: the memory-table index of the raw read being evaluated, for readback attribution.
    [ThreadStatic] public static long CurrentRawRead;
}
