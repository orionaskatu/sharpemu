// TEMP: bisection switches; remove before commit.
namespace SharpEmu.ShaderCompiler;

public static class DbgFlags
{
    private static readonly string Off = Environment.GetEnvironmentVariable("SHARPEMU_DBG_OFF") ?? "";
    public static bool Disabled(string name) => Off == "all" || Off.Split(',').Contains(name);
}
