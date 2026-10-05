// TEMPORARY DEBUG - do not commit.
namespace SharpEmu.HLE.GpuMemory;

public static class DbgWatch
{
    public static readonly ulong Begin;
    public static readonly ulong End;
    public static Action<string>? Note;

    static DbgWatch()
    {
        var value = Environment.GetEnvironmentVariable("SHARPEMU_DBG_WATCH");
        if (string.IsNullOrWhiteSpace(value)) return;
        var parts = value.Split(':');
        Begin = Convert.ToUInt64(parts[0].Replace("0x", ""), 16);
        End = Begin + Convert.ToUInt64(parts[1].Replace("0x", ""), 16);
    }

    public static bool Hits(ulong address, ulong size) => Note != null && address < End && Begin < address + size;

    public static void Log(ulong address, ulong size, string what)
    {
        if (Hits(address, size)) Note!($"watch {what} 0x{address:X}+0x{size:X}");
    }
}
