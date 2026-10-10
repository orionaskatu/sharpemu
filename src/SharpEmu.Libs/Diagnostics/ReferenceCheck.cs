// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Reference;

namespace SharpEmu.Libs.Diagnostics;

// TEMP: SHARPEMU_DBG_REF_CS=hash[,hash] runs the first matching compute dispatch after
// SHARPEMU_DBG_REF_AT seconds (default 0) on the CPU reference interpreter before the GPU runs it,
// then compares every word the reference wrote with what the GPU wrote. Groups past
// SHARPEMU_DBG_REF_GROUPS (default 1) are not interpreted; their writes are not compared.
internal static class ReferenceCheck
{
    public static readonly HashSet<ulong> Hashes = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_CS") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(text => Convert.ToUInt64(text.Replace("0x", ""), 16)).ToHashSet();
    public static readonly bool Enabled = Hashes.Count != 0;
    private static readonly double At = double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_AT"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var at) ? at : 0;
    private static readonly int Groups = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_GROUPS"), out var groups) ? groups : 1;
    private static readonly int Count = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_COUNT"), out var count) ? count : 1;
    private static readonly Stopwatch Clock = Stopwatch.StartNew();
    private static readonly Dictionary<ulong, int> _done = new();
    public static readonly ConcurrentDictionary<ulong, (ulong Hash, Gen5ShaderProgram Program)> Programs = new();
    public static (ulong Hash, Gen5ReferenceInterpreter Interpreter)? Pending;

    public static Gen5ReferenceInterpreter? Begin(ulong address, Gen5ReferenceInterpreter.MemoryReader read, uint[] userData,
        uint localX, uint localY, uint localZ, uint groupsX, uint groupsY, uint groupsZ, Gen5ComputeSystemRegisters? systemRegisters,
        int threadIdCount, int waveSize)
    {
        if (!Enabled || Clock.Elapsed.TotalSeconds < At) return null;
        if (!Programs.TryGetValue(address, out var entry) || !Hashes.Contains(entry.Hash)) return null;
        var (hash, program) = entry;
        var done = _done.GetValueOrDefault(hash);
        if (done >= Count) return null;
        if (done == 0 && Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_DISASM") is { Length: > 0 } disasmDir) // TEMP: dump decoded program
        {
            static string Op(Gen5Operand o) => $"{o.Kind}:{o.Value}";
            System.IO.File.WriteAllLines(System.IO.Path.Combine(disasmDir, $"cs_{hash:X16}.txt"), program.Instructions.Select(i =>
                $"{i.Pc:X4} {i.Opcode} d[{string.Join(' ', i.Destinations.Select(Op))}] s[{string.Join(' ', i.Sources.Select(Op))}] {i.Control}"));
        }
        _done[hash] = done + 1;
        var interpreter = new Gen5ReferenceInterpreter(program, read, waveSize);
        if (Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_TRACE") is { Length: > 0 } traceSpec) // TEMP: log the wave storing to this address
            interpreter.TraceAddress = Convert.ToUInt64(traceSpec, 16);
        var watch = Stopwatch.StartNew();
        try
        {
            var interpreted = 0;
            for (uint z = 0; z < groupsZ && interpreted < Groups; z++)
            for (uint y = 0; y < groupsY && interpreted < Groups; y++)
            for (uint x = 0; x < groupsX && interpreted < Groups; x++, interpreted++)
                interpreter.RunWorkgroup(userData, localX, localY, localZ, x, y, z, systemRegisters, threadIdCount);
            Console.Error.WriteLine($"[DBG][REF] cs=0x{hash:X16} interpreted groups={interpreted}/{groupsX * groupsY * groupsZ} instructions={interpreter.ExecutedInstructions} ms={watch.ElapsedMilliseconds} failedPages={interpreter.FailedPageReads}");
            if (interpreter.Trace is { } trace && Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_DISASM") is { Length: > 0 } traceDir)
                System.IO.File.WriteAllLines(System.IO.Path.Combine(traceDir, $"trace_{hash:X16}_{done}.txt"), trace);
            if (Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_SCALARS") == "1") // TEMP: distinct scalar constants as floats
            {
                var seen = new HashSet<ulong>();
                var line = new System.Text.StringBuilder();
                foreach (var (pc, loadAddress, value) in interpreter.ScalarLoads)
                {
                    if (!seen.Add(loadAddress) || seen.Count > 400) continue;
                    var f = BitConverter.UInt32BitsToSingle(value);
                    line.Append(float.IsFinite(f) && Math.Abs(f) > 1e-6 && Math.Abs(f) < 1e6 ? $" {pc:X}:{f:G5}" : "");
                }

                Console.Error.WriteLine($"[DBG][REF] cs=0x{hash:X16} scalars:{line}");
            }

            if (interpreter.LastLds is { } lds && Environment.GetEnvironmentVariable("SHARPEMU_DBG_REF_LDS") is { Length: > 0 } ldsSpec) // TEMP: print LDS floats [from,to)
            {
                var parts = ldsSpec.Split(':');
                var from = int.Parse(parts[0]); var to = int.Parse(parts[1]);
                var floats = string.Join(" ", Enumerable.Range(0, (to - from) / 4).Select(i => BitConverter.ToSingle(lds, from + i * 4).ToString("G6")));
                Console.Error.WriteLine($"[DBG][REF] cs=0x{hash:X16} lds[{from}..{to})= {floats}");
            }
            return interpreter;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"[DBG][REF] cs=0x{hash:X16} interpreter failed: {exception.GetType().Name}: {exception.Message}");
            Console.Error.WriteLine($"[DBG][REF] cs=0x{hash:X16} opcodes: {string.Join(',', program.Instructions.Select(i => i.Opcode).Distinct().Order())}");
            return null;
        }
    }

    public static void Compare(ulong hash, Gen5ReferenceInterpreter interpreter, Func<ulong, uint?> readGpuWord)
    {
        long total = 0, mismatched = 0, logged = 0;
        var byPc = new Dictionary<ulong, (long Count, ulong FirstAddress, uint Cpu, uint Gpu)>();
        foreach (var (address, value, pc) in interpreter.WrittenWords())
        {
            total++;
            var gpu = readGpuWord(address);
            if (gpu is not { } g) continue;
            if (g == value) continue;
            var cpuFloat = BitConverter.UInt32BitsToSingle(value);
            var gpuFloat = BitConverter.UInt32BitsToSingle(g);
            if (float.IsFinite(cpuFloat) && float.IsFinite(gpuFloat) && Math.Abs(cpuFloat - gpuFloat) <= 1e-3f * Math.Max(1f, Math.Abs(cpuFloat)))
                continue;
            mismatched++;
            byPc[pc] = byPc.TryGetValue(pc, out var entry) ? (entry.Count + 1, entry.FirstAddress, entry.Cpu, entry.Gpu) : (1, address, value, g);
            if (logged++ < 24)
                Console.Error.WriteLine($"[DBG][REF] mismatch addr=0x{address:X} pc=0x{pc:X} cpu=0x{value:X8}({cpuFloat}) gpu=0x{g:X8}({gpuFloat})");
        }

        Console.Error.WriteLine($"[DBG][REF] cs=0x{hash:X16} compared words={total} mismatched={mismatched}");
        foreach (var (pc, entry) in byPc.OrderByDescending(pair => pair.Value.Count).Take(20))
            Console.Error.WriteLine($"[DBG][REF]   store pc=0x{pc:X} mismatches={entry.Count} first=0x{entry.FirstAddress:X} cpu=0x{entry.Cpu:X8} gpu=0x{entry.Gpu:X8}");
    }
}
