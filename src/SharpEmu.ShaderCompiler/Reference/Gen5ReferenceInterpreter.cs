// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;

namespace SharpEmu.ShaderCompiler.Reference;

// A slow, direct CPU execution of a decoded Gen5 compute program, for differential checks of the
// SPIR-V translation: one workgroup at a time, every wave and lane modeled explicitly, guest
// memory read through a callback and every write kept in an overlay. It implements the opcodes
// the programs under test use and throws NotSupportedException on anything else, so a gap is
// reported instead of producing a silently wrong reference.
public sealed class Gen5ReferenceInterpreter
{
    public delegate bool MemoryReader(ulong address, Span<byte> destination);

    private const int Vcc = 106;
    private const int M0 = 124;
    private const int Exec = 126;

    private readonly Gen5ShaderProgram _program;
    private readonly Dictionary<ulong, int> _indexOfPc = new();
    private readonly MemoryReader _read;
    private readonly int _waveSize;
    private readonly Dictionary<ulong, byte[]> _pages = new();
    private readonly Dictionary<ulong, ulong> _pageWritten = new();
    private readonly Dictionary<ulong, ulong> _lastWriterPc = new();
    private byte[] _lds = new byte[65536];

    public long ExecutedInstructions { get; private set; }

    // Scalar memory loads of the first workgroup (pc, address, value), for inspecting constants.
    public List<(ulong Pc, ulong Address, uint Value)> ScalarLoads { get; } = new();

    // The LDS contents when the last workgroup finished.
    public byte[]? LastLds { get; private set; }

    public Gen5ReferenceInterpreter(Gen5ShaderProgram program, MemoryReader read, int waveSize = 64)
    {
        _program = program;
        _read = read;
        _waveSize = waveSize;
        for (var index = 0; index < program.Instructions.Count; index++)
            _indexOfPc[program.Instructions[index].Pc] = index;
    }

    // Writes as (address of a 4-byte word, value, pc of the last write). Partial words are filled
    // from guest memory.
    public IEnumerable<(ulong Address, uint Value, ulong Pc)> WrittenWords()
    {
        var words = new SortedSet<ulong>();
        foreach (var (address, _) in _lastWriterPc) words.Add(address & ~3UL);
        var word = new byte[4];
        foreach (var address in words)
        {
            ReadBytes(address, word);
            _lastWriterPc.TryGetValue(address, out var pc);
            for (var b = 0UL; b < 4 && pc == 0; b++) _lastWriterPc.TryGetValue(address + b, out pc);
            yield return (address, BinaryPrimitives.ReadUInt32LittleEndian(word), pc);
        }
    }

    private sealed class Wave
    {
        public readonly uint[] S = new uint[128];
        public uint[] V = null!;
        public bool Scc;
        public int Pc;
        public bool Done;
        public bool AtBarrier;
        public ulong ExecMask { get => S[Exec] | ((ulong)S[Exec + 1] << 32); set { S[Exec] = (uint)value; S[Exec + 1] = (uint)(value >> 32); } }
        public bool Wave32;
        public List<string>? Log;
        public ulong VccMask
        {
            get => Wave32 ? S[Vcc] : S[Vcc] | ((ulong)S[Vcc + 1] << 32);
            set { S[Vcc] = (uint)value; if (!Wave32) S[Vcc + 1] = (uint)(value >> 32); }
        }
    }

    // When set, the instruction log of the wave that first stores to this address is kept in Trace.
    public ulong TraceAddress { get; set; }
    public List<string>? Trace { get; private set; }
    public int FailedPageReads { get; private set; }
    private Wave? _currentWave;

    // Runs one workgroup: user data in s0..s[n-1], then the workgroup id registers, thread ids in v0..v2.
    public void RunWorkgroup(uint[] userData, uint localX, uint localY, uint localZ, uint groupX, uint groupY, uint groupZ,
        Gen5ComputeSystemRegisters? systemRegisters, int threadIdCount, int maxInstructions = 400_000_000)
    {
        var threads = (int)(localX * localY * localZ);
        var waveCount = (threads + _waveSize - 1) / _waveSize;
        var waves = new Wave[waveCount];
        Array.Clear(_lds);
        for (var w = 0; w < waveCount; w++)
        {
            var wave = new Wave { V = new uint[256 * _waveSize], Wave32 = _waveSize == 32, Log = TraceAddress != 0 && Trace == null ? new List<string>() : null };
            Array.Copy(userData, wave.S, Math.Min(userData.Length, 104));
            if (systemRegisters is { } regs)
            {
                if (regs.WorkGroupXRegister is { } rx) wave.S[rx] = groupX;
                if (regs.WorkGroupYRegister is { } ry) wave.S[ry] = groupY;
                if (regs.WorkGroupZRegister is { } rz) wave.S[rz] = groupZ;
                if (regs.ThreadGroupSizeRegister is { } rt) wave.S[rt] = (uint)waveCount | ((uint)w << 6);
            }

            ulong exec = 0;
            for (var lane = 0; lane < _waveSize; lane++)
            {
                var flat = w * _waveSize + lane;
                if (flat >= threads) continue;
                exec |= 1UL << lane;
                var x = (uint)(flat % localX);
                var y = (uint)(flat / localX % localY);
                var z = (uint)(flat / (localX * localY));
                wave.V[0 * _waveSize + lane] = x;
                if (threadIdCount > 1) wave.V[1 * _waveSize + lane] = y;
                if (threadIdCount > 2) wave.V[2 * _waveSize + lane] = z;
            }

            wave.ExecMask = exec;
            waves[w] = wave;
        }

        while (true)
        {
            var progressed = false;
            foreach (var wave in waves)
            {
                if (wave.Done || wave.AtBarrier) continue;
                while (!wave.Done && !wave.AtBarrier)
                {
                    _currentWave = wave;
                    var tracePc = wave.Pc;
                    Step(wave);
                    if (wave.Log != null) wave.Log.Add(DescribeStep(wave, tracePc));
                    progressed = true;
                    if (++ExecutedInstructions > maxInstructions)
                        throw new InvalidOperationException("instruction budget exceeded");
                }
            }

            if (waves.All(static w => w.Done)) { LastLds = (byte[])_lds.Clone(); return; }
            if (waves.All(static w => w.Done || w.AtBarrier))
            {
                foreach (var wave in waves) wave.AtBarrier = false;
                continue;
            }

            if (!progressed) throw new InvalidOperationException("waves deadlocked");
        }
    }

    // ---------------- memory ----------------
    private byte[] Page(ulong page)
    {
        if (_pages.TryGetValue(page, out var data)) return data;
        data = new byte[4096];
        if (!_read(page << 12, data)) { Array.Clear(data); FailedPageReads++; }
        _pages[page] = data;
        return data;
    }

    private void ReadBytes(ulong address, Span<byte> destination)
    {
        for (var i = 0; i < destination.Length; i++)
        {
            var a = address + (ulong)i;
            destination[i] = Page(a >> 12)[(int)(a & 0xFFF)];
        }
    }

    private void WriteBytes(ulong address, ReadOnlySpan<byte> source, ulong pc)
    {
        for (var i = 0; i < source.Length; i++)
        {
            var a = address + (ulong)i;
            Page(a >> 12)[(int)(a & 0xFFF)] = source[i];
            _lastWriterPc[a] = pc;
            if (a == TraceAddress && Trace == null && _currentWave?.Log is { } log) Trace = new List<string>(log) { $"store pc=0x{pc:X} byte={source[i]:X2}" };
        }
    }

    private uint ReadU32(ulong address) { Span<byte> b = stackalloc byte[4]; ReadBytes(address, b); return BinaryPrimitives.ReadUInt32LittleEndian(b); }
    private void WriteU32(ulong address, uint value, ulong pc) { Span<byte> b = stackalloc byte[4]; BinaryPrimitives.WriteUInt32LittleEndian(b, value); WriteBytes(address, b, pc); }

    // ---------------- operands ----------------
    private static float F(uint v) => BitConverter.UInt32BitsToSingle(v);
    private static uint ReverseBits(uint v)
    {
        uint r = 0;
        for (var i = 0; i < 32; i++) r |= ((v >> i) & 1) << (31 - i);
        return r;
    }
    private static uint U(float v) => BitConverter.SingleToUInt32Bits(v);

    private uint ReadScalarOperand(Wave wave, Gen5Operand operand)
    {
        switch (operand.Kind)
        {
            case Gen5OperandKind.ScalarRegister: return wave.S[operand.Value];
            case Gen5OperandKind.LiteralConstant: return operand.Value;
            case Gen5OperandKind.EncodedConstant:
                if (operand.Value == 251) return wave.VccMask == 0 ? 1u : 0u;
                if (operand.Value == 252) return wave.ExecMask == 0 ? 1u : 0u;
                if (operand.Value == 253) return wave.Scc ? 1u : 0u;
                if (Gen5InlineConstants.TryDecode(operand.Value, out var value)) return value;
                throw new NotSupportedException($"constant {operand.Value}");
            default: throw new NotSupportedException($"scalar read of {operand}");
        }
    }

    private uint Read(Wave wave, Gen5Operand operand, int lane) =>
        operand.Kind == Gen5OperandKind.VectorRegister ? wave.V[operand.Value * (uint)_waveSize + (uint)lane] : ReadScalarOperand(wave, operand);

    private ulong Read64(Wave wave, Gen5Operand operand, int lane)
    {
        switch (operand.Kind)
        {
            case Gen5OperandKind.VectorRegister:
                return wave.V[operand.Value * (uint)_waveSize + (uint)lane] | ((ulong)wave.V[(operand.Value + 1) * (uint)_waveSize + (uint)lane] << 32);
            case Gen5OperandKind.ScalarRegister:
                return wave.S[operand.Value] | ((ulong)wave.S[operand.Value + 1] << 32);
            case Gen5OperandKind.LiteralConstant:
                return operand.Value;
            default:
                if (operand.Value is >= 193 and <= 208) return unchecked((ulong)(long)(int)ReadScalarOperand(wave, operand));
                return ReadScalarOperand(wave, operand);
        }
    }

    private void SetV(Wave wave, uint register, int lane, uint value) => wave.V[register * (uint)_waveSize + (uint)lane] = value;
    private uint GetV(Wave wave, uint register, int lane) => wave.V[register * (uint)_waveSize + (uint)lane];

    private void SetS(Wave wave, uint register, uint value) { if (register < 128) wave.S[register] = value; }
    private void SetS64(Wave wave, uint register, ulong value) { SetS(wave, register, (uint)value); SetS(wave, register + 1, (uint)(value >> 32)); }

    // Lane masks are one SGPR in wave32: vcc_hi and the register after an SDST are left alone.
    private void SetMask(Wave wave, uint register, ulong value)
    {
        if (_waveSize == 32) SetS(wave, register, (uint)value);
        else SetS64(wave, register, value);
    }

    // ---------------- step ----------------
    private string DescribeStep(Wave wave, int index)
    {
        if (index >= _program.Instructions.Count) return "end";
        var instruction = _program.Instructions[index];
        var text = new System.Text.StringBuilder($"{instruction.Pc:X4} {instruction.Opcode} exec={wave.ExecMask:X} scc={(wave.Scc ? 1 : 0)} vcc={wave.VccMask:X}");
        foreach (var destination in instruction.Destinations)
        {
            if (destination.Kind == Gen5OperandKind.ScalarRegister && destination.Value < 128) text.Append($" s{destination.Value}={wave.S[destination.Value]:X}");
            else if (destination.Kind == Gen5OperandKind.VectorRegister)
            {
                text.Append($" v{destination.Value}=");
                for (var lane = 0; lane < Math.Min(_waveSize, 32); lane++) text.Append($"{GetV(wave, destination.Value, lane):X},");
            }
        }
        var sdst = instruction.Control switch { Gen5Vop3Control { ScalarDestination: { } v } => v, Gen5SdwaControl { ScalarDestination: { } v } => v, _ => (uint?)null };
        if (sdst is { } sd && sd < 128) text.Append($" sdst s{sd}={wave.S[sd]:X}");
        return text.ToString();
    }

    private void Step(Wave wave)
    {
        if (wave.Pc >= _program.Instructions.Count) { wave.Done = true; return; }
        var instruction = _program.Instructions[wave.Pc];
        var next = wave.Pc + 1;
        var op = instruction.Opcode;
        switch (instruction.Encoding)
        {
            case Gen5ShaderEncoding.Sopp:
                next = ExecuteSopp(wave, instruction, next);
                break;
            case Gen5ShaderEncoding.Sop1:
            case Gen5ShaderEncoding.Sop2:
            case Gen5ShaderEncoding.Sopc:
            case Gen5ShaderEncoding.Sopk:
                ExecuteScalar(wave, instruction);
                break;
            case Gen5ShaderEncoding.Smem:
            case Gen5ShaderEncoding.Smrd:
                ExecuteScalarMemory(wave, instruction);
                break;
            case Gen5ShaderEncoding.Ds:
                ExecuteDataShare(wave, instruction);
                break;
            case Gen5ShaderEncoding.Mubuf:
            case Gen5ShaderEncoding.Mtbuf:
                ExecuteBuffer(wave, instruction);
                break;
            case Gen5ShaderEncoding.Flat:
                ExecuteGlobal(wave, instruction);
                break;
            default:
                ExecuteVector(wave, instruction);
                break;
        }

        wave.Pc = next;
        if (op == "SEndpgm") wave.Done = true;
    }

    private int Target(Gen5ShaderInstruction instruction)
    {
        var simm = (short)(instruction.Words[0] & 0xFFFF);
        var pc = (long)instruction.Pc + 4 + simm * 4L;
        return _indexOfPc.TryGetValue((ulong)pc, out var index) ? index : throw new InvalidOperationException($"branch target 0x{pc:X}");
    }

    private int ExecuteSopp(Wave wave, Gen5ShaderInstruction instruction, int next)
    {
        switch (instruction.Opcode)
        {
            case "SEndpgm": wave.Done = true; return next;
            case "SBarrier": wave.AtBarrier = true; return next;
            case "SBranch": return Target(instruction);
            case "SCbranchScc0": return !wave.Scc ? Target(instruction) : next;
            case "SCbranchScc1": return wave.Scc ? Target(instruction) : next;
            case "SCbranchVccz": return wave.VccMask == 0 ? Target(instruction) : next;
            case "SCbranchVccnz": return wave.VccMask != 0 ? Target(instruction) : next;
            case "SCbranchExecz": return wave.ExecMask == 0 ? Target(instruction) : next;
            case "SCbranchExecnz": return wave.ExecMask != 0 ? Target(instruction) : next;
            case "SWaitcnt": case "SNop": case "SInstPrefetch": case "SSetprio": case "SSleep": case "SWaitcntDepctr": case "SClause": case "SCodeEnd":
                return next;
            default: throw new NotSupportedException(instruction.Opcode);
        }
    }

    // ---------------- scalar ALU ----------------
    private void ExecuteScalar(Wave wave, Gen5ShaderInstruction instruction)
    {
        var op = instruction.Opcode;
        var dst = instruction.Destinations.Count > 0 ? instruction.Destinations[0].Value : 0u;
        uint A() => ReadScalarOperand(wave, instruction.Sources[0]);
        uint B() => ReadScalarOperand(wave, instruction.Sources[1]);
        ulong A64() => Read64(wave, instruction.Sources[0], 0);
        ulong B64() => Read64(wave, instruction.Sources[1], 0);
        void R32(uint value) { SetS(wave, dst, value); wave.Scc = value != 0; }
        void R64(ulong value) { SetS64(wave, dst, value); wave.Scc = value != 0; }
        switch (op)
        {
            case "SMovB32": SetS(wave, dst, A()); return;
            case "SMovB64": SetS64(wave, dst, A64()); return;
            case "SMovkI32": SetS(wave, dst, (uint)(int)(short)(instruction.Words[0] & 0xFFFF)); return;
            case "SCmovB32": if (wave.Scc) SetS(wave, dst, A()); return;
            case "SCmovB64": if (wave.Scc) SetS64(wave, dst, A64()); return;
            case "SNotB32": R32(~A()); return;
            case "SNotB64": R64(~A64()); return;
            case "SAndB32": R32(A() & B()); return;
            case "SOrB32": R32(A() | B()); return;
            case "SXorB32": R32(A() ^ B()); return;
            case "SAndn2B32": R32(A() & ~B()); return;
            case "SOrn2B32": R32(A() | ~B()); return;
            case "SNorB32": R32(~(A() | B())); return;
            case "SNandB32": R32(~(A() & B())); return;
            case "SXnorB32": R32(~(A() ^ B())); return;
            case "SAndB64": R64(A64() & B64()); return;
            case "SOrB64": R64(A64() | B64()); return;
            case "SXorB64": R64(A64() ^ B64()); return;
            case "SAndn2B64": R64(A64() & ~B64()); return;
            case "SOrn2B64": R64(A64() | ~B64()); return;
            case "SNorB64": R64(~(A64() | B64())); return;
            case "SNandB64": R64(~(A64() & B64())); return;
            case "SXnorB64": R64(~(A64() ^ B64())); return;
            case "SLshlB32": R32(A() << (int)(B() & 31)); return;
            case "SLshrB32": R32(A() >> (int)(B() & 31)); return;
            case "SAshrI32": R32((uint)((int)A() >> (int)(B() & 31))); return;
            case "SLshlB64": R64(A64() << (int)(B() & 63)); return;
            case "SLshrB64": R64(A64() >> (int)(B() & 63)); return;
            case "SLshl1AddU32": case "SLshl2AddU32": case "SLshl3AddU32": case "SLshl4AddU32":
            {
                var shift = op[5] - '0';
                var shifted = (ulong)A() << shift;
                var sum = shifted + B();
                SetS(wave, dst, (uint)sum); wave.Scc = sum > uint.MaxValue; return;
            }
            case "SAddU32": { var s = (ulong)A() + B(); SetS(wave, dst, (uint)s); wave.Scc = s > uint.MaxValue; return; }
            case "SSubU32": { var a = A(); var b = B(); SetS(wave, dst, a - b); wave.Scc = b > a; return; }
            case "SAddcU32": { var s = (ulong)A() + B() + (wave.Scc ? 1u : 0u); SetS(wave, dst, (uint)s); wave.Scc = s > uint.MaxValue; return; }
            case "SSubbU32": { var a = A(); var b = B(); var c = wave.Scc ? 1u : 0u; SetS(wave, dst, a - b - c); wave.Scc = (ulong)b + c > a; return; }
            case "SAddI32": { var a = (int)A(); var b = (int)B(); var r = unchecked(a + b); SetS(wave, dst, (uint)r); wave.Scc = ((a ^ r) & (b ^ r)) < 0; return; }
            case "SSubI32": { var a = (int)A(); var b = (int)B(); var r = unchecked(a - b); SetS(wave, dst, (uint)r); wave.Scc = ((a ^ b) & (a ^ r)) < 0; return; }
            case "SMulI32": SetS(wave, dst, unchecked((uint)((int)A() * (int)B()))); return;
            case "SMulHiU32": SetS(wave, dst, (uint)(((ulong)A() * B()) >> 32)); return;
            case "SMulHiI32": SetS(wave, dst, (uint)(((long)(int)A() * (int)B()) >> 32)); return;
            case "SMinI32": { var a = (int)A(); var b = (int)B(); SetS(wave, dst, (uint)Math.Min(a, b)); wave.Scc = a < b; return; }
            case "SMaxI32": { var a = (int)A(); var b = (int)B(); SetS(wave, dst, (uint)Math.Max(a, b)); wave.Scc = a > b; return; }
            case "SMinU32": { var a = A(); var b = B(); SetS(wave, dst, Math.Min(a, b)); wave.Scc = a < b; return; }
            case "SMaxU32": { var a = A(); var b = B(); SetS(wave, dst, Math.Max(a, b)); wave.Scc = a > b; return; }
            case "SCselectB32": SetS(wave, dst, wave.Scc ? A() : B()); return;
            case "SCselectB64": SetS64(wave, dst, wave.Scc ? A64() : B64()); return;
            case "SBfmB32": SetS(wave, dst, ((1u << (int)(A() & 31)) - 1) << (int)(B() & 31)); return;
            case "SBfeU32": { var b = B(); var off = (int)(b & 31); var w = (int)((b >> 16) & 0x7F); var v = w == 0 ? 0 : (A() >> off) & (w >= 32 ? uint.MaxValue : (1u << w) - 1); R32(v); return; }
            case "SBfeI32": { var b = B(); var off = (int)(b & 31); var w = (int)((b >> 16) & 0x7F); var v = w == 0 ? 0 : (int)(A() << (32 - off - w)) >> (32 - w); R32((uint)v); return; }
            case "SBitcmp0B32": wave.Scc = ((A() >> (int)(B() & 31)) & 1) == 0; return;
            case "SBitcmp1B32": wave.Scc = ((A() >> (int)(B() & 31)) & 1) != 0; return;
            case "SBcnt1I32B32": R32((uint)System.Numerics.BitOperations.PopCount(A())); return;
            case "SBcnt1I32B64": R32((uint)System.Numerics.BitOperations.PopCount(A64())); return;
            case "SFF1I32B32": { var a = A(); SetS(wave, dst, a == 0 ? uint.MaxValue : (uint)System.Numerics.BitOperations.TrailingZeroCount(a)); return; }
            case "SFF1I32B64": { var a = A64(); SetS(wave, dst, a == 0 ? uint.MaxValue : (uint)System.Numerics.BitOperations.TrailingZeroCount(a)); return; }
            case "SFlbitI32B32": { var a = A(); SetS(wave, dst, a == 0 ? uint.MaxValue : (uint)System.Numerics.BitOperations.LeadingZeroCount(a)); return; }
            case "SAbsI32": { var v = (uint)Math.Abs((long)(int)A()); R32(v); return; }
            case "SAndSaveexecB64": case "SOrSaveexecB64": case "SXorSaveexecB64": case "SAndn2SaveexecB64": case "SOrn2SaveexecB64":
            case "SNandSaveexecB64": case "SNorSaveexecB64": case "SXnorSaveexecB64": case "SAndn1SaveexecB64": case "SOrn1SaveexecB64":
            {
                var s0 = A64();
                var exec = wave.ExecMask;
                var result = op switch
                {
                    "SAndSaveexecB64" => s0 & exec,
                    "SOrSaveexecB64" => s0 | exec,
                    "SXorSaveexecB64" => s0 ^ exec,
                    "SAndn2SaveexecB64" => s0 & ~exec,
                    "SOrn2SaveexecB64" => s0 | ~exec,
                    "SNandSaveexecB64" => ~(s0 & exec),
                    "SNorSaveexecB64" => ~(s0 | exec),
                    "SXnorSaveexecB64" => ~(s0 ^ exec),
                    "SAndn1SaveexecB64" => ~s0 & exec,
                    _ => ~s0 | exec,
                };
                SetS64(wave, dst, exec);
                wave.ExecMask = result;
                wave.Scc = result != 0;
                return;
            }
            case "SAndSaveexecB32": case "SOrSaveexecB32": case "SXorSaveexecB32": case "SAndn2SaveexecB32": case "SOrn2SaveexecB32":
            case "SNandSaveexecB32": case "SNorSaveexecB32": case "SXnorSaveexecB32": case "SAndn1SaveexecB32": case "SOrn1SaveexecB32":
            {
                var s0 = A();
                var exec = (uint)wave.ExecMask;
                var result = op switch
                {
                    "SAndSaveexecB32" => s0 & exec,
                    "SOrSaveexecB32" => s0 | exec,
                    "SXorSaveexecB32" => s0 ^ exec,
                    "SAndn2SaveexecB32" => s0 & ~exec,
                    "SOrn2SaveexecB32" => s0 | ~exec,
                    "SNandSaveexecB32" => ~(s0 & exec),
                    "SNorSaveexecB32" => ~(s0 | exec),
                    "SXnorSaveexecB32" => ~(s0 ^ exec),
                    "SAndn1SaveexecB32" => ~s0 & exec,
                    _ => ~s0 | exec,
                };
                SetS(wave, dst, exec);
                wave.ExecMask = (wave.ExecMask & 0xFFFFFFFF00000000UL) | result;
                wave.Scc = result != 0;
                return;
            }
            case "SWqmB32": { var v = A(); uint r = 0; for (var i = 0; i < 32; i += 4) if (((v >> i) & 0xF) != 0) r |= 0xFu << i; R32(r); return; }
            case "SWqmB64": { var v = A64(); ulong r = 0; for (var i = 0; i < 64; i += 4) if (((v >> i) & 0xF) != 0) r |= 0xFUL << i; SetS64(wave, dst, r); wave.Scc = r != 0; return; }
            case "SCmpEqU32": wave.Scc = A() == B(); return;
            case "SCmpLgU32": wave.Scc = A() != B(); return;
            case "SCmpGtU32": wave.Scc = A() > B(); return;
            case "SCmpGeU32": wave.Scc = A() >= B(); return;
            case "SCmpLtU32": wave.Scc = A() < B(); return;
            case "SCmpLeU32": wave.Scc = A() <= B(); return;
            case "SCmpEqI32": wave.Scc = (int)A() == (int)B(); return;
            case "SCmpLgI32": wave.Scc = (int)A() != (int)B(); return;
            case "SCmpGtI32": wave.Scc = (int)A() > (int)B(); return;
            case "SCmpGeI32": wave.Scc = (int)A() >= (int)B(); return;
            case "SCmpLtI32": wave.Scc = (int)A() < (int)B(); return;
            case "SCmpLeI32": wave.Scc = (int)A() <= (int)B(); return;
            case "SCmpEqU64": wave.Scc = A64() == B64(); return;
            case "SCmpLgU64": wave.Scc = A64() != B64(); return;
            case "SGetpcB64": SetS64(wave, dst, _program.Address + instruction.Pc + 4); return;
            default: throw new NotSupportedException(op);
        }
    }

    // ---------------- scalar memory ----------------
    private void ExecuteScalarMemory(Wave wave, Gen5ShaderInstruction instruction)
    {
        if (instruction.Control is not Gen5ScalarMemoryControl control) throw new NotSupportedException(instruction.Opcode);
        var baseRegister = instruction.Sources[0].Value;
        var offset = (long)control.ImmediateOffsetBytes + (control.DynamicOffsetRegister is { } dyn ? wave.S[dyn] : 0);
        var count = (int)control.DestinationCount;
        var dst = instruction.Destinations[0].Value;
        if (instruction.Opcode.StartsWith("SBufferLoad", StringComparison.Ordinal))
        {
            var w0 = wave.S[baseRegister]; var w1 = wave.S[baseRegister + 1]; var w2 = wave.S[baseRegister + 2];
            var address = (w0 | ((ulong)(w1 & 0xFFFF) << 32)) & ~3UL;
            for (var i = 0; i < count; i++)
            {
                var byteOffset = (ulong)(offset & ~3L) + (ulong)(i * 4);
                var loaded = byteOffset + 4 <= w2 ? ReadU32(address + byteOffset) : 0u;
                if (ScalarLoads.Count < 4096) ScalarLoads.Add((instruction.Pc, address + byteOffset, loaded));
                SetS(wave, dst + (uint)i, loaded);
            }

            return;
        }

        if (instruction.Opcode.StartsWith("SLoad", StringComparison.Ordinal))
        {
            var address = (wave.S[baseRegister] | ((ulong)(wave.S[baseRegister + 1] & 0xFFFF) << 32)) + (ulong)offset;
            address &= ~3UL;
            for (var i = 0; i < count; i++)
            {
                var loaded = ReadU32(address + (ulong)(i * 4));
                if (ScalarLoads.Count < 4096) ScalarLoads.Add((instruction.Pc, address + (ulong)(i * 4), loaded));
                SetS(wave, dst + (uint)i, loaded);
            }
            return;
        }

        throw new NotSupportedException(instruction.Opcode);
    }

    // ---------------- LDS ----------------
    private uint LdsRead(uint address) => address + 4 <= (uint)_lds.Length ? BinaryPrimitives.ReadUInt32LittleEndian(_lds.AsSpan((int)address)) : 0u;
    private void LdsWrite(uint address, uint value) { if (address + 4 <= (uint)_lds.Length) BinaryPrimitives.WriteUInt32LittleEndian(_lds.AsSpan((int)address), value); }

    private void ExecuteDataShare(Wave wave, Gen5ShaderInstruction instruction)
    {
        if (instruction.Control is not Gen5DataShareControl control || control.Gds) throw new NotSupportedException(instruction.Opcode);
        var op = instruction.Opcode;
        var exec = wave.ExecMask;
        for (var lane = 0; lane < _waveSize; lane++)
        {
            if ((exec >> lane & 1) == 0) continue;
            var address = Read(wave, instruction.Sources[0], lane);
            switch (op)
            {
                case "DsWriteB32": LdsWrite(address + control.SingleOffsetBytes, Read(wave, instruction.Sources[1], lane)); break;
                case "DsWriteB64":
                    LdsWrite(address + control.SingleOffsetBytes, Read(wave, instruction.Sources[1], lane));
                    LdsWrite(address + control.SingleOffsetBytes + 4, Read(wave, instruction.Sources[2], lane)); break;
                case "DsWriteB128":
                    for (var i = 0; i < 4; i++) LdsWrite(address + control.SingleOffsetBytes + (uint)i * 4, Read(wave, instruction.Sources[1 + i], lane)); break;
                case "DsWrite2B32":
                    LdsWrite(address + control.Offset0 * 4, Read(wave, instruction.Sources[1], lane));
                    LdsWrite(address + control.Offset1 * 4, Read(wave, instruction.Sources[2], lane)); break;
                case "DsWrite2St64B32":
                    LdsWrite(address + control.Offset0 * 256, Read(wave, instruction.Sources[1], lane));
                    LdsWrite(address + control.Offset1 * 256, Read(wave, instruction.Sources[2], lane)); break;
                case "DsReadB32": SetV(wave, instruction.Destinations[0].Value, lane, LdsRead(address + control.SingleOffsetBytes)); break;
                case "DsReadB64":
                    for (var i = 0; i < 2; i++) SetV(wave, instruction.Destinations[i].Value, lane, LdsRead(address + control.SingleOffsetBytes + (uint)i * 4)); break;
                case "DsReadB128":
                    for (var i = 0; i < 4; i++) SetV(wave, instruction.Destinations[i].Value, lane, LdsRead(address + control.SingleOffsetBytes + (uint)i * 4)); break;
                case "DsRead2B32":
                    SetV(wave, instruction.Destinations[0].Value, lane, LdsRead(address + control.Offset0 * 4));
                    SetV(wave, instruction.Destinations[1].Value, lane, LdsRead(address + control.Offset1 * 4)); break;
                case "DsRead2St64B32":
                    SetV(wave, instruction.Destinations[0].Value, lane, LdsRead(address + control.Offset0 * 256));
                    SetV(wave, instruction.Destinations[1].Value, lane, LdsRead(address + control.Offset1 * 256)); break;
                case "DsMinF32": case "DsMaxF32":
                {
                    var a = address + control.SingleOffsetBytes;
                    var mem = F(LdsRead(a));
                    var data = F(Read(wave, instruction.Sources[1], lane));
                    var take = op == "DsMinF32" ? data < mem : data > mem;
                    if (take || float.IsNaN(mem)) LdsWrite(a, U(data));
                    break;
                }
                case "DsAddU32": { var a = address + control.SingleOffsetBytes; LdsWrite(a, LdsRead(a) + Read(wave, instruction.Sources[1], lane)); break; }
                case "DsAddRtnU32": { var a = address + control.SingleOffsetBytes; var old = LdsRead(a); LdsWrite(a, old + Read(wave, instruction.Sources[1], lane)); SetV(wave, instruction.Destinations[0].Value, lane, old); break; }
                case "DsMinU32": { var a = address + control.SingleOffsetBytes; LdsWrite(a, Math.Min(LdsRead(a), Read(wave, instruction.Sources[1], lane))); break; }
                case "DsMaxU32": { var a = address + control.SingleOffsetBytes; LdsWrite(a, Math.Max(LdsRead(a), Read(wave, instruction.Sources[1], lane))); break; }
                case "DsOrB32": { var a = address + control.SingleOffsetBytes; LdsWrite(a, LdsRead(a) | Read(wave, instruction.Sources[1], lane)); break; }
                case "DsAndB32": { var a = address + control.SingleOffsetBytes; LdsWrite(a, LdsRead(a) & Read(wave, instruction.Sources[1], lane)); break; }
                default: throw new NotSupportedException(op);
            }
        }
    }

    // ---------------- buffers ----------------
    private readonly record struct Resource(ulong Base, uint Stride, uint Records, uint Word3)
    {
        public uint Format => (Word3 >> 12) & 0x7F;
        public bool AddTid => ((Word3 >> 23) & 1) != 0;
        public uint DstSel(int component) => (Word3 >> (component * 3)) & 7;
    }

    private Resource GetResource(Wave wave, uint register) =>
        new(wave.S[register] | ((ulong)(wave.S[register + 1] & 0xFFFF) << 32), (wave.S[register + 1] >> 16) & 0x3FFF, wave.S[register + 2], wave.S[register + 3]);

    private void ExecuteBuffer(Wave wave, Gen5ShaderInstruction instruction)
    {
        if (instruction.Control is not Gen5BufferMemoryControl control) throw new NotSupportedException(instruction.Opcode);
        var op = instruction.Opcode;
        var resource = GetResource(wave, control.ScalarResource);
        var soffset = ReadScalarOperand(wave, instruction.Sources[2]);
        var exec = wave.ExecMask;
        var store = op.StartsWith("BufferStore", StringComparison.Ordinal) || op.StartsWith("TBufferStore", StringComparison.Ordinal);
        for (var lane = 0; lane < _waveSize; lane++)
        {
            if ((exec >> lane & 1) == 0) continue;
            uint index = 0, offset = 0;
            var vaddr = control.VectorAddress;
            if (control.IndexEnabled) { index = GetV(wave, vaddr, lane); vaddr++; }
            if (control.OffsetEnabled) offset = GetV(wave, vaddr, lane);
            if (resource.AddTid) index += (uint)lane;
            var byteOffset = (ulong)offset + (ulong)control.OffsetBytes + soffset;
            bool InRange(ulong size) => resource.Stride != 0 ? index < resource.Records && (!control.IndexEnabled || true) : byteOffset + size <= resource.Records;
            var address = resource.Base + (ulong)index * resource.Stride + byteOffset;
            if (op.StartsWith("BufferStoreFormat", StringComparison.Ordinal))
            {
                if (InRange(4)) StoreFormatted(wave, resource, address, control.VectorData, (int)control.ComponentCount, lane, instruction.Pc);
                continue;
            }

            if (op.StartsWith("BufferLoadFormat", StringComparison.Ordinal))
            {
                var components = (int)control.ComponentCount;
                LoadFormatted(wave, resource, address, InRange(4), control.VectorData, components, lane);
                continue;
            }

            switch (op)
            {
                case "BufferLoadDword": case "BufferLoadDwordx2": case "BufferLoadDwordx3": case "BufferLoadDwordx4":
                    for (var i = 0; i < control.DwordCount; i++)
                        SetV(wave, control.VectorData + (uint)i, lane, InRange((ulong)(i + 1) * 4) ? ReadU32(address + (ulong)i * 4) : 0u);
                    break;
                case "BufferLoadUshort": case "BufferLoadSshort": case "BufferLoadShortD16": case "BufferLoadShortD16Hi":
                case "BufferLoadUbyte": case "BufferLoadSbyte":
                {
                    var bytes = op.Contains("byte", StringComparison.Ordinal) ? 1 : 2;
                    uint raw = 0;
                    if (InRange((ulong)bytes)) { Span<byte> b = stackalloc byte[2]; ReadBytes(address, b[..bytes]); raw = bytes == 1 ? b[0] : BinaryPrimitives.ReadUInt16LittleEndian(b); }
                    var prev = GetV(wave, control.VectorData, lane);
                    var value = op switch
                    {
                        "BufferLoadSshort" => (uint)(int)(short)raw,
                        "BufferLoadSbyte" => (uint)(int)(sbyte)raw,
                        "BufferLoadShortD16" => (prev & 0xFFFF0000) | (raw & 0xFFFF),
                        "BufferLoadShortD16Hi" => (prev & 0xFFFF) | (raw << 16),
                        _ => raw,
                    };
                    SetV(wave, control.VectorData, lane, value);
                    break;
                }
                case "BufferStoreDword": case "BufferStoreDwordx2": case "BufferStoreDwordx3": case "BufferStoreDwordx4":
                    for (var i = 0; i < control.DwordCount; i++)
                        if (InRange((ulong)(i + 1) * 4)) WriteU32(address + (ulong)i * 4, GetV(wave, control.VectorData + (uint)i, lane), instruction.Pc);
                    break;
                case "BufferStoreShort": case "BufferStoreShortD16Hi": case "BufferStoreByte":
                {
                    if (!InRange(2)) break;
                    var v = GetV(wave, control.VectorData, lane);
                    if (op == "BufferStoreShortD16Hi") v >>= 16;
                    Span<byte> b = stackalloc byte[2];
                    BinaryPrimitives.WriteUInt16LittleEndian(b, (ushort)v);
                    WriteBytes(address, op == "BufferStoreByte" ? b[..1] : b, instruction.Pc);
                    break;
                }
                case "BufferAtomicSwap": case "BufferAtomicCmpswap": case "BufferAtomicAdd": case "BufferAtomicSub":
                case "BufferAtomicSmin": case "BufferAtomicUmin": case "BufferAtomicSmax": case "BufferAtomicUmax":
                case "BufferAtomicAnd": case "BufferAtomicOr": case "BufferAtomicXor": case "BufferAtomicInc": case "BufferAtomicDec":
                case "BufferAtomicFmin": case "BufferAtomicFmax":
                {
                    if (!InRange(4)) { if (control.Glc) SetV(wave, control.VectorData, lane, 0); break; }
                    var old = ReadU32(address);
                    var data = GetV(wave, control.VectorData, lane);
                    var value = op switch
                    {
                        "BufferAtomicSwap" => data,
                        "BufferAtomicCmpswap" => old == GetV(wave, control.VectorData + 1, lane) ? data : old,
                        "BufferAtomicAdd" => old + data,
                        "BufferAtomicSub" => old - data,
                        "BufferAtomicSmin" => (uint)Math.Min((int)old, (int)data),
                        "BufferAtomicUmin" => Math.Min(old, data),
                        "BufferAtomicSmax" => (uint)Math.Max((int)old, (int)data),
                        "BufferAtomicUmax" => Math.Max(old, data),
                        "BufferAtomicAnd" => old & data,
                        "BufferAtomicOr" => old | data,
                        "BufferAtomicXor" => old ^ data,
                        "BufferAtomicInc" => old >= data ? 0 : old + 1,
                        "BufferAtomicDec" => old == 0 || old > data ? data : old - 1,
                        "BufferAtomicFmin" => U(MathF.Min(F(old), F(data))),
                        _ => U(MathF.Max(F(old), F(data))),
                    };
                    WriteU32(address, value, instruction.Pc);
                    if (control.Glc) SetV(wave, control.VectorData, lane, old);
                    break;
                }
                default:
                    if (store) throw new NotSupportedException(op);
                    throw new NotSupportedException(op);
            }
        }
    }

    private void LoadFormatted(Wave wave, Resource resource, ulong address, bool inRange, uint vdata, int components, int lane)
    {
        Gfx10UnifiedFormat.TryDecode(resource.Format, out var dataFormat, out var numberFormat);
        var values = new uint[4];
        var defaults = new uint[] { 0, 0, 0, numberFormat is 4 or 5 ? 1u : U(1f) };
        for (var c = 0; c < 4; c++)
        {
            if (!inRange || c >= Gfx10UnifiedFormat.ComponentCount(dataFormat) ||
                !Gfx10UnifiedFormat.TryGetComponentLayout(dataFormat, (uint)c, out var byteOffset, out var bitOffset, out var bitCount))
            {
                values[c] = inRange ? defaults[c] : 0u;
                continue;
            }

            Span<byte> b = stackalloc byte[8];
            ReadBytes(address + byteOffset, b[..(int)Math.Min(8, (bitOffset + bitCount + 7) / 8)]);
            var raw64 = BinaryPrimitives.ReadUInt64LittleEndian(b);
            var raw = (uint)((raw64 >> (int)bitOffset) & (bitCount >= 32 ? 0xFFFFFFFFUL : (1UL << (int)bitCount) - 1));
            values[c] = numberFormat switch
            {
                7 when bitCount == 32 => raw,
                7 when bitCount == 16 => U((float)BitConverter.UInt16BitsToHalf((ushort)raw)),
                0 => U(raw / (float)((1UL << (int)bitCount) - 1)),
                1 => U(Math.Max(-1f, ((int)(raw << (32 - (int)bitCount)) >> (32 - (int)bitCount)) / (float)((1UL << ((int)bitCount - 1)) - 1))),
                2 => U(raw),
                3 => U((int)(raw << (32 - (int)bitCount)) >> (32 - (int)bitCount)),
                4 => raw,
                5 => (uint)((int)(raw << (32 - (int)bitCount)) >> (32 - (int)bitCount)),
                _ => throw new NotSupportedException($"buffer number format {numberFormat} bits {bitCount}"),
            };
        }

        for (var c = 0; c < components; c++)
        {
            var sel = resource.DstSel(c);
            var value = sel switch { 0 => 0u, 1 => numberFormat is 4 or 5 ? 1u : U(1f), 4 => values[0], 5 => values[1], 6 => values[2], 7 => values[3], _ => 0u };
            SetV(wave, vdata + (uint)c, lane, value);
        }
    }

    private void StoreFormatted(Wave wave, Resource resource, ulong address, uint vdata, int components, int lane, ulong pc)
    {
        Gfx10UnifiedFormat.TryDecode(resource.Format, out var dataFormat, out var numberFormat);
        for (var c = 0; c < components && c < Gfx10UnifiedFormat.ComponentCount(dataFormat); c++)
        {
            if (!Gfx10UnifiedFormat.TryGetComponentLayout(dataFormat, (uint)c, out var byteOffset, out var bitOffset, out var bitCount)) continue;
            var value = GetV(wave, vdata + (uint)c, lane);
            var max = bitCount >= 32 ? uint.MaxValue : (1u << (int)bitCount) - 1;
            uint encoded = numberFormat switch
            {
                7 when bitCount == 32 => value,
                7 when bitCount == 16 => BitConverter.HalfToUInt16Bits((Half)F(value)),
                0 => (uint)MathF.Round(Math.Clamp(float.IsNaN(F(value)) ? 0 : F(value), 0f, 1f) * max),
                1 => (uint)(int)MathF.Round(Math.Clamp(float.IsNaN(F(value)) ? 0 : F(value), -1f, 1f) * (max >> 1)) & max,
                4 or 5 => value & max,
                _ => throw new NotSupportedException($"buffer store number format {numberFormat} bits {bitCount}"),
            };
            var byteCount = (int)((bitOffset + bitCount + 7) / 8);
            Span<byte> b = stackalloc byte[8];
            ReadBytes(address + byteOffset, b[..byteCount]);
            var raw = BinaryPrimitives.ReadUInt64LittleEndian(b);
            var fieldMask = (ulong)max << (int)bitOffset;
            raw = (raw & ~fieldMask) | (((ulong)encoded << (int)bitOffset) & fieldMask);
            BinaryPrimitives.WriteUInt64LittleEndian(b, raw);
            WriteBytes(address + byteOffset, b[..byteCount], pc);
        }
    }

    // ---------------- global ----------------
    private void ExecuteGlobal(Wave wave, Gen5ShaderInstruction instruction)
    {
        if (instruction.Control is not Gen5GlobalMemoryControl control || control.UsesFlatAddress) throw new NotSupportedException(instruction.Opcode);
        var exec = wave.ExecMask;
        var useScalar = control.ScalarAddress != 125 && control.ScalarAddress < 106;
        for (var lane = 0; lane < _waveSize; lane++)
        {
            if ((exec >> lane & 1) == 0) continue;
            ulong address = useScalar
                ? (wave.S[control.ScalarAddress] | ((ulong)wave.S[control.ScalarAddress + 1] << 32)) + GetV(wave, control.VectorAddress, lane)
                : GetV(wave, control.VectorAddress, lane) | ((ulong)GetV(wave, control.VectorAddress + 1, lane) << 32);
            address = (ulong)((long)address + control.OffsetBytes);
            if (instruction.Opcode.StartsWith("GlobalLoadDword", StringComparison.Ordinal))
            {
                for (var i = 0; i < control.DwordCount; i++)
                    SetV(wave, control.DestinationVectorRegister + (uint)i, lane, ReadU32(address + (ulong)i * 4));
            }
            else if (instruction.Opcode.StartsWith("GlobalStoreDword", StringComparison.Ordinal))
            {
                for (var i = 0; i < control.DwordCount; i++)
                    WriteU32(address + (ulong)i * 4, GetV(wave, control.SourceVectorRegister + (uint)i, lane), instruction.Pc);
            }
            else throw new NotSupportedException(instruction.Opcode);
        }
    }

    // ---------------- vector ALU ----------------
    private static readonly float InvTwoPi = 0.15915494f;

    private uint SourceFloatModifiers(Gen5ShaderInstruction instruction, int index, uint value)
    {
        uint abs = 0, neg = 0;
        switch (instruction.Control)
        {
            case Gen5Vop3Control v: abs = v.AbsoluteMask; neg = v.NegateMask; break;
            case Gen5SdwaControl s: abs = s.AbsoluteMask; neg = s.NegateMask; break;
            case Gen5DppControl d: abs = d.AbsoluteMask; neg = d.NegateMask; break;
        }

        if ((abs >> index & 1) != 0) value &= 0x7FFF_FFFF;
        if ((neg >> index & 1) != 0) value ^= 0x8000_0000;
        return value;
    }

    private static uint SdwaSelect(uint value, uint select, bool signExtend) => select switch
    {
        0 => signExtend ? (uint)(int)(sbyte)(value & 0xFF) : value & 0xFF,
        1 => signExtend ? (uint)(int)(sbyte)((value >> 8) & 0xFF) : (value >> 8) & 0xFF,
        2 => signExtend ? (uint)(int)(sbyte)((value >> 16) & 0xFF) : (value >> 16) & 0xFF,
        3 => signExtend ? (uint)(int)(sbyte)(value >> 24) : value >> 24,
        4 => signExtend ? (uint)(int)(short)(value & 0xFFFF) : value & 0xFFFF,
        5 => signExtend ? (uint)(int)(short)(value >> 16) : value >> 16,
        _ => value,
    };

    // The value a lane reads for source index, after DPP lane remapping (src0) and SDWA selection.
    private uint Src(Wave wave, Gen5ShaderInstruction instruction, int index, int lane, out bool dppValid)
    {
        dppValid = true;
        var operand = instruction.Sources[index];
        uint value;
        if (index == 0 && instruction.Control is Gen5DppControl dpp)
        {
            var source = DppSourceLane(dpp.Control, lane, out var inRange);
            var active = (wave.ExecMask >> source & 1) != 0;
            dppValid = inRange && (dpp.FetchInactive || active);
            value = dppValid ? Read(wave, operand, source) : 0u;
        }
        else if (index == 0 && instruction.Control is Gen5Dpp8Control dpp8)
        {
            var source = (lane & ~7) + (int)((dpp8.LaneSelectors >> ((lane & 7) * 3)) & 7);
            var active = (wave.ExecMask >> source & 1) != 0;
            value = dpp8.FetchInactive || active ? Read(wave, operand, source) : 0u;
        }
        else
        {
            value = Read(wave, operand, lane);
        }

        if (instruction.Control is Gen5SdwaControl sdwa && index < 2)
        {
            value = SdwaSelect(value, index == 0 ? sdwa.Source0Select : sdwa.Source1Select, index == 0 ? sdwa.Source0SignExtend : sdwa.Source1SignExtend);
        }

        return value;
    }

    private static int DppSourceLane(uint control, int lane, out bool inRange)
    {
        inRange = true;
        var rowBase = lane & ~15;
        var rowLane = lane & 15;
        if (control <= 0xFF) return (lane & ~3) + (int)((control >> ((lane & 3) * 2)) & 3);
        if (control is >= 0x101 and <= 0x10F) { var s = rowLane + (int)(control & 15); inRange = s < 16; return rowBase + (s & 15); }
        if (control is >= 0x111 and <= 0x11F) { var s = rowLane - (int)(control & 15); inRange = s >= 0; return rowBase + (s & 15); }
        if (control is >= 0x121 and <= 0x12F) return rowBase + ((rowLane - (int)(control & 15)) & 15);
        if (control == 0x140) return rowBase + 15 - rowLane;
        if (control == 0x141) return (lane & ~7) + 7 - (lane & 7);
        if (control is >= 0x150 and <= 0x15F) return rowBase + (int)(control & 15);
        if (control is >= 0x160 and <= 0x16F) return rowBase + (rowLane ^ (int)(control & 15));
        throw new NotSupportedException($"dpp control 0x{control:X}");
    }

    private bool DppWriteEnabled(Gen5DppControl dpp, int lane, bool valid) =>
        ((dpp.RowMask >> (lane >> 4)) & 1) != 0 && ((dpp.BankMask >> ((lane >> 2) & 3)) & 1) != 0 && (valid || dpp.BoundControl);

    private uint FinishFloat(Gen5ShaderInstruction instruction, float value)
    {
        uint omod = 0; var clamp = false;
        switch (instruction.Control)
        {
            case Gen5Vop3Control v: omod = v.OutputModifier; clamp = v.Clamp; break;
            case Gen5SdwaControl s: omod = s.OutputModifier; clamp = s.Clamp; break;
        }

        value = omod switch { 1 => value * 2, 2 => value * 4, 3 => value * 0.5f, _ => value };
        if (clamp) value = float.IsNaN(value) ? 0 : Math.Clamp(value, 0f, 1f);
        return U(value);
    }

    private static float Med3(float a, float b, float c)
    {
        if (float.IsNaN(a) || float.IsNaN(b) || float.IsNaN(c)) return MathF.Min(MathF.Min(a, b), c);
        return MathF.Max(MathF.Min(a, b), MathF.Min(MathF.Max(a, b), c));
    }

    private static float Min(float a, float b) => float.IsNaN(a) ? b : float.IsNaN(b) ? a : MathF.Min(a, b);
    private static float Max(float a, float b) => float.IsNaN(a) ? b : float.IsNaN(b) ? a : MathF.Max(a, b);

    private static uint CvtU32(float f) => float.IsNaN(f) || f <= 0 ? 0u : f >= 4294967295f ? uint.MaxValue : (uint)f;
    private static uint CvtI32(float f) => float.IsNaN(f) ? 0u : f >= 2147483647f ? int.MaxValue : f <= -2147483648f ? unchecked((uint)int.MinValue) : (uint)(int)f;

    private static bool ClassTest(uint raw, uint mask)
    {
        var exponent = (raw >> 23) & 0xFF;
        var mantissa = raw & 0x7FFFFF;
        var negative = (raw >> 31) != 0;
        int index;
        if (exponent == 0xFF && mantissa != 0) index = (raw & 0x400000) != 0 ? 1 : 0;
        else if (exponent == 0xFF) index = negative ? 2 : 9;
        else if (exponent == 0 && mantissa == 0) index = negative ? 5 : 6;
        else if (exponent == 0) index = negative ? 4 : 7;
        else index = negative ? 3 : 8;
        return ((mask >> index) & 1) != 0;
    }

    private void ExecuteVector(Wave wave, Gen5ShaderInstruction instruction)
    {
        var op = instruction.Opcode;
        var exec = wave.ExecMask;

        if (op == "VReadfirstlaneB32")
        {
            var lane = exec == 0 ? 0 : System.Numerics.BitOperations.TrailingZeroCount(exec);
            SetS(wave, instruction.Destinations[0].Value, Read(wave, instruction.Sources[0], lane));
            return;
        }

        if (op == "VReadlaneB32")
        {
            var lane = (int)(ReadScalarOperand(wave, instruction.Sources[1]) & (uint)(_waveSize - 1));
            SetS(wave, instruction.Destinations[0].Value, Read(wave, instruction.Sources[0], lane));
            return;
        }

        if (op == "VWritelaneB32")
        {
            var lane = (int)(ReadScalarOperand(wave, instruction.Sources[1]) & (uint)(_waveSize - 1));
            SetV(wave, instruction.Destinations[0].Value, lane, ReadScalarOperand(wave, instruction.Sources[0]));
            return;
        }

        if (op.StartsWith("VCmp", StringComparison.Ordinal))
        {
            ExecuteCompare(wave, instruction);
            return;
        }

        if (op is "VPermlanex16B32" or "VPermlane16B32")
        {
            var control = (Gen5Vop3Control)instruction.Control!;
            var fi = (control.OperandSelect & 1) != 0;
            var bc = (control.OperandSelect & 2) != 0;
            var results = new uint[_waveSize];
            var write = new bool[_waveSize];
            var low = ReadScalarOperand(wave, instruction.Sources[1]);
            var high = ReadScalarOperand(wave, instruction.Sources[2]);
            for (var lane = 0; lane < _waveSize; lane++)
            {
                if ((exec >> lane & 1) == 0) continue;
                var local = lane & 15;
                var select = local < 8 ? (low >> (local * 4)) & 15 : (high >> ((local - 8) * 4)) & 15;
                var rowBase = lane & ~15;
                if (op == "VPermlanex16B32") rowBase ^= 16;
                var source = rowBase + (int)select;
                var active = (exec >> source & 1) != 0;
                if (fi || active) { results[lane] = Read(wave, instruction.Sources[0], source); write[lane] = true; }
                else if (bc) { results[lane] = 0; write[lane] = true; }
            }

            for (var lane = 0; lane < _waveSize; lane++)
                if (write[lane]) SetV(wave, instruction.Destinations[0].Value, lane, results[lane]);
            return;
        }

        // Every lane reads its sources before any lane writes (DPP reads other lanes).
        var dst = instruction.Destinations.Count > 0 ? instruction.Destinations[0] : default;
        var computed = new uint[_waveSize];
        var computedHigh = new uint[_waveSize];
        var enabled = new bool[_waveSize];
        ulong carryOut = 0;
        for (var lane = 0; lane < _waveSize; lane++)
        {
            if ((exec >> lane & 1) == 0) continue;
            var valid = true;
            var a = instruction.Sources.Count > 0 ? Src(wave, instruction, 0, lane, out valid) : 0u;
            var write = true;
            if (instruction.Control is Gen5DppControl dpp) write = DppWriteEnabled(dpp, lane, instruction.Sources.Count > 0 && valid);
            var b = instruction.Sources.Count > 1 ? Src(wave, instruction, 1, lane, out _) : 0u;
            var c = instruction.Sources.Count > 2 ? Src(wave, instruction, 2, lane, out _) : 0u;
            float fa = F(SourceFloatModifiers(instruction, 0, a)), fb = F(SourceFloatModifiers(instruction, 1, b)), fc = F(SourceFloatModifiers(instruction, 2, c));
            uint r;
            uint high = 0;
            switch (op)
            {
                case "VNop": continue;
                case "VMovB32": r = a; break;
                case "VCndmaskB32":
                {
                    var mask = instruction.Encoding == Gen5ShaderEncoding.Vop3 && instruction.Sources.Count > 2 ? Read64(wave, instruction.Sources[2], 0) : wave.VccMask;
                    a = SourceFloatModifiers(instruction, 0, a);
                    b = SourceFloatModifiers(instruction, 1, b);
                    r = ((mask >> lane) & 1) != 0 ? b : a;
                    break;
                }
                case "VAddF32": r = FinishFloat(instruction, fa + fb); break;
                case "VSubF32": r = FinishFloat(instruction, fa - fb); break;
                case "VSubrevF32": r = FinishFloat(instruction, fb - fa); break;
                case "VMulF32": r = FinishFloat(instruction, fa * fb); break;
                case "VMinF32": r = FinishFloat(instruction, Min(fa, fb)); break;
                case "VMaxF32": r = FinishFloat(instruction, Max(fa, fb)); break;
                case "VMadF32": r = FinishFloat(instruction, (float)(fa * fb) + fc); break;
                case "VFmaF32": r = FinishFloat(instruction, MathF.FusedMultiplyAdd(fa, fb, fc)); break;
                case "VMacF32": r = FinishFloat(instruction, (float)(fa * fb) + F(GetV(wave, dst.Value, lane))); break;
                case "VFmacF32": r = FinishFloat(instruction, MathF.FusedMultiplyAdd(fa, fb, F(GetV(wave, dst.Value, lane)))); break;
                case "VMadMkF32": r = FinishFloat(instruction, (float)(fa * fb) + fc); break;
                case "VMadAkF32": r = FinishFloat(instruction, (float)(fa * fb) + fc); break;
                case "VMed3F32": r = FinishFloat(instruction, Med3(fa, fb, fc)); break;
                case "VMin3F32": r = FinishFloat(instruction, Min(Min(fa, fb), fc)); break;
                case "VMax3F32": r = FinishFloat(instruction, Max(Max(fa, fb), fc)); break;
                case "VSqrtF32": r = FinishFloat(instruction, MathF.Sqrt(fa)); break;
                case "VRsqF32": r = FinishFloat(instruction, 1f / MathF.Sqrt(fa)); break;
                case "VRcpF32": case "VRcpIflagF32": r = FinishFloat(instruction, 1f / fa); break;
                case "VExpF32": r = FinishFloat(instruction, MathF.Pow(2f, fa)); break;
                case "VLogF32": r = FinishFloat(instruction, MathF.Log2(fa)); break;
                case "VSinF32": r = FinishFloat(instruction, MathF.Sin(fa * 2f * MathF.PI)); break;
                case "VCosF32": r = FinishFloat(instruction, MathF.Cos(fa * 2f * MathF.PI)); break;
                case "VFractF32": r = FinishFloat(instruction, float.IsInfinity(fa) ? float.NaN : Math.Min(fa - MathF.Floor(fa), 0.99999994f)); break;
                case "VFloorF32": r = FinishFloat(instruction, MathF.Floor(fa)); break;
                case "VLdexpF32": r = FinishFloat(instruction, MathF.ScaleB(fa, (int)b)); break;
                case "VCvtFlrI32F32": r = (uint)(int)Math.Clamp(MathF.Floor(fa), int.MinValue, int.MaxValue); break;
                case "VCvtRpiI32F32": r = (uint)(int)Math.Clamp(MathF.Floor(fa + 0.5f), int.MinValue, int.MaxValue); break;
                case "VCvtF16F32": r = (uint)BitConverter.HalfToUInt16Bits((Half)fa); break;
                case "VCvtF32F16": r = FinishFloat(instruction, (float)BitConverter.UInt16BitsToHalf((ushort)a)); break;
                case "VCvtPkrtzF16F32": case "VCvtPkrtzF16F32E64": r = (uint)BitConverter.HalfToUInt16Bits((Half)fa) | ((uint)BitConverter.HalfToUInt16Bits((Half)fb) << 16); break;
                case "VCvtF32Ubyte0": r = U((float)(a & 0xFF)); break;
                case "VCvtF32Ubyte1": r = U((float)((a >> 8) & 0xFF)); break;
                case "VCvtF32Ubyte2": r = U((float)((a >> 16) & 0xFF)); break;
                case "VCvtF32Ubyte3": r = U((float)(a >> 24)); break;
                case "VCeilF32": r = FinishFloat(instruction, MathF.Ceiling(fa)); break;
                case "VTruncF32": r = FinishFloat(instruction, MathF.Truncate(fa)); break;
                case "VRndneF32": r = FinishFloat(instruction, MathF.Round(fa, MidpointRounding.ToEven)); break;
                case "VCvtF32U32": r = FinishFloat(instruction, (float)a); break;
                case "VCvtF32I32": r = FinishFloat(instruction, (float)(int)a); break;
                case "VCvtU32F32": r = CvtU32(fa); break;
                case "VCvtI32F32": r = CvtI32(fa); break;
                case "VFmaMixF32":
                {
                    var control = (Gen5Vop3pControl)instruction.Control!;
                    float Mix(int i, uint raw)
                    {
                        var operand = instruction.Sources[i];
                        var half = ((control.OpSelHiMask >> i) & 1) != 0 && operand.Kind is Gen5OperandKind.VectorRegister or Gen5OperandKind.ScalarRegister or Gen5OperandKind.LiteralConstant;
                        var value = half ? (float)BitConverter.UInt16BitsToHalf((ushort)(((control.OpSelMask >> i) & 1) != 0 ? raw >> 16 : raw)) : F(raw);
                        if (((control.NegHiMask >> i) & 1) != 0) value = MathF.Abs(value);
                        if (((control.NegLoMask >> i) & 1) != 0) value = -value;
                        return value;
                    }

                    var mixed = MathF.FusedMultiplyAdd(Mix(0, a), Mix(1, b), Mix(2, c));
                    if (control.Clamp) mixed = Math.Clamp(mixed, 0f, 1f);
                    r = U(mixed);
                    break;
                }
                case "VAddI32": case "VAddU32": case "VAddNcU32": r = a + b; break;
                case "VSubI32": case "VSubU32": case "VSubNcU32": r = a - b; break;
                case "VSubrevI32": case "VSubrevU32": case "VSubrevNcU32": r = b - a; break;
                case "VAddCoU32":
                {
                    var s = (ulong)a + b; r = (uint)s; if (s > uint.MaxValue) carryOut |= 1UL << lane; break;
                }
                case "VAddcU32": case "VAddCoCiU32":
                {
                    var carryMask = instruction.Sources.Count > 2 ? Read64(wave, instruction.Sources[2], 0) : wave.VccMask;
                    var s = (ulong)a + b + ((carryMask >> lane) & 1); r = (uint)s; if (s > uint.MaxValue) carryOut |= 1UL << lane; break;
                }
                case "VSubCoU32": { r = a - b; if (b > a) carryOut |= 1UL << lane; break; }
                case "VSubbU32": case "VSubCoCiU32":
                {
                    var carryMask = instruction.Sources.Count > 2 ? Read64(wave, instruction.Sources[2], 0) : wave.VccMask;
                    var borrow = (carryMask >> lane) & 1;
                    r = a - b - (uint)borrow; if ((ulong)b + borrow > a) carryOut |= 1UL << lane; break;
                }
                case "VMulLoU32": case "VMulLoI32": r = a * b; break;
                case "VMulHiU32": r = (uint)(((ulong)a * b) >> 32); break;
                case "VMulHiI32": r = (uint)(((long)(int)a * (int)b) >> 32); break;
                case "VMulU32U24": r = (a & 0xFFFFFF) * (b & 0xFFFFFF); break;
                case "VMulI32I24": r = (uint)(((int)(a << 8) >> 8) * ((int)(b << 8) >> 8)); break;
                case "VMadU32U24": r = (a & 0xFFFFFF) * (b & 0xFFFFFF) + c; break;
                case "VAdd3U32": r = a + b + c; break;
                case "VLshlAddU32": r = (a << (int)(b & 31)) + c; break;
                case "VAddLshlU32": r = (a + b) << (int)(c & 31); break;
                case "VLshlOrB32": case "VLshlOrU32": r = (a << (int)(b & 31)) | c; break;
                case "VXor3B32": r = a ^ b ^ c; break;
                case "VXadU32": r = (a ^ b) + c; break;
                case "VAlignbitB32": r = (uint)((((ulong)a << 32) | b) >> (int)(c & 31)); break;
                case "VAlignbyteB32": r = (uint)((((ulong)a << 32) | b) >> (int)((c & 3) * 8)); break;
                case "VMin3I32": r = (uint)Math.Min(Math.Min((int)a, (int)b), (int)c); break;
                case "VMax3I32": r = (uint)Math.Max(Math.Max((int)a, (int)b), (int)c); break;
                case "VMin3U32": r = Math.Min(Math.Min(a, b), c); break;
                case "VMax3U32": r = Math.Max(Math.Max(a, b), c); break;
                case "VMed3I32": r = (uint)Math.Max(Math.Min((int)a, (int)b), Math.Min(Math.Max((int)a, (int)b), (int)c)); break;
                case "VMed3U32": r = Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c)); break;
                case "VSadU32": r = (uint)Math.Abs((long)a - b) + c; break;
                case "VBfmB32": r = ((1u << (int)(a & 31)) - 1) << (int)(b & 31); break;
                case "VBfrevB32": r = ReverseBits(a); break;
                case "VFfbhU32": r = a == 0 ? uint.MaxValue : (uint)System.Numerics.BitOperations.LeadingZeroCount(a); break;
                case "VFfblB32": r = a == 0 ? uint.MaxValue : (uint)System.Numerics.BitOperations.TrailingZeroCount(a); break;
                case "VPermB32":
                {
                    var bytes = ((ulong)a << 32) | b;
                    r = 0;
                    for (var i = 0; i < 4; i++)
                    {
                        var sel = (c >> (i * 8)) & 0xFF;
                        uint v = sel < 8 ? (uint)((bytes >> (int)(sel * 8)) & 0xFF) : sel == 12 ? 0u : sel > 12 ? 0xFFu : (uint)(((bytes >> (int)(((sel - 8) * 2 + 1) * 8 + 7)) & 1) * 0xFF);
                        r |= v << (i * 8);
                    }
                    break;
                }
                case "VAndOrB32": r = (a & b) | c; break;
                case "VOr3B32": r = a | b | c; break;
                case "VAndB32": r = a & b; break;
                case "VOrB32": r = a | b; break;
                case "VXorB32": r = a ^ b; break;
                case "VNotB32": r = ~a; break;
                case "VLshlrevB32": r = b << (int)(a & 31); break;
                case "VLshrrevB32": r = b >> (int)(a & 31); break;
                case "VAshrrevI32": r = (uint)((int)b >> (int)(a & 31)); break;
                case "VLshlB32": r = a << (int)(b & 31); break;
                case "VLshrB32": r = a >> (int)(b & 31); break;
                case "VBfeU32": { var off = (int)(b & 31); var w = (int)(c & 31); r = w == 0 ? 0 : (a >> off) & ((1u << w) - 1); break; }
                case "VBfeI32": { var off = (int)(b & 31); var w = (int)(c & 31); r = w == 0 ? 0 : (uint)((int)(a << (32 - off - w)) >> (32 - w)); break; }
                case "VBfiB32": r = (a & b) | (~a & c); break;
                case "VBcntU32B32": r = (uint)System.Numerics.BitOperations.PopCount(a) + b; break;
                case "VMinI32": r = (uint)Math.Min((int)a, (int)b); break;
                case "VMaxI32": r = (uint)Math.Max((int)a, (int)b); break;
                case "VMinU32": r = Math.Min(a, b); break;
                case "VMaxU32": r = Math.Max(a, b); break;
                case "VMbcntLoU32B32": r = (uint)System.Numerics.BitOperations.PopCount(a & (uint)((1UL << Math.Min(lane, 32)) - 1)) + b; break;
                case "VMbcntHiU32B32": r = (uint)System.Numerics.BitOperations.PopCount(lane <= 32 ? 0u : a & (uint)((1UL << (lane - 32)) - 1)) + b; break;
                case "VLshlrevB64":
                {
                    var value = Read64(wave, instruction.Sources[1], lane) << (int)(a & 63);
                    r = (uint)value; high = (uint)(value >> 32); break;
                }
                case "VLshrrevB64":
                {
                    var value = Read64(wave, instruction.Sources[1], lane) >> (int)(a & 63);
                    r = (uint)value; high = (uint)(value >> 32); break;
                }
                default: throw new NotSupportedException(op);
            }

            if (!write) continue;
            computed[lane] = r;
            computedHigh[lane] = high;
            enabled[lane] = true;
        }

        var sdwaDst = instruction.Control as Gen5SdwaControl;
        for (var lane = 0; lane < _waveSize; lane++)
        {
            if (!enabled[lane] || dst.Kind != Gen5OperandKind.VectorRegister) continue;
            var value = computed[lane];
            if (sdwaDst is not null && sdwaDst.DestinationSelect != 6)
            {
                var (shift, width) = sdwaDst.DestinationSelect switch { 0 => (0, 8), 1 => (8, 8), 2 => (16, 8), 3 => (24, 8), 4 => (0, 16), _ => (16, 16) };
                var mask = ((1u << width) - 1) << shift;
                var positioned = (value << shift) & mask;
                var previous = GetV(wave, dst.Value, lane);
                value = sdwaDst.DestinationUnused switch
                {
                    0 => positioned,
                    1 => positioned | ((positioned >> (shift + width - 1) & 1) != 0 ? ~((1u << (shift + width)) - 1) : 0u),
                    _ => (previous & ~mask) | positioned,
                };
            }

            SetV(wave, dst.Value, lane, value);
            if (op is "VLshlrevB64" or "VLshrrevB64") SetV(wave, dst.Value + 1, lane, computedHigh[lane]);
        }

        if (op is "VAddCoU32" or "VAddcU32" or "VAddCoCiU32" or "VSubCoU32" or "VSubbU32" or "VSubCoCiU32")
        {
            var carryDst = instruction.Control switch
            {
                Gen5Vop3Control { ScalarDestination: { } s } => s,
                Gen5SdwaControl { ScalarDestination: { } s } => s,
                _ => (uint)Vcc,
            };
            SetMask(wave, carryDst, carryOut & exec);
        }
    }

    private void ExecuteCompare(Wave wave, Gen5ShaderInstruction instruction)
    {
        var op = instruction.Opcode;
        var cmpx = op.StartsWith("VCmpx", StringComparison.Ordinal);
        var name = op[(cmpx ? 5 : 4)..];
        var exec = wave.ExecMask;
        ulong mask = 0;
        for (var lane = 0; lane < _waveSize; lane++)
        {
            if ((exec >> lane & 1) == 0) continue;
            bool result;
            if (name.EndsWith("U64", StringComparison.Ordinal) || name.EndsWith("I64", StringComparison.Ordinal))
            {
                var a = Read64(wave, instruction.Sources[0], lane);
                var b = Read64(wave, instruction.Sources[1], lane);
                result = name[..^3] switch
                {
                    "Eq" => a == b, "Ne" => a != b, "Lg" => a != b,
                    "Lt" => name.EndsWith("I64") ? (long)a < (long)b : a < b,
                    "Gt" => name.EndsWith("I64") ? (long)a > (long)b : a > b,
                    "Le" => name.EndsWith("I64") ? (long)a <= (long)b : a <= b,
                    "Ge" => name.EndsWith("I64") ? (long)a >= (long)b : a >= b,
                    _ => throw new NotSupportedException(op),
                };
            }
            else
            {
                var a = Src(wave, instruction, 0, lane, out _);
                var b = Src(wave, instruction, 1, lane, out _);
                if (name == "ClassF32")
                {
                    result = ClassTest(SourceFloatModifiers(instruction, 0, a), b);
                }
                else if (name.EndsWith("F32", StringComparison.Ordinal))
                {
                    var fa = F(SourceFloatModifiers(instruction, 0, a));
                    var fb = F(SourceFloatModifiers(instruction, 1, b));
                    var unordered = float.IsNaN(fa) || float.IsNaN(fb);
                    result = name[..^3] switch
                    {
                        "F" => false, "Tru" => true,
                        "Lt" => fa < fb, "Eq" => fa == fb, "Le" => fa <= fb, "Gt" => fa > fb, "Lg" => !unordered && fa != fb, "Ge" => fa >= fb,
                        "O" => !unordered, "U" => unordered,
                        "Nge" => !(fa >= fb), "Nlg" => !(!unordered && fa != fb), "Ngt" => !(fa > fb), "Nle" => !(fa <= fb), "Neq" => !(fa == fb), "Nlt" => !(fa < fb),
                        _ => throw new NotSupportedException(op),
                    };
                }
                else if (name.EndsWith("16", StringComparison.Ordinal))
                {
                    var signed = name.EndsWith("I16", StringComparison.Ordinal);
                    long sa = signed ? (short)(a & 0xFFFF) : (a & 0xFFFF);
                    long sb = signed ? (short)(b & 0xFFFF) : (b & 0xFFFF);
                    result = IntegerCompare(name[..^3], sa, sb, op);
                }
                else
                {
                    var signed = name.EndsWith("I32", StringComparison.Ordinal);
                    long sa = signed ? (int)a : a;
                    long sb = signed ? (int)b : b;
                    result = IntegerCompare(name[..^3], sa, sb, op);
                }
            }

            if (result) mask |= 1UL << lane;
        }

        if (cmpx)
        {
            wave.ExecMask = _waveSize == 32 ? (wave.ExecMask & 0xFFFFFFFF00000000UL) | (uint)mask : mask;
            return;
        }

        var destination = instruction.Control switch
        {
            Gen5Vop3Control { ScalarDestination: { } s } => s,
            Gen5SdwaControl { ScalarDestination: { } s } => s,
            _ => instruction.Destinations.Count > 0 && instruction.Destinations[0].Kind == Gen5OperandKind.ScalarRegister ? instruction.Destinations[0].Value : (uint)Vcc,
        };
        SetMask(wave, destination, mask);
    }

    private static bool IntegerCompare(string name, long a, long b, string op) => name switch
    {
        "F" => false, "T" => true,
        "Lt" => a < b, "Eq" => a == b, "Le" => a <= b, "Gt" => a > b, "Ne" => a != b, "Lg" => a != b, "Ge" => a >= b,
        _ => throw new NotSupportedException(op),
    };
}
