// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.GpuMemory;

// One bit per tracked block: set while the block has at least one CPU-dirty page.
// Regions update their bit under their own lock whenever their CPU-dirty mask
// changes, so a clear bit lets a reader skip the block without taking any lock.
// A reader that must observe a CPU write is ordered after it by the guest's own
// submission, which happens after the write fault published the bit.
public sealed class CpuDirtySummary
{
    private const int WordBits = 64;
    private readonly long[] _words = new long[(TrackerLayout.BlockCount + WordBits - 1) / WordBits];
    private long _epoch;

    public long Epoch => Volatile.Read(ref _epoch);

    public void NoteDirtied() => Interlocked.Increment(ref _epoch);

    public void Set(ulong block, bool dirty)
    {
        var bit = 1L << (int)(block % WordBits);
        ref var word = ref _words[block / WordBits];
        if (dirty)
        {
            Interlocked.Or(ref word, bit);
        }
        else
        {
            Interlocked.And(ref word, ~bit);
        }
    }

    public void ForEachDirty(Action<ulong> visit)
    {
        for (var index = 0; index < _words.Length; index++)
        {
            var word = (ulong)Volatile.Read(ref _words[index]);
            while (word != 0)
            {
                var bit = System.Numerics.BitOperations.TrailingZeroCount(word);
                visit((ulong)index * WordBits + (ulong)bit);
                word &= word - 1;
            }
        }
    }

    // A second bit per block, set when the block gains CPU-dirty pages or pages whose CPU
    // bytes can change without a fault. Device-address preparation takes and clears these
    // bits, so it only revisits blocks that changed since its previous sweep.
    private readonly long[] _pending = new long[(TrackerLayout.BlockCount + WordBits - 1) / WordBits];

    public void MarkPending(ulong block) =>
        Interlocked.Or(ref _pending[block / WordBits], 1L << (int)(block % WordBits));

    // Clears each pending word before visiting it, so a block marked during the visit is kept.
    public void TakePending(Action<ulong> visit)
    {
        for (var index = 0; index < _pending.Length; index++)
        {
            if (Volatile.Read(ref _pending[index]) == 0)
            {
                continue;
            }

            var word = (ulong)Interlocked.Exchange(ref _pending[index], 0);
            while (word != 0)
            {
                var bit = System.Numerics.BitOperations.TrailingZeroCount(word);
                visit((ulong)index * WordBits + (ulong)bit);
                word &= word - 1;
            }
        }
    }

    public bool IsDirty(ulong block) => (Volatile.Read(ref _words[block / WordBits]) & (1L << (int)(block % WordBits))) != 0;

    public ulong Word(int index) => (ulong)Volatile.Read(ref _words[index]);
}
