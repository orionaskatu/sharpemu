// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

// Reads one dword of guest memory; false when the address cannot be read.
public delegate bool GuestWordReader(ulong address, out uint word);
// A contiguous clean range; false makes the caller use its individual word reads.
public delegate bool GuestWordsReader(ulong address, Span<uint> words);

// What one draw supplies to materialise a plan: its user data, the shader base and
// the two memory readers. The clean reader refuses memory the GPU may still own.
public sealed record ResourceRuntimeInputs
{
    public IReadOnlyList<uint> UserData { get; init; } = [];
    public ulong ShaderBase { get; init; }
    public GuestWordReader? ReadMemory { get; init; }
    public GuestWordReader? ReadCleanMemory { get; init; }
    public GuestWordsReader? ReadCleanWords { get; init; }

    // Whether a guest address is mapped, without reading or synchronizing it. Null makes
    // the callers probe with ReadMemory instead.
    public Func<ulong, bool>? IsMapped { get; init; }
    public ComputeSelectorState? ComputeState { get; init; }

    // Told true before the flattened table's words are evaluated and false after, so a reader
    // wrapper can tell the words only the table reads from those the descriptors depend on.
    public Action<bool>? TablePhase { get; init; }

    // Whether the GPU still owns a guest dword: earlier recorded work writes it and the CPU
    // copy is stale. A flattened table word read straight from such a dword is left for the
    // host to copy on the device, in order, instead of draining the queue to read it here.
    public Func<ulong, bool>? IsGpuPendingWord { get; init; }

    // Set only when the result is not cached: a large descriptor table probed within the last
    // few milliseconds may then be reused instead of read again.
    public bool AllowTransientTableReuse { get; init; }

    public ResidentGuestBytesReader? ReadResidentMemory { get; init; }

    public bool ReadsClean { get; init; }

    public ResourceRuntimeInputs WithReader(GuestWordReader? reader) => this with
    {
        ReadMemory = reader,
        ReadsClean = ReadsClean || ReferenceEquals(reader, ReadCleanMemory),
    };
}

public readonly record struct ComputeSelectorState(uint WaveSize, uint ThreadsX, uint ThreadsY, uint ThreadsZ,
    bool HasPartialWorkgroups, uint LocalDataShareDwords, int LocalInvocationIdComponents);
