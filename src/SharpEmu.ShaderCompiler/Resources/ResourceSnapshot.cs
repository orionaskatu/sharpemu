// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

// The descriptor words and user data captured for one program at draw time, plus the
// flattened resource table and the device-address ranges the planner could bound.
public sealed class ResourceSnapshot
{
    public uint[][] Buffers { get; init; } = [];
    public uint[][] Images { get; init; } = [];
    public uint[][] Samplers { get; init; } = [];
    public uint[] FlattenedResourceTable { get; init; } = [];

    // Table words the host copies from GPU-owned guest memory when it binds the table.
    public TableWordPatch[] TablePatches { get; init; } = [];
    public uint[] UserData { get; init; } = [];
    public DeviceAddressRange[] DeviceAddressRanges { get; init; } = [];
}

// A flattened table word that holds the guest dword at Address once the GPU has written it.
public readonly record struct TableWordPatch(uint FlatOffset, ulong Address);

// One device-address handle's evaluated base and bounded extent for this draw.
public readonly record struct DeviceAddressRange(uint Handle, ulong Base, ulong Size, bool Planned, bool Written);
