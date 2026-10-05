// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.Pipelines;

namespace SharpEmu.Libs.Agc;

public static partial class AgcExports
{
    // This partial exposes the shader headers the create export registered.

    internal static ulong GetShaderHeaderAddress(ulong codeAddress) =>
        _shaderHeadersByCode.TryGetValue(codeAddress, out var header) ? header : 0;

    // AGC uploads a tessellation pipeline as consecutive descriptors: the hull front, then
    // its hull back. The front ends by jumping to the back, whose address the hardware
    // receives outside the registers, so the pair is taken from that layout. Registered at
    // draw time: when the front is created the back descriptor is not written yet.
    internal static bool TryRegisterAdjacentHullContinuation(CpuContext context, ulong frontCodeAddress)
    {
        var frontHeader = GetShaderHeaderAddress(frontCodeAddress);
        if (frontHeader == 0 || !TryReadByte(context, frontHeader + ShaderTypeOffset, out var frontType) || frontType != HsFrontShaderType)
        {
            return false;
        }

        const int ScanBytes = 0x400;
        var bytes = new byte[ScanBytes + ShaderStructBytes];
        if (!context.Memory.TryRead(frontHeader, bytes))
        {
            return false;
        }

        for (var offset = sizeof(uint); offset <= ScanBytes; offset += sizeof(uint))
        {
            var descriptor = bytes.AsSpan(offset, ShaderStructBytes);
            if (System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(descriptor) != ShaderFileHeader ||
                System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(descriptor[sizeof(uint)..]) != ShaderVersion)
            {
                continue;
            }

            // The next descriptor must be the back half; anything else means another layout.
            if (descriptor[(int)ShaderTypeOffset] != HsBackShaderType)
            {
                return false;
            }

            var backCode = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(descriptor[(int)ShaderCodeOffset..]);
            var backSize = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(descriptor[(int)ShaderSizeOffset..]);
            if (backCode == 0 || backCode == frontCodeAddress || !IsValidDeclaredShaderSize(backSize))
            {
                return false;
            }

            var backHeader = frontHeader + (ulong)offset;
            if (ShaderCompiler.Gen5ShaderTranslator.TryGetFusedProgramParts(context, frontCodeAddress, out var registered, out var registeredHeader) &&
                registered == backCode && registeredHeader == backHeader)
            {
                return true;
            }

            ShaderCompiler.Gen5ShaderTranslator.RegisterFusedProgram(context, frontCodeAddress, frontHeader, backCode, backHeader);
            return true;
        }

        return false;
    }

    internal static ShaderHeaderRegistry CreateShaderHeaderRegistry(CpuContext context) =>
        new(
            context,
            GetShaderHeaderAddress,
            codeAddress => ShaderCompiler.Gen5ShaderTranslator.TryGetFusedProgramParts(context, codeAddress, out var continuation, out var continuationHeader)
                ? new FusedProgramParts(continuation, continuationHeader)
                : null);
}
