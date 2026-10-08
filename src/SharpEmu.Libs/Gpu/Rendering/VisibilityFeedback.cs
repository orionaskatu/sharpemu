// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Rendering;

// Ghost of Yotei culls its static world with a visibility feedback loop: compute 17444E6A... turns the
// previous frame's visibility images into a per-triangle bitmask that the compaction passes read. The
// first image (R32Uint, object << 17 | triangle) is never rendered into at the address the pass reads;
// the world pass writes the same format into its slot-1 target (the object/triangle id target), which
// still holds the previous frame when the pass runs. Reading that target instead feeds the loop.
internal static class VisibilityFeedback
{
    internal const ulong CullProgramHash = 0x17444E6ABBF4F82CUL;
    // SHARPEMU_VIS_KEEP_IMAGES=0 applies the pending DCC clear to the culling pass's inputs again.
    internal static readonly bool KeepVisibilityImages = Environment.GetEnvironmentVariable("SHARPEMU_VIS_KEEP_IMAGES") != "0";
    // Off by default: the id target is not the triangle-id image (feeding it left the static world mostly missing); SHARPEMU_VIS_FEEDBACK=1 enables it.
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPEMU_VIS_FEEDBACK") == "1";

    // CB_COLOR_INFO of the R32Uint id target (format bits; the DCC flag bit is ignored).
    private const uint IdTargetInfo = 0x50410;
    private const uint IdTargetInfoMask = 0xFFFFF;

    internal static ulong IdTarget;
    internal static uint[]? IdTargetWords;

    // Called for every draw with its slot-1 base and info.
    internal static void NoteSlotOne(ulong baseAddress, uint info, bool written)
    {
        if (written && baseAddress != 0 && (info & IdTargetInfoMask) == IdTargetInfo && baseAddress != IdTarget)
        {
            IdTarget = baseAddress;
            IdTargetWords = null;
        }
    }
}
