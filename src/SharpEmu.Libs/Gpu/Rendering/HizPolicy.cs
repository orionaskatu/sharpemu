// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Rendering;

// Ghost of Yotei culls its static world on the CPU against a depth pyramid (R32Sfloat, 960x540, linear depth in
// meters, 1e7 = far) that a compute pass (9DCF7A37...) rebuilds every frame into one of two alternating images.
// The guest reads the image back after each build; an image built from a frame whose depth lacks the world makes
// the next frame's test hide the world, and the frame after that it comes back, so the world flickered every other
// frame. The policy keeps the test conservative:
//   union (default): every guest read of one pyramid page sees the farther of the two images' depths;
//   far:             the build is skipped and both images read as empty (nothing is occluded);
//   real:            the unmodified behaviour.
// SHARPEMU_HIZ picks the mode; SHARPEMU_HIZ_SCHED="from:len:mode;..." (seconds of process time) overrides it for tests.
internal static class HizPolicy
{
    internal enum Mode { Real, Union, Far }

    internal const ulong BuilderProgramHash = 0x9DCF7A3711A288C0UL;
    internal const ulong ReprojectProgramHash = 0xF00717DE7B897C69UL;

    // The pyramid images are rewritten only where the frame's depth reaches: the rest keeps its old contents, zero
    // (an occluder at distance 0 that culls everything behind it) in the image that was never fully written. The GPU
    // consumers (terrain culling, tile passes) then dropped most of the terrain on the frames that read that image.
    // The images are filled with "far" ahead of the pass that rewrites them. SHARPEMU_HIZ_PREFILL=0 disables.
    internal static readonly bool PrefillFar = Environment.GetEnvironmentVariable("SHARPEMU_HIZ_PREFILL") != "0";

    internal static bool ShouldPrefill(ulong programHash, int imageIndex) =>
        PrefillFar && ((programHash == BuilderProgramHash && imageIndex == BuilderPyramidImageIndex) ||
                       (programHash == ReprojectProgramHash && imageIndex == 0));

    internal const ulong ImageBytes = 960UL * 540UL * 4UL;
    internal const uint FarBits = 0x4B189680; // 1e7
    private const int BuilderPyramidImageIndex = 3;

    private static readonly Mode DefaultMode = Parse(Environment.GetEnvironmentVariable("SHARPEMU_HIZ")) ?? Mode.Union;
    private static readonly (double From, double Length, Mode Mode)[] Schedule =
        (Environment.GetEnvironmentVariable("SHARPEMU_HIZ_SCHED") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Select(item => item.Split(':'))
            .Where(parts => parts.Length == 3 && Parse(parts[2]) is not null)
            .Select(parts => (double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture), double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture), Parse(parts[2])!.Value))
            .ToArray();
    private static readonly long Start = System.Diagnostics.Stopwatch.GetTimestamp();
    private static readonly object Gate = new();
    private static readonly ulong[] Bases = new ulong[2];

    private static Mode? Parse(string? text) => text?.ToLowerInvariant() switch
    {
        "real" or "off" or "0" => Mode.Real,
        "union" => Mode.Union,
        "far" or "1" => Mode.Far,
        _ => null,
    };

    internal static Mode Current
    {
        get
        {
            if (Schedule.Length == 0) return DefaultMode;
            var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(Start).TotalSeconds;
            foreach (var (from, length, mode) in Schedule)
                if (elapsed >= from && elapsed < from + length) return mode;
            return DefaultMode;
        }
    }

    // The pyramid image a builder dispatch writes (its fourth image); the two alternating bases are learned here.
    internal static bool TryGetBuilderImage(IReadOnlyList<uint[]> images, out ulong address)
    {
        address = 0;
        if (images.Count <= BuilderPyramidImageIndex || images[BuilderPyramidImageIndex].Length < 8) return false;
        address = new SharpEmu.Libs.Gpu.Images.TextureDescriptorWords(images[BuilderPyramidImageIndex]).BaseAddress;
        if (address == 0) return false;
        lock (Gate)
        {
            if (Bases[0] == address || Bases[1] == address) return true;
            if (Bases[0] == 0) Bases[0] = address;
            else if (Bases[1] == 0) Bases[1] = address;
        }

        return true;
    }

    // For a guest read at the address: the pyramid image it falls in and the other one, once both are known.
    internal static bool TryGetPair(ulong address, ulong size, out ulong image, out ulong other)
    {
        image = other = 0;
        lock (Gate)
        {
            if (Bases[0] == 0 || Bases[1] == 0) return false;
            for (var index = 0; index < 2; index++)
            {
                if (address + size > Bases[index] && address < Bases[index] + ImageBytes)
                {
                    image = Bases[index];
                    other = Bases[1 - index];
                    return true;
                }
            }
        }

        return false;
    }
}
