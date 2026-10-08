// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

public sealed class HizPolicyTests
{
    private static uint[][] Images(ulong address)
    {
        var images = new uint[4][];
        for (var index = 0; index < images.Length; index++) images[index] = new uint[8];
        images[3][0] = (uint)(address >> 8);
        images[3][1] = (uint)(address >> 40) & 0xFF;
        return images;
    }

    [Fact]
    public void TheTwoBuilderImagesPairUpAndAReadFindsItsOther()
    {
        Assert.True(HizPolicy.TryGetBuilderImage(Images(0x504C270000UL), out var first));
        Assert.True(HizPolicy.TryGetBuilderImage(Images(0x504C51FC00UL), out var second));
        Assert.Equal(0x504C270000UL, first);
        Assert.Equal(0x504C51FC00UL, second);

        Assert.True(HizPolicy.TryGetPair(0x504C54E600UL, 8, out var image, out var other));
        Assert.Equal(0x504C51FC00UL, image);
        Assert.Equal(0x504C270000UL, other);
        Assert.False(HizPolicy.TryGetPair(0x5000000000UL, 8, out _, out _));
    }

    [Fact]
    public void ADescriptorWithoutAFourthImageIsNotABuilderImage()
    {
        Assert.False(HizPolicy.TryGetBuilderImage(new uint[2][], out _));
    }
}
