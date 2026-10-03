// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class BufferTableImageTests
{
    private const ulong TableAddress = 0x1000;
    private const uint RecordBytes = 32;

    private static readonly IndirectImageSelector Selector = new(0, 0, 0, 0, 0) { Dense = true, BufferTableStride = RecordBytes };

    // Two direct records: the second names a texture at a different base address.
    private static (TestWordMemory Memory, uint[] First, uint[] Second) Table()
    {
        var memory = new TestWordMemory { Base = TableAddress, Words = new uint[0x40 / 4] };
        var first = ResourceTrackerTests.ImageDescriptor();
        var second = first.ToArray();
        second[0] += 1;
        ResourceTrackerTests.WriteImage(memory, TableAddress, first);
        ResourceTrackerTests.WriteImage(memory, TableAddress + RecordBytes, second);
        return (memory, first, second);
    }

    private static DescriptorWords TableDescriptor() => new([(uint)TableAddress, 0, 2 * RecordBytes, 0]);

    private static ulong ImageBase(uint[] descriptor) => (ulong)descriptor[0] << 8;

    [Fact]
    public void ResidencyIsAMappingQueryThatNeverReadsTheTexture()
    {
        var (memory, first, second) = Table();
        var textureReads = 0;
        var inputs = new ResourceRuntimeInputs
        {
            ReadCleanMemory = memory.Read,
            ReadMemory = (ulong address, out uint word) =>
            {
                textureReads++;
                word = 0;
                return true;
            },
            IsMapped = address => address == ImageBase(first),
        };

        Assert.True(ResourceMaterializer.MaterializeBufferTableImage(Selector, TableDescriptor(), false, inputs, out var table, out _));

        Assert.Equal(0, textureReads);
        Assert.Equal(new uint[] { 0, RecordBytes }, table.Keys);
        Assert.Equal(first, table.Descriptors[(int)table.Candidates[0]].Dwords);
        // The unmapped texture binds as null.
        Assert.Equal(new uint[8], table.Descriptors[(int)table.Candidates[1]].Dwords);
        Assert.NotEqual(second, table.Descriptors[(int)table.Candidates[1]].Dwords);
    }

    [Fact]
    public void WithoutAMappingQueryResidencyStillProbesWithARead()
    {
        var (memory, first, _) = Table();
        var probed = new List<ulong>();
        var inputs = new ResourceRuntimeInputs
        {
            ReadCleanMemory = memory.Read,
            ReadMemory = (ulong address, out uint word) =>
            {
                probed.Add(address);
                word = 0;
                return address == ImageBase(first);
            },
        };

        Assert.True(ResourceMaterializer.MaterializeBufferTableImage(Selector, TableDescriptor(), false, inputs, out var table, out _));

        Assert.Equal(2, probed.Count);
        Assert.Equal(first, table.Descriptors[(int)table.Candidates[0]].Dwords);
        Assert.Equal(new uint[8], table.Descriptors[(int)table.Candidates[1]].Dwords);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void EachRecordIsOneBulkReadAndARefusedBulkReadFallsBackToWords(bool bulkAccepted)
    {
        var (memory, first, second) = Table();
        var bulkReads = 0;
        var inputs = new ResourceRuntimeInputs
        {
            ReadCleanMemory = memory.Read,
            ReadCleanWords = (address, words) =>
            {
                bulkReads++;
                if (!bulkAccepted)
                    return false;
                for (var index = 0; index < words.Length; index++)
                    memory.Read(address + (ulong)index * sizeof(uint), out words[index]);
                return true;
            },
            IsMapped = _ => true,
        };
        var wordReadsBefore = memory.Reads;

        Assert.True(ResourceMaterializer.MaterializeBufferTableImage(Selector, TableDescriptor(), false, inputs, out var table, out _));

        Assert.Equal(2, bulkReads);
        // Accepted bulk reads go through the test memory as well, one word at a time.
        Assert.Equal(16u, memory.Reads - wordReadsBefore);
        Assert.Equal(first, table.Descriptors[(int)table.Candidates[0]].Dwords);
        Assert.Equal(second, table.Descriptors[(int)table.Candidates[1]].Dwords);
    }
}
