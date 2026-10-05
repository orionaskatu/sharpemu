// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using Silk.NET.Vulkan;

namespace SharpEmu.Libs.Gpu.Rendering;

// Tessellated draws (local + hull + domain stages). The merged local+hull program runs as
// compute workgroups that write the control points and factors to the guest rings, as the
// hardware does; the domain program then runs as a vertex shader over a uniform grid per patch.
public sealed partial class RenderExecutor
{
    private const uint TessellationStageMask = 0x0200210D;
    private const uint WaveSizeStageBits = (1u << 21) | (1u << 22) | (1u << 23);

    // The off-chip slot of one hull workgroup: the domain program addresses its patch
    // constants from the end of this slot.
    private const uint OffchipBytesPerGroup = 0x8000;

    // The off-chip ring holds this many workgroup slots; larger draws run in chunks.
    private const uint OffchipRingGroups = 256;

    private const uint QuadDomain = 2;
    private const uint TriangleClockwiseTopology = 2;

    private static readonly uint? TessellationSegmentsOverride =
        uint.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_TESSELLATION_SEGMENTS"), out var segments) ? segments : null;

    private static int _tessellationWarningShown;

    private static bool IsTessellationStageMask(uint stages) => (stages & ~WaveSizeStageBits) == TessellationStageMask;

    private void DrawTessellated(ulong submitId, RegisterBanks banks, uint vertexCount, uint instanceCount, uint firstVertex)
    {
        var shaderInterface = banks.Context.ShaderInterface;
        var configuration = shaderInterface.LocalHullConfiguration;
        var maxPatchesPerGroup = configuration & 0xFF;
        var inputControlPoints = (configuration >> 8) & 0x3F;
        var outputControlPoints = (configuration >> 14) & 0x3F;
        var factorParameter = shaderInterface.TessellationFactorParameter;
        var domainType = factorParameter & 0x3;
        var outputTopology = (factorParameter >> 5) & 0x7;
        if (inputControlPoints == 0 || outputControlPoints == 0 || outputControlPoints > 64 || inputControlPoints > 64 ||
            domainType != QuadDomain || vertexCount < inputControlPoints)
        {
            WarnTessellation($"configuration=0x{configuration:X8} factors=0x{factorParameter:X8} vertices={vertexCount}");
            return;
        }

        var patchesPerGroup = Math.Max(1, Math.Min(Math.Max(maxPatchesPerGroup, 1), 64 / Math.Max(inputControlPoints, outputControlPoints)));
        var patchesPerInstance = vertexCount / inputControlPoints;
        var totalPatches = (ulong)patchesPerInstance * instanceCount;
        if (totalPatches == 0)
        {
            return;
        }

        var maxLevel = BitConverter.UInt32BitsToSingle(shaderInterface.MaxTessellationLevel);
        var segments = TessellationSegmentsOverride ??
            (uint)Math.Clamp(float.IsFinite(maxLevel) ? MathF.Ceiling(maxLevel) : 1f, 1f, 16f);
        var hull = new Gen5HullDispatch(patchesPerGroup, inputControlPoints, outputControlPoints, OffchipBytesPerGroup, 6 * sizeof(float));
        var domain = new Gen5DomainGrid(patchesPerGroup, OffchipBytesPerGroup, segments, Triangles: false,
            Clockwise: outputTopology == TriangleClockwiseTopology);

        var vertex = banks.Shader.Vertex;
        var hullUserCount = Math.Max(vertex.HullResource2.UserScalarCount, vertex.HullUserScalars.Count);
        var chunkPatches = (ulong)OffchipRingGroups * patchesPerGroup;
        for (var firstPatch = 0ul; firstPatch < totalPatches; firstPatch += chunkPatches)
        {
            var patches = (uint)Math.Min(chunkPatches, totalPatches - firstPatch);
            var userData = new uint[8 + hullUserCount];
            userData[0] = (uint)vertex.HullUserDataAddress;
            userData[1] = (uint)(vertex.HullUserDataAddress >> 32);
            userData[2] = (uint)firstPatch;
            userData[3] = patches;
            userData[4] = patchesPerInstance;
            userData[5] = firstVertex;
            Array.Copy(vertex.HullUserScalars.Values, 0, userData, 8, hullUserCount);
            if (!DispatchHull(vertex, hull, userData, (patches + patchesPerGroup - 1) / patchesPerGroup))
            {
                return;
            }

            DrawDomainGrid(submitId, banks, domain, patches);
        }
    }

    private bool DispatchHull(VertexStageRegisters vertex, Gen5HullDispatch hull, uint[] userData, uint groups)
    {
        var hullProgram = _pipelines.GetHullProgram(vertex, hull, userData);
        if (hullProgram is not { Available: true })
        {
            WarnTessellation(hullProgram is null
                ? $"the hull program has no registered continuation: local=0x{vertex.LocalAddress:X16}"
                : $"the hull program could not be built: local=0x{vertex.LocalAddress:X16}");
            return false;
        }

        var input = hullProgram.Input;
        var program = input.Stage.Program ?? throw _host.Fatal($"The hull program is missing: local=0x{vertex.LocalAddress:X16}.");
        _host.EndRendering();
        using (_host.BeginPreparation())
        {
            var pipeline = _pipelines.CreateComputePipeline(input, hullProgram.Program);
            var bindings = _host.PrepareBindings(input.Stage);
            if (program.UsesDeviceAddresses)
            {
                _host.PrepareDeviceAddresses();
            }

            _host.BindResources(bindings);
            Span<IPreparedBindings> stages = [bindings];
            _host.CommitBindings(PipelineBindPoint.Compute, in pipeline, stages);
            _host.ShaderWriteHazardBarrier();
            _host.BindPipeline(PipelineBindPoint.Compute, in pipeline);
            _host.Dispatch(groups, 1, 1);
            _host.ShaderAccessBarrier();
        }

        _host.ResetBindings();
        return true;
    }

    private void DrawDomainGrid(ulong submitId, RegisterBanks banks, Gen5DomainGrid domain, uint patches)
    {
        var draw = new DrawCall("DrawTessellated", RecordedOperation.DrawIndexAuto, 6 * domain.Segments * domain.Segments, patches, 0);
        var state = DrawState.Create();
        if (!TryResolveDrawTargets(banks, in draw, ref state))
        {
            _host.ResetBindings();
            return;
        }

        var context = banks.Context;
        Span<ColorComponentMapArray> mappingStorage = stackalloc ColorComponentMapArray[1];
        Span<ColorComponentMap> targetExportMapping = mappingStorage[0];
        targetExportMapping.Fill(ColorComponentMap.Identity);
        foreach (ref readonly var color in BoundColors(ref state))
        {
            targetExportMapping[(int)color.Slot] = color.Resolution.ExportMapping;
        }

        var programs = _pipelines.GetDomainPrograms(
            banks.Shader.Vertex, banks.Shader.Pixel, context.ShaderInterface, context, targetExportMapping,
            state.PixelActive, state.Depth.HasTarget, domain);
        if (programs is not { Available: true })
        {
            WarnTessellation($"the domain program could not be built: export=0x{banks.Shader.Vertex.ExportAddress:X16}");
            _host.ResetBindings();
            return;
        }

        state.Programs = programs;
        TraceDrawState(submitId, banks, in draw, in state);
        var emission = new DrawEmission(false, 0, 0, 0);
        // The grid is a triangle list; the guest primitive type names the patches.
        var primitiveType = banks.UserConfig.PrimitiveType;
        banks.UserConfig.PrimitiveType = (uint)GuestPrimitiveType.TriangleList;
        try
        {
            RecordDraw(submitId, banks, in draw, ref state, PrimitiveTopology.TriangleList, in emission, default,
                primitiveRestart: false, setBindDebug: false, setAutoDebug: true);
        }
        finally
        {
            banks.UserConfig.PrimitiveType = primitiveType;
        }

        _host.ResetBindings();
    }

    private static void WarnTessellation(string detail)
    {
        if (Interlocked.Exchange(ref _tessellationWarningShown, 1) == 0)
        {
            Console.Error.WriteLine($"[GPU][WARN] A tessellated draw was skipped: {detail}.");
        }
    }
}
