// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Libs.Gpu.Pipelines;

// How a guest compute wave maps onto host subgroups. The translator request and the pipeline's
// required subgroup size must agree, so both ask here.
public static class ComputeWaveModel
{
    private static readonly bool NativeWave64Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_NATIVE_WAVE64") != "0";

    // A wave64 program whose workgroup is whole waves runs each guest wave as one 64-lane host
    // subgroup when the device can require that size for compute.
    public static bool UsesNativeWave64(ComputeInputInfo info, bool deviceSupportsSubgroup64, uint maxWorkgroupSubgroups)
    {
        if (!NativeWave64Enabled || !deviceSupportsSubgroup64 || info.WaveSize != 64)
        {
            return false;
        }

        var invocations = (ulong)Math.Max(info.ThreadsX, 1) * Math.Max(info.ThreadsY, 1) * Math.Max(info.ThreadsZ, 1);
        return invocations % 64 == 0 && (maxWorkgroupSubgroups == 0 || invocations / 64 <= maxWorkgroupSubgroups);
    }
}
