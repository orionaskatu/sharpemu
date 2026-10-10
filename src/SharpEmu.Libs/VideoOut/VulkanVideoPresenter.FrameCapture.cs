// TEMPORARY DEBUG TOOL - do not commit.
// SHARPEMU_CAPTURE_FRAME_AT_SECONDS=<s> arms one frame after that many seconds; every color and
// depth attachment the frame renders to is read back at the next frame boundary and written to
// SHARPEMU_CAPTURE_FRAME_DIR as PNG (final contents of each target in that frame).

using System.Diagnostics;
using System.IO.Compression;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Vulkan;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Gpu.Buffers;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace SharpEmu.Libs.VideoOut;

internal static unsafe partial class VulkanVideoPresenter
{
    private sealed partial class Presenter
    {
        private sealed record CaptureTarget(CachedImage Image, ImageViewDescription View, ulong Address, bool Depth, Format Format, uint Width, uint Height)
        {
            public int FirstPass;
            public int LastPass;
        }

        private static readonly double _captureAtSeconds =
            double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_FRAME_AT_SECONDS"), System.Globalization.CultureInfo.InvariantCulture, out var seconds) ? seconds : -1;
        private static readonly string? _captureBaseDirectory = Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_FRAME_DIR");
        // Without SHARPEMU_CAPTURE_FRAME_AT_SECONDS, creating <dir>/trigger captures the next frame into <dir>/capNN.
        private static readonly bool _captureTriggerMode = _captureAtSeconds < 0;
        private string? _captureDirectory;
        private int _captureIndex;
        private static readonly long _captureStart = Stopwatch.GetTimestamp();
        private readonly Dictionary<ulong, CaptureTarget> _captureViews = new();
        private int _captureState; // 0 idle, 1 armed (recording passes), 2 pending readback, 3 done
        private int _capturePass;

        private bool CaptureEnabled => !string.IsNullOrEmpty(_captureBaseDirectory) && (_captureTriggerMode || _captureState < 3);

        private void CaptureNoteView(CachedImage image, in ImageViewDescription view, ImageView handle, ulong address, bool depth, uint width, uint height)
        {
            if (!CaptureEnabled || handle.Handle == 0)
                return;
            _captureViews[handle.Handle] = new CaptureTarget(image, view, address, depth, view.Format, width, height);
        }

        private readonly List<(int Pass, int Slot, CaptureTarget Target, VkBuffer Buffer, DeviceMemory Memory, ulong Size, uint TexelBytes)> _captureReadbacks = new();
        private readonly List<string> _captureManifest = new();
        private static readonly HashSet<ulong> _captureTextureHashes =
            (Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_TEXTURE_HASH") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(text => ulong.Parse(text.Trim().Replace("0x", ""), System.Globalization.NumberStyles.HexNumber)).ToHashSet();
        private static readonly ulong _captureTextureHash = _captureTextureHashes.Count > 0 ? 1ul : 0ul;
        private readonly Dictionary<ulong, (int Index, CachedImage Image, ImageViewDescription View, ulong Address, string Words)> _captureTextures = new();
        private readonly List<(string Name, VkBuffer Buffer, DeviceMemory Memory, ulong Size, Format Format, uint Width, uint Height, uint TexelBytes, string Info)> _captureTextureReadbacks = new();

        private void CaptureNoteTexture(ShaderProgramInfo program, int index, ResourceSlotIdentifier imageIdentifier, in ImageRequest request, ulong address, uint[] words)
        {
            if (_captureState != 1 || _captureTextureHash == 0 || (!_captureTextureHashes.Contains(0xA11) && !_captureTextureHashes.Contains(program.Hash)))
                return;
            var image = _imageCache.GetImage(imageIdentifier);
            var key = (ulong)image.Backing.Handle.Handle ^ ((ulong)request.View.BaseLevel << 56) ^ ((ulong)request.View.BaseLayer << 48);
            _captureTextures.TryAdd(key, (index, image, request.View, address, $"ps=0x{program.Hash:X16} {program.Stage} " + string.Join(",", words.Select(word => word.ToString("x8")))));
            if (program.Stage == ShaderStageKind.Compute) _capturePendingTextures?.Add((image, request.View, address, program.Hash, index));
        }

        private static uint BlockBytes(Format format)
        {
            var name = format.ToString();
            if (name.StartsWith("BC1", StringComparison.Ordinal) || name.StartsWith("BC4", StringComparison.Ordinal)) return 8;
            return name.StartsWith("BC", StringComparison.Ordinal) ? 16u : 0u;
        }

        private void CaptureQueueTextures()
        {
            var number = 0;
            ulong textureBytes = 0;
            foreach (var (index, image, view, address, words) in _captureTextures.Values)
            {
                var backing = image.Backing;
                var level = Math.Min(view.BaseLevel, Math.Max(backing.MipLevels, 1) - 1);
                var width = Math.Max(1u, backing.Extent.Width >> (int)level);
                var height = Math.Max(1u, backing.Extent.Height >> (int)level);
                var block = BlockBytes(backing.Format);
                var texel = block == 0 ? TexelBytes(backing.Format, false) : 0;
                var size = block != 0 ? (ulong)((width + 3) / 4) * ((height + 3) / 4) * block : (ulong)width * height * texel;
                var guest = new byte[65536];
                var guestRead = _guestMemory.TryRead(address, guest);
                var guestNonZero = guestRead ? guest.Count(value => value != 0) * 100.0 / guest.Length : -1;
                var info = FormattableString.Invariant(
                    $"index={index} addr=0x{address:X} backingFmt={backing.Format} viewFmt={view.Format} extent={backing.Extent.Width}x{backing.Extent.Height} mips={backing.MipLevels} layers={backing.Layers} type={backing.ImageType} level={level} mapped={IsGuestMapped(address)} guest64k_nonzero={guestNonZero:F1}% head={Convert.ToHexString(guest, 0, 16)} words={words}");
                var name = $"tex{number++:D3}_i{index}_{backing.Format}_{width}x{height}_0x{address:X}";
                var depthFormat = backing.Format.ToString().StartsWith("D", StringComparison.Ordinal) || backing.Format.ToString().StartsWith("S8", StringComparison.Ordinal);
                if (size == 0 || !backing.Exists || backing.Samples != 1 || backing.ImageType != ImageType.Type2D || (backing.Usage & ImageUsageFlags.TransferSrcBit) == 0 ||
                    depthFormat || view.BaseLayer >= backing.Layers || textureBytes + size > 1536ul * 1024 * 1024)
                {
                    _captureManifest.Add($"{name} skip {info}");
                    continue;
                }

                textureBytes += size;
                var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
                var command = BeginBatchedGuestCommands();
                var range = new SubresourceRange(level, 1, view.BaseLayer, 1);
                var previous = backing.State;
                image.Transition(ImageLayout.TransferSrcOptimal, AccessFlags.TransferReadBit, range, command);
                var region = new BufferImageCopy
                {
                    ImageSubresource = new ImageSubresourceLayers(ImageAspectFlags.ColorBit, level, view.BaseLayer, 1),
                    ImageExtent = new Extent3D(width, height, 1),
                };
                _vk.CmdCopyImageToBuffer(command, backing.Handle, ImageLayout.TransferSrcOptimal, buffer, 1, &region);
                image.Transition(ImageLayout.ShaderReadOnlyOptimal, AccessFlags.ShaderReadBit, range, command);
                _captureTextureReadbacks.Add((name, buffer, memory, size, backing.Format, width, height, texel, info));
            }
        }

        private void CaptureWriteTextures()
        {
            foreach (var (name, buffer, memory, size, format, width, height, texel, info) in _captureTextureReadbacks)
            {
                void* mapped;
                if (_vk.MapMemory(_device, memory, 0, size, 0, &mapped) != Result.Success || mapped == null)
                    continue;
                var bytes = new ReadOnlySpan<byte>(mapped, checked((int)size));
                var nonZero = 0L;
                foreach (var value in bytes)
                    if (value != 0) nonZero++;
                var stats = FormattableString.Invariant($"host_nonzero={nonZero * 100.0 / bytes.Length:F1}%");
                if (texel != 0)
                {
                    try { stats += " " + WriteCapturePng(Path.Combine(_captureDirectory!, name + ".png"), bytes, format, false, width, height, texel); }
                    catch (Exception exception) { stats += " png failed " + exception.Message; }
                }

                _captureManifest.Add($"{name} {stats} {info}");
                _vk.UnmapMemory(_device, memory);
                _vk.DestroyBuffer(_device, buffer, null);
                _deviceInfo.FreeMemory(memory);
            }
        }

        private ulong _captureBytes;

        private string _dbgDepthState = "";
        private static readonly bool _dbgEqualToLequal = Environment.GetEnvironmentVariable("SHARPEMU_DBG_EQUAL_TO_LEQUAL") == "1";

        private readonly List<(BufferBinding Arguments, string Info)> _captureIndirect = new();

        private VkBuffer _captureArgsBuffer;
        private DeviceMemory _captureArgsMemory;
        private const int CaptureArgsCapacity = 4096;

        public bool DebugCapturing
        {
            get
            {
                SharpEmu.HLE.GpuMemory.DbgWatch.Note = _captureState == 1 ? DebugNote : null;
                return _captureState == 1;
            }
        }

        private readonly List<(string Label, VkBuffer Buffer, DeviceMemory Memory, ulong Size)> _captureCopies = new();

        public void DebugCopyBuffer(ulong address, ulong size, string label)
        {
            if (_captureState != 1 || size == 0)
                return;
            EndRendering();
            var (source, sourceOffset) = _bufferCache.ObtainBuffer(address, size, false);
            var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
            var command = BeginBatchedGuestCommands();
            var barrier = new MemoryBarrier2 { SType = StructureType.MemoryBarrier2, SrcAccessMask = AccessFlags2.MemoryWriteBit, DstAccessMask = AccessFlags2.TransferReadBit };
            VulkanSynchronization.PipelineBarrier(_vk, command, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit, 0, 1, &barrier, 0, null, 0, null);
            var region = new BufferCopy(sourceOffset, 0, size);
            _vk.CmdCopyBuffer(command, source.Handle, buffer, 1, &region);
            _captureCopies.Add(($"{label}_0x{address:X}", buffer, memory, size));
        }

        private void CaptureWriteCopies()
        {
            foreach (var (label, buffer, memory, size) in _captureCopies)
            {
                void* mapped;
                if (_vk.MapMemory(_device, memory, 0, size, 0, &mapped) == Result.Success)
                {
                    File.WriteAllBytes(Path.Combine(_captureDirectory!, label + ".bin"), new ReadOnlySpan<byte>(mapped, (int)size).ToArray());
                    _vk.UnmapMemory(_device, memory);
                }

                _vk.DestroyBuffer(_device, buffer, null);
                _deviceInfo.FreeMemory(memory);
            }

            _captureCopies.Clear();
        }

        public string DebugBufferState(ulong address, ulong size) => // TEMP
            $"cpuDirty={_bufferCache.HasCpuDirtyPages(address, size)} gpuDirty={_bufferCache.HasGpuDirtyPages(address, size)} gpuBytes={_bufferCache.HasGpuDirtyBytes(address, size)} hot={_bufferCache.DbgIsCpuWriteHot(address, size)} stream=0x{_bufferCache.GetUtilityBuffer(Gpu.Buffers.GpuBufferUsage.Stream).Handle.Handle:X}";

        public void DebugNote(string line)
        {
            if (_captureState == 1)
                lock (_captureManifest)
                    _captureManifest.Add(line);
        }

        // Copies the arguments before the draw records its bindings; the draw begins the pass again.
        public void DebugCaptureIndirectArguments(BufferBinding arguments)
        {
            if (_captureState != 1 || _captureIndirect.Count >= CaptureArgsCapacity)
                return;
            if (_captureArgsBuffer.Handle == 0)
                _captureArgsBuffer = CreateBuffer(CaptureArgsCapacity * 20, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out _captureArgsMemory);
            EndRendering();
            var command = BeginBatchedGuestCommands();
            var barrier = new MemoryBarrier2
            {
                SType = StructureType.MemoryBarrier2,
                SrcAccessMask = AccessFlags2.MemoryWriteBit,
                DstAccessMask = AccessFlags2.TransferReadBit,
            };
            VulkanSynchronization.PipelineBarrier(_vk, command, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit, 0, 1, &barrier, 0, null, 0, null);
            var region = new BufferCopy(arguments.Offset, (ulong)_captureIndirect.Count * 20, 20);
            _vk.CmdCopyBuffer(command, new VkBuffer(arguments.Handle), _captureArgsBuffer, 1, &region);
            _captureIndirect.Add((arguments, ""));
        }

        private void CaptureNoteIndirect(BufferBinding arguments)
        {
            if (_captureState == 1 && _captureIndirect.Count > 0 && _captureIndirect[^1].Info.Length == 0)
                _captureIndirect[^1] = (arguments, $"pass={_capturePass} marker='{SharpEmu.Libs.Diagnostics.DbgSequence.Marker}' ps=0x{_boundGraphicsPipeline?.ProfilePixelHash ?? 0:X} vs=0x{_boundGraphicsPipeline?.ProfileVertexHash ?? 0:X}");
        }

        // TEMP: SHARPEMU_CAPTURE_BUFFERS_AT=csHash:addr:size:nth,... copies guest buffers right before the nth dispatch of that compute shader.
        private static readonly (ulong Hash, ulong Address, ulong Size, int Nth)[] CaptureBuffersAt =
            (Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_BUFFERS_AT") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(item => item.Split(':')).Where(parts => parts.Length >= 3)
            .Select(parts => (Convert.ToUInt64(parts[0].Replace("0x", ""), 16), Convert.ToUInt64(parts[1].Replace("0x", ""), 16), Convert.ToUInt64(parts[2].Replace("0x", ""), 16), parts.Length > 3 ? int.Parse(parts[3]) : 0)).ToArray();
        private (ulong Address, ulong Size)[] _dbgDispatchBuffers = [];
        public void DebugSetDispatchBuffers((ulong Address, ulong Size)[] buffers) => _dbgDispatchBuffers = buffers;
        private ulong _dbgDrawHash;
        public void DebugSetDrawHash(ulong vertexHash) => _dbgDrawHash = vertexHash;

        private void CaptureSnapshotBuffersAtDraw()
        {
            if (CaptureBuffersAt.Length == 0 || _captureState != 1)
                return;
            var saved = _dbgComputeHash;
            _dbgComputeHash = _dbgDrawHash;
            CaptureSnapshotBuffersAt();
            _dbgComputeHash = saved;
        }
        private readonly int[] _captureBuffersAtSeen = new int[CaptureBuffersAt.Length];
        private readonly List<(string Name, VkBuffer Buffer, DeviceMemory Memory, ulong Size, ulong Address)> _captureBufferSnaps = [];

        private void CaptureSnapshotBuffersAt()
        {
            if (CaptureBuffersAt.Length == 0 || _captureState != 1)
                return;
            for (var index = 0; index < CaptureBuffersAt.Length; index++)
            {
                var entry = CaptureBuffersAt[index];
                if (entry.Hash != _dbgComputeHash || _captureBuffersAtSeen[index]++ != entry.Nth)
                    continue;
                var address = entry.Address;
                var size = entry.Size;
                if (address < 256) // slot form: "hash:slot:0:nth" captures the slot's whole resolved range
                {
                    if ((int)address >= _dbgDispatchBuffers.Length || _dbgDispatchBuffers[(int)address].Size == 0)
                        continue;
                    var slot = (int)address;
                    address = _dbgDispatchBuffers[slot].Address;
                    size = size == 0 ? Math.Min(_dbgDispatchBuffers[slot].Size, 64ul << 20) : size;
                    _captureManifest.Add($"bufferat-slot hash=0x{entry.Hash:X} slot={slot} addr=0x{address:X} size=0x{size:X}");
                }

                var (source, sourceOffset) = _bufferCache.ObtainBuffer(address, size, false);
                var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
                var command = BeginBatchedGuestCommands();
                var barrier = new MemoryBarrier2 { SType = StructureType.MemoryBarrier2, SrcAccessMask = AccessFlags2.MemoryWriteBit, DstAccessMask = AccessFlags2.TransferReadBit };
                VulkanSynchronization.PipelineBarrier(_vk, command, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit, 0, 1, &barrier, 0, null, 0, null);
                var region = new BufferCopy(sourceOffset, 0, size);
                _vk.CmdCopyBuffer(command, source.Handle, buffer, 1, &region);
                _captureBufferSnaps.Add(($"at_{entry.Hash:X16}_{(entry.Address < 256 ? "slot" + entry.Address : "0x" + entry.Address.ToString("X"))}_n{entry.Nth}", buffer, memory, size, address));
            }
        }

        private void CaptureWriteBufferSnaps()
        {
            if (_captureBufferSnaps.Count == 0)
                return;
            SynchronizeGpu();
            foreach (var (name, buffer, memory, size, snapAddress) in _captureBufferSnaps)
            {
                void* mapped;
                if (_vk.MapMemory(_device, memory, 0, size, 0, &mapped) == Result.Success)
                {
                    var bytes = new ReadOnlySpan<byte>(mapped, (int)size);
                    File.WriteAllBytes(Path.Combine(_captureDirectory!, $"buffer_{name}.bin"), bytes.ToArray());
                    var nonZero = 0;
                    for (var i = 0; i + 3 < bytes.Length; i += 4)
                        if (BitConverter.ToUInt32(bytes.Slice(i, 4)) != 0) nonZero++;
                    var guestNow = new byte[size];
                    var guestNonZero = -1;
                    if (TryReadGuest(snapAddress, guestNow))
                    {
                        guestNonZero = 0;
                        for (var i = 0; i + 3 < guestNow.Length; i += 4)
                            if (BitConverter.ToUInt32(guestNow, i) != 0) guestNonZero++;
                    }

                    _captureManifest.Add($"bufferat {name} size=0x{size:X} nonzero_dwords={nonZero} guest_nonzero_at_frame_end={guestNonZero}");
                    _vk.UnmapMemory(_device, memory);
                }

                _vk.DestroyBuffer(_device, buffer, null);
                _deviceInfo.FreeMemory(memory);
            }

            _captureBufferSnaps.Clear();
        }

        private void CaptureReadBuffers()
        {
            var list = Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_BUFFERS");
            if (list is { Length: > 1 } && list[0] == '@')
                list = File.Exists(list[1..]) ? File.ReadAllText(list[1..]).Trim() : null;
            if (string.IsNullOrWhiteSpace(list))
                return;
            foreach (var item in list.Split(','))
            {
                var parts = item.Split(':');
                var address = Convert.ToUInt64(parts[0].Replace("0x", ""), 16);
                var size = Convert.ToUInt64(parts[1].Replace("0x", ""), 16);
                var (source, sourceOffset) = _bufferCache.ObtainBuffer(address, size, false);
                var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
                var command = BeginBatchedGuestCommands();
                var barrier = new MemoryBarrier2 { SType = StructureType.MemoryBarrier2, SrcAccessMask = AccessFlags2.MemoryWriteBit, DstAccessMask = AccessFlags2.TransferReadBit };
                VulkanSynchronization.PipelineBarrier(_vk, command, PipelineStageFlags.AllCommandsBit, PipelineStageFlags.TransferBit, 0, 1, &barrier, 0, null, 0, null);
                var region = new BufferCopy(sourceOffset, 0, size);
                _vk.CmdCopyBuffer(command, source.Handle, buffer, 1, &region);
                SynchronizeGpu();
                void* mapped;
                if (_vk.MapMemory(_device, memory, 0, size, 0, &mapped) == Result.Success)
                {
                    var bytes = new ReadOnlySpan<byte>(mapped, (int)size);
                    File.WriteAllBytes(Path.Combine(_captureDirectory!, $"buffer_0x{address:X}.bin"), bytes.ToArray());
                    var guestCopy = new byte[size]; // TEMP: compare GPU copy against guest memory
                    if (TryReadGuest(address, guestCopy))
                    {
                        File.WriteAllBytes(Path.Combine(_captureDirectory!, $"guest_0x{address:X}.bin"), guestCopy);
                        var differing = 0;
                        for (var i = 0; i + 3 < guestCopy.Length; i += 4)
                            if (BitConverter.ToUInt32(guestCopy, i) != BitConverter.ToUInt32(bytes.Slice(i, 4))) differing++;
                        _captureManifest.Add($"guestcmp 0x{address:X}+0x{size:X} differing_dwords={differing}");
                    }
                    var histogram = new long[256];
                    foreach (var value in bytes) histogram[value]++;
                    _captureManifest.Add($"buffer 0x{address:X}+0x{size:X} top=" + string.Join(",", Enumerable.Range(0, 256).OrderByDescending(index => histogram[index]).Take(8).Select(index => $"{index:X2}:{histogram[index]}")));
                    _vk.UnmapMemory(_device, memory);
                }

                _vk.DestroyBuffer(_device, buffer, null);
                _deviceInfo.FreeMemory(memory);
            }
        }

        private void CaptureReadIndirect()
        {
            if (_captureIndirect.Count == 0)
                return;
            SynchronizeGpu();
            void* mapped;
            if (_vk.MapMemory(_device, _captureArgsMemory, 0, CaptureArgsCapacity * 20, 0, &mapped) == Result.Success)
            {
                var words = new ReadOnlySpan<uint>(mapped, _captureIndirect.Count * 5);
                for (var index = 0; index < _captureIndirect.Count; index++)
                    _captureManifest.Add($"indirect {_captureIndirect[index].Info} indexCount={words[index * 5]} instances={words[index * 5 + 1]} firstIndex={words[index * 5 + 2]} vertexOffset={(int)words[index * 5 + 3]} firstInstance={words[index * 5 + 4]}");
                _vk.UnmapMemory(_device, _captureArgsMemory);
            }

            _captureIndirect.Clear();
        }

        private void CaptureNoteDraw(string kind, uint count, uint instances)
        {
            CaptureCheckFrame();
            if (_captureState == 1)
                _captureManifest.Add($"drawcall pass={_capturePass} marker='{SharpEmu.Libs.Diagnostics.DbgSequence.Marker}' active={_renderingActive} kind={kind} count={count} instances={instances} pipeline={_boundGraphicsPipeline?.Id ?? 0} ps=0x{_boundGraphicsPipeline?.ProfilePixelHash ?? 0:X} vs=0x{_boundGraphicsPipeline?.ProfileVertexHash ?? 0:X} {_dbgDepthState}");
        }

        private void CaptureNotePass(in Gpu.Rendering.RenderingState state)
        {
            CaptureCheckFrame();
            if (_captureState == 1)
                _capturePass++;
        }

        // Called right after a rendering scope ends: copies each attachment as the pass left it.
        private void CaptureSnapshotPass(in Gpu.Rendering.RenderingState state)
        {
            if (_captureState != 1 || Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_NO_PASSES") == "1")
                return;
            for (var index = 0; index < state.ColorAttachmentCount; index++)
                CaptureSnapshotAttachment(state.ColorAttachments[index], index, false);
            if (state.DepthStencilAttachment.HasDepth)
                CaptureSnapshotAttachment(state.DepthStencilAttachment, 8, true);
        }

        private void CaptureSnapshotAttachment(in Gpu.Rendering.RenderingAttachment attachment, int slot, bool depth)
        {
            if (!_captureViews.TryGetValue(attachment.View.Handle, out var target))
            {
                _captureManifest.Add($"pass={_capturePass} slot={slot} unknown view");
                return;
            }

            const ulong budget = 3ul * 1024 * 1024 * 1024;
            var backing = target.Image.Backing;
            // TEMP: SHARPEMU_CAPTURE_LAST_ADDRS=addr,... keeps only the last write of those images
            // (one readback buffer per address, overwritten by every later pass).
            var captureEveryPass = _captureAllAddresses is { } everyPass && everyPass.Contains(target.Address); // TEMP: SHARPEMU_CAPTURE_ALL_ADDRS keeps every pass of those images
            if (!captureEveryPass && _captureLastAddresses is { } lastAddresses && !lastAddresses.Contains(target.Address))
                return;
            var lastIndex = captureEveryPass || _captureLastAddresses is null ? -1 : _captureReadbacks.FindIndex(entry => entry.Target.Address == target.Address && entry.Slot == slot);
            var texelBytes = TexelBytes(target.Format, depth);
            var width = Math.Max(1u, backing.Extent.Width >> (int)target.View.BaseLevel);
            var height = Math.Max(1u, backing.Extent.Height >> (int)target.View.BaseLevel);
            var size = (ulong)width * height * texelBytes;
            if (texelBytes == 0 || _captureBytes + size > budget || backing.Samples != 1 || !backing.Exists ||
                backing.ImageType != ImageType.Type2D || (backing.Usage & ImageUsageFlags.TransferSrcBit) == 0 ||
                target.View.BaseLevel >= backing.MipLevels || target.View.BaseLayer >= backing.Layers)
            {
                _captureManifest.Add($"pass={_capturePass} slot={slot} skip addr=0x{target.Address:X} fmt={target.Format} {width}x{height} samples={backing.Samples} type={backing.ImageType}");
                return;
            }

            if (lastIndex >= 0 && _captureReadbacks[lastIndex].Size == (ulong)width * height * texelBytes)
            {
                var previous = _captureReadbacks[lastIndex];
                var reuseCommand = new CommandBuffer(_scheduler.Current.Handle);
                var reuseRange = new SubresourceRange(target.View.BaseLevel, 1, target.View.BaseLayer, 1);
                target.Image.Transition(ImageLayout.TransferSrcOptimal, AccessFlags.TransferReadBit, reuseRange, reuseCommand);
                var reuseRegion = new BufferImageCopy
                {
                    ImageSubresource = new ImageSubresourceLayers(depth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit, target.View.BaseLevel, target.View.BaseLayer, 1),
                    ImageExtent = new Extent3D(width, height, 1),
                };
                _vk.CmdCopyImageToBuffer(reuseCommand, backing.Handle, ImageLayout.TransferSrcOptimal, previous.Buffer, 1, &reuseRegion);
                target.Image.Transition(attachment.Layout, depth
                    ? AccessFlags.DepthStencilAttachmentReadBit | AccessFlags.DepthStencilAttachmentWriteBit
                    : AccessFlags.ColorAttachmentReadBit | AccessFlags.ColorAttachmentWriteBit, reuseRange, reuseCommand);
                _captureReadbacks[lastIndex] = previous with { Pass = _capturePass };
                return;
            }

            _captureManifest.Add($"attach pass={_capturePass} slot={slot} view={target.Format} backing={backing.Format} usage={backing.Usage} flags={backing.Flags} layout={attachment.Layout} image=0x{backing.Handle.Handle:X} viewHandle=0x{attachment.View.Handle:X} attFmt={attachment.Format} clear={attachment.IsClear}");
            _captureBytes += size;
            var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
            var command = new CommandBuffer(_scheduler.Current.Handle);
            var range = new SubresourceRange(target.View.BaseLevel, 1, target.View.BaseLayer, 1);
            target.Image.Transition(ImageLayout.TransferSrcOptimal, AccessFlags.TransferReadBit, range, command);
            var region = new BufferImageCopy
            {
                ImageSubresource = new ImageSubresourceLayers(depth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit, target.View.BaseLevel, target.View.BaseLayer, 1),
                ImageExtent = new Extent3D(width, height, 1),
            };
            _vk.CmdCopyImageToBuffer(command, backing.Handle, ImageLayout.TransferSrcOptimal, buffer, 1, &region);
            target.Image.Transition(attachment.Layout, depth
                ? AccessFlags.DepthStencilAttachmentReadBit | AccessFlags.DepthStencilAttachmentWriteBit
                : AccessFlags.ColorAttachmentReadBit | AccessFlags.ColorAttachmentWriteBit, range, command);
            _captureReadbacks.Add((_capturePass, slot, target with { Width = width, Height = height }, buffer, memory, size, texelBytes));
        }

        private static readonly HashSet<ulong>? _captureAllAddresses = // TEMP
            Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_ALL_ADDRS") is { Length: > 0 } allText
                ? allText.Split(',').Select(text => Convert.ToUInt64(text.Trim(), 16)).ToHashSet()
                : null;
        private static readonly HashSet<ulong>? _captureLastAddresses = // TEMP
            Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_LAST_ADDRS") is { Length: > 0 } lastText
                ? lastText.Split(',').Select(text => Convert.ToUInt64(text.Trim(), 16)).ToHashSet()
                : null;

        // Storage images a compute dispatch writes, snapshotted right after the dispatch.
        // SHARPEMU_CAPTURE_STORAGE_ADDRS=addr,addr (hex) selects the images.
        private static readonly HashSet<ulong>? _captureStorageAddresses =
            Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_STORAGE_ADDRS") is { Length: > 0 } storageText
                ? storageText.Split(',').Select(text => Convert.ToUInt64(text.Trim(), 16)).ToHashSet()
                : null;
        private static readonly bool _captureStorageAll = Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_STORAGE_ALL") == "1";
        private readonly List<(CachedImage Image, ImageViewDescription View, ulong Address, ulong Hash, int Slot)> _capturePendingStorage = new();
        private int _captureDispatch;
        private readonly List<(CachedImage Image, ImageViewDescription View, ulong Address, ulong Hash, int Slot)> _capturePendingTextures = new();

        // TEMP: sampled textures of the selected compute programs, copied right after their dispatch (not at frame end).
        private void CaptureSnapshotTextures(CommandBuffer command, int dispatch)
        {
            foreach (var (image, view, address, hash, slot) in _capturePendingTextures)
            {
                var backing = image.Backing;
                var depthFormat = backing.Format.ToString().StartsWith("D", StringComparison.Ordinal);
                var texelBytes = TexelBytes(view.Format, false);
                var width = Math.Max(1u, backing.Extent.Width >> (int)view.BaseLevel);
                var height = Math.Max(1u, backing.Extent.Height >> (int)view.BaseLevel);
                var size = (ulong)width * height * texelBytes;
                if (depthFormat || texelBytes == 0 || backing.Samples != 1 || !backing.Exists || backing.ImageType != ImageType.Type2D ||
                    (backing.Usage & ImageUsageFlags.TransferSrcBit) == 0 || view.BaseLayer >= backing.Layers || _captureBytes + size > 3ul * 1024 * 1024 * 1024)
                {
                    _captureManifest.Add($"dtex dispatch={dispatch} hash=0x{hash:X16} slot={slot} skip addr=0x{address:X} fmt={view.Format}");
                    continue;
                }

                _captureManifest.Add($"dtex dispatch={dispatch} hash=0x{hash:X16} slot={slot} addr=0x{address:X} view={view.Format} backing={backing.Format}");
                _captureBytes += size;
                var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
                var range = new SubresourceRange(view.BaseLevel, 1, view.BaseLayer, 1);
                image.Transition(ImageLayout.TransferSrcOptimal, AccessFlags.TransferReadBit, range, command);
                var region = new BufferImageCopy
                {
                    ImageSubresource = new ImageSubresourceLayers(ImageAspectFlags.ColorBit, view.BaseLevel, view.BaseLayer, 1),
                    ImageExtent = new Extent3D(width, height, 1),
                };
                _vk.CmdCopyImageToBuffer(command, backing.Handle, ImageLayout.TransferSrcOptimal, buffer, 1, &region);
                image.Transition(ImageLayout.ShaderReadOnlyOptimal, AccessFlags.ShaderReadBit, range, command);
                _captureReadbacks.Add((dispatch + 5000, slot, new CaptureTarget(image, view, address, false, view.Format, width, height), buffer, memory, size, texelBytes));
            }

            _capturePendingTextures.Clear();
        }

        private void CaptureNoteStorage(CachedImage image, in ImageViewDescription view, ulong address, ulong hash, int slot)
        {
            if (_captureState != 1) return;
            // TEMP: SHARPEMU_CAPTURE_STORAGE_ALL=1 snapshots every near-full-resolution 2D storage image after each dispatch.
            var all = _captureStorageAll && image.Backing.Extent.Width >= 800 && image.Backing.Extent.Height >= 450 && image.Backing.ImageType == ImageType.Type2D;
            if (all || (_captureStorageAddresses is { } addresses && addresses.Contains(address)))
                _capturePendingStorage.Add((image, view, address, hash, slot));
        }

        private void CaptureSnapshotStorage(CommandBuffer command)
        {
            if (_capturePendingTextures is { Count: not 0 })
            {
                if (_captureState == 1) CaptureSnapshotTextures(command, 20000 + _captureDispatch++);
                else _capturePendingTextures.Clear();
            }

            if (_capturePendingStorage is not { Count: not 0 })
                return;
            if (_captureState != 1)
            {
                _capturePendingStorage.Clear();
                return;
            }

            var dispatch = 10000 + _captureDispatch++;
            foreach (var (image, view, address, hash, slot) in _capturePendingStorage)
            {
                var backing = image.Backing;
                var texelBytes = TexelBytes(view.Format, false);
                var width = Math.Max(1u, backing.Extent.Width >> (int)view.BaseLevel);
                var height = Math.Max(1u, backing.Extent.Height >> (int)view.BaseLevel);
                var size = (ulong)width * height * texelBytes;
                if (texelBytes == 0 || backing.Samples != 1 || !backing.Exists || backing.ImageType != ImageType.Type2D ||
                    (backing.Usage & ImageUsageFlags.TransferSrcBit) == 0)
                {
                    _captureManifest.Add($"storage dispatch={dispatch} hash=0x{hash:X16} slot={slot} skip addr=0x{address:X} fmt={view.Format}");
                    continue;
                }

                _captureManifest.Add($"storage dispatch={dispatch} pass={_capturePass} hash=0x{hash:X16} slot={slot} addr=0x{address:X} view={view.Format} backing={backing.Format} image=0x{backing.Handle.Handle:X}");
                _captureBytes += size;
                var buffer = CreateBuffer(size, BufferUsageFlags.TransferDstBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, out var memory);
                var range = new SubresourceRange(view.BaseLevel, 1, view.BaseLayer, 1);
                image.Transition(ImageLayout.TransferSrcOptimal, AccessFlags.TransferReadBit, range, command);
                var region = new BufferImageCopy
                {
                    ImageSubresource = new ImageSubresourceLayers(ImageAspectFlags.ColorBit, view.BaseLevel, view.BaseLayer, 1),
                    ImageExtent = new Extent3D(width, height, 1),
                };
                _vk.CmdCopyImageToBuffer(command, backing.Handle, ImageLayout.TransferSrcOptimal, buffer, 1, &region);
                image.Transition(ImageLayout.General, AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit, range, command);
                _captureReadbacks.Add((dispatch, slot, new CaptureTarget(image, view, address, false, view.Format, width, height), buffer, memory, size, texelBytes));
            }

            _capturePendingStorage.Clear();
        }

        // Called once per presented frame.
        private void CaptureOnPresent()
        {
            if (!CaptureEnabled)
                return;
            var trigger = Path.Combine(_captureBaseDirectory!, "trigger");
            if (_captureState == 0 && (_captureTriggerMode
                    ? File.Exists(trigger)
                    : Stopwatch.GetElapsedTime(_captureStart).TotalSeconds >= _captureAtSeconds))
            {
                if (_captureTriggerMode)
                    File.Delete(trigger);
                _captureBaseFrames = SharpEmu.Libs.Diagnostics.DbgSequence.Frames;
                _captureState = 4; // waiting for the next guest frame boundary
            }
        }

        // TEMP: the presenter presents independently of guest frames, so a capture spans exactly one guest frame (Frame marker to Frame marker).
        private int _captureBaseFrames;
        private void CaptureCheckFrame()
        {
            if (_captureState != 1 && _captureState != 4)
                return;
            var frames = SharpEmu.Libs.Diagnostics.DbgSequence.Frames;
            if (_captureState == 4 && frames != _captureBaseFrames)
            {
                _captureDirectory = Path.Combine(_captureBaseDirectory!, $"cap{_captureIndex++:D2}");
                _capturePass = 0;
                _captureDispatch = 0;
                _captureManifest.Clear();
                _captureTextures.Clear();
                _captureTextureReadbacks.Clear();
                _captureBaseFrames = frames;
                _captureState = 1;
                Console.Error.WriteLine("[DBG][CAPTURE] armed");
            }
            else if (_captureState == 1 && frames != _captureBaseFrames)
            {
                _captureState = 2;
            }
        }

        // Runs at the start of a render iteration, before new guest commands are recorded.
        private void CaptureReadbackIfPending()
        {
            if (_captureState != 2)
                return;
            _captureState = 3;
            EndRendering();
            Directory.CreateDirectory(_captureDirectory!);
            CaptureReadIndirect();
            CaptureReadBuffers();
            CaptureWriteBufferSnaps();
            SynchronizeGpu();
            CaptureWriteCopies();
            CaptureQueueTextures();
            SynchronizeGpu();
            CaptureWriteTextures();
            foreach (var (pass, slot, target, buffer, memory, size, texelBytes) in _captureReadbacks)
            {
                void* mapped;
                var name = $"pass{pass:D4}_s{slot}_{(target.Depth ? "depth" : "color")}_{target.Format}_{target.Width}x{target.Height}_0x{target.Address:X}";
                if (_vk.MapMemory(_device, memory, 0, size, 0, &mapped) != Result.Success || mapped == null)
                {
                    _captureManifest.Add($"{name} map failed");
                    continue;
                }

                try
                {
                    if ((texelBytes == 4 || (texelBytes == 8 && Environment.GetEnvironmentVariable("SHARPEMU_CAPTURE_RAW64") == "1")) && !target.Depth) // TEMP: raw dump next to the PNG (offline value analysis)
                        File.WriteAllBytes(Path.Combine(_captureDirectory!, name + ".raw"), new ReadOnlySpan<byte>(mapped, checked((int)size)).ToArray());
                    var stats = WriteCapturePng(Path.Combine(_captureDirectory!, name + ".png"),
                        new ReadOnlySpan<byte>(mapped, checked((int)size)), target.Format, target.Depth, target.Width, target.Height, texelBytes);
                    _captureManifest.Add($"{name} layer={target.View.BaseLayer} level={target.View.BaseLevel} {stats}");
                }
                catch (Exception exception)
                {
                    _captureManifest.Add($"{name} failed: {exception.Message}");
                }

                _vk.UnmapMemory(_device, memory);
                _vk.DestroyBuffer(_device, buffer, null);
                _deviceInfo.FreeMemory(memory);
            }

            File.WriteAllLines(Path.Combine(_captureDirectory!, "manifest.txt"), _captureManifest);
            Console.Error.WriteLine($"[DBG][CAPTURE] wrote {_captureReadbacks.Count} snapshots ({_capturePass} passes) to {_captureDirectory}");
            _captureReadbacks.Clear();
            if (_captureTriggerMode)
                _captureState = 0;
        }

        private static uint TexelBytes(Format format, bool depth)
        {
            if (depth)
                return format switch { Format.D16Unorm => 2, Format.D32Sfloat or Format.D32SfloatS8Uint or Format.D24UnormS8Uint or Format.X8D24UnormPack32 => 4, _ => 0 };
            return format switch
            {
                Format.R8Unorm or Format.R8SNorm or Format.R8Uint or Format.R8Srgb => 1,
                Format.R8G8Unorm or Format.R8G8SNorm or Format.R16Sfloat or Format.R16Unorm or Format.R16Uint or Format.R16SNorm => 2,
                Format.R8G8B8A8Unorm or Format.R8G8B8A8Srgb or Format.B8G8R8A8Unorm or Format.B8G8R8A8Srgb or Format.R8G8B8A8SNorm or Format.R8G8B8A8Uint or
                    Format.A2B10G10R10UnormPack32 or Format.A2R10G10B10UnormPack32 or Format.B10G11R11UfloatPack32 or Format.E5B9G9R9UfloatPack32 or
                    Format.R16G16Sfloat or Format.R16G16Unorm or Format.R16G16SNorm or Format.R16G16Uint or Format.R32Sfloat or Format.R32Uint or Format.R32Sint => 4,
                Format.R16G16B16A16Sfloat or Format.R16G16B16A16Unorm or Format.R16G16B16A16SNorm or Format.R16G16B16A16Uint or Format.R32G32Sfloat or Format.R32G32Uint => 8,
                Format.R32G32B32A32Sfloat or Format.R32G32B32A32Uint => 16,
                _ => 0,
            };
        }

        // Decodes up to four channels as floats; unknown channels read as 0 (alpha 1).
        private static void DecodeTexel(ReadOnlySpan<byte> texel, Format format, bool depth, Span<float> rgba)
        {
            rgba[0] = rgba[1] = rgba[2] = 0;
            rgba[3] = 1;
            static float Half(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);
            static float F11(uint bits, int mantissaBits)
            {
                var exponent = (int)(bits >> mantissaBits) & 0x1F;
                var mantissa = bits & ((1u << mantissaBits) - 1);
                if (exponent == 0) return mantissa / (float)(1 << mantissaBits) * MathF.Pow(2, -14);
                if (exponent == 31) return 65504;
                return (1 + mantissa / (float)(1 << mantissaBits)) * MathF.Pow(2, exponent - 15);
            }
            var u32 = texel.Length >= 4 ? BitConverter.ToUInt32(texel) : 0;
            if (depth)
            {
                rgba[0] = rgba[1] = rgba[2] = format == Format.D16Unorm ? BitConverter.ToUInt16(texel) / 65535f
                    : format == Format.D32Sfloat || format == Format.D32SfloatS8Uint ? BitConverter.ToSingle(texel) : (u32 & 0xFFFFFF) / 16777215f;
                return;
            }

            switch (format)
            {
                case Format.R8Unorm or Format.R8Srgb or Format.R8Uint: rgba[0] = rgba[1] = rgba[2] = texel[0] / 255f; break;
                case Format.R8SNorm: rgba[0] = rgba[1] = rgba[2] = (sbyte)texel[0] / 127f * 0.5f + 0.5f; break;
                case Format.R8G8Unorm: rgba[0] = texel[0] / 255f; rgba[1] = texel[1] / 255f; break;
                case Format.R8G8SNorm: rgba[0] = (sbyte)texel[0] / 127f * 0.5f + 0.5f; rgba[1] = (sbyte)texel[1] / 127f * 0.5f + 0.5f; break;
                case Format.R16Sfloat: rgba[0] = rgba[1] = rgba[2] = Half(BitConverter.ToUInt16(texel)); break;
                case Format.R16Unorm or Format.R16Uint: rgba[0] = rgba[1] = rgba[2] = BitConverter.ToUInt16(texel) / 65535f; break;
                case Format.R16SNorm: rgba[0] = rgba[1] = rgba[2] = (short)BitConverter.ToUInt16(texel) / 32767f * 0.5f + 0.5f; break;
                case Format.R8G8B8A8Unorm or Format.R8G8B8A8Srgb or Format.R8G8B8A8Uint:
                    rgba[0] = texel[0] / 255f; rgba[1] = texel[1] / 255f; rgba[2] = texel[2] / 255f; rgba[3] = texel[3] / 255f; break;
                case Format.R8G8B8A8SNorm:
                    for (var c = 0; c < 4; c++) rgba[c] = (sbyte)texel[c] / 127f * 0.5f + 0.5f; break;
                case Format.B8G8R8A8Unorm or Format.B8G8R8A8Srgb:
                    rgba[0] = texel[2] / 255f; rgba[1] = texel[1] / 255f; rgba[2] = texel[0] / 255f; rgba[3] = texel[3] / 255f; break;
                case Format.A2B10G10R10UnormPack32:
                    rgba[0] = (u32 & 0x3FF) / 1023f; rgba[1] = ((u32 >> 10) & 0x3FF) / 1023f; rgba[2] = ((u32 >> 20) & 0x3FF) / 1023f; rgba[3] = (u32 >> 30) / 3f; break;
                case Format.A2R10G10B10UnormPack32:
                    rgba[2] = (u32 & 0x3FF) / 1023f; rgba[1] = ((u32 >> 10) & 0x3FF) / 1023f; rgba[0] = ((u32 >> 20) & 0x3FF) / 1023f; rgba[3] = (u32 >> 30) / 3f; break;
                case Format.B10G11R11UfloatPack32:
                    rgba[0] = F11(u32 & 0x7FF, 6); rgba[1] = F11((u32 >> 11) & 0x7FF, 6); rgba[2] = F11((u32 >> 22) & 0x3FF, 5); break;
                case Format.E5B9G9R9UfloatPack32:
                {
                    var scale = MathF.Pow(2, (int)(u32 >> 27) - 15 - 9);
                    rgba[0] = (u32 & 0x1FF) * scale; rgba[1] = ((u32 >> 9) & 0x1FF) * scale; rgba[2] = ((u32 >> 18) & 0x1FF) * scale; break;
                }
                case Format.R16G16Sfloat: rgba[0] = Half(BitConverter.ToUInt16(texel)); rgba[1] = Half(BitConverter.ToUInt16(texel[2..])); break;
                case Format.R16G16Unorm or Format.R16G16Uint: rgba[0] = BitConverter.ToUInt16(texel) / 65535f; rgba[1] = BitConverter.ToUInt16(texel[2..]) / 65535f; break;
                case Format.R16G16SNorm: rgba[0] = (short)BitConverter.ToUInt16(texel) / 32767f * 0.5f + 0.5f; rgba[1] = (short)BitConverter.ToUInt16(texel[2..]) / 32767f * 0.5f + 0.5f; break;
                case Format.R32Sfloat: rgba[0] = rgba[1] = rgba[2] = BitConverter.ToSingle(texel); break;
                case Format.R32Uint or Format.R32Sint: rgba[0] = (u32 & 0xFF) / 255f; rgba[1] = ((u32 >> 8) & 0xFF) / 255f; rgba[2] = ((u32 >> 16) & 0xFF) / 255f; break;
                case Format.R16G16B16A16Sfloat:
                    for (var c = 0; c < 4; c++) rgba[c] = Half(BitConverter.ToUInt16(texel[(c * 2)..])); break;
                case Format.R16G16B16A16Unorm or Format.R16G16B16A16Uint:
                    for (var c = 0; c < 4; c++) rgba[c] = BitConverter.ToUInt16(texel[(c * 2)..]) / 65535f; break;
                case Format.R16G16B16A16SNorm:
                    for (var c = 0; c < 4; c++) rgba[c] = (short)BitConverter.ToUInt16(texel[(c * 2)..]) / 32767f * 0.5f + 0.5f; break;
                case Format.R32G32Sfloat: rgba[0] = BitConverter.ToSingle(texel); rgba[1] = BitConverter.ToSingle(texel[4..]); break;
                case Format.R32G32B32A32Sfloat:
                    for (var c = 0; c < 4; c++) rgba[c] = BitConverter.ToSingle(texel[(c * 4)..]); break;
            }
        }

        private static uint Crc32(byte[] bytes)
        {
            var crc = 0xFFFFFFFFu;
            foreach (var value in bytes)
            {
                crc ^= value;
                for (var bit = 0; bit < 8; bit++)
                    crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
            }

            return ~crc;
        }

        private static bool IsHdr(Format format, bool depth) => depth || format is Format.R16Sfloat or Format.R16G16Sfloat or Format.R16G16B16A16Sfloat or
            Format.B10G11R11UfloatPack32 or Format.E5B9G9R9UfloatPack32 or Format.R32Sfloat or Format.R32G32Sfloat or Format.R32G32B32A32Sfloat;

        // Writes an RGB PNG at most 960 wide; HDR targets are normalized by their 99th percentile.
        private static string WriteCapturePng(string path, ReadOnlySpan<byte> data, Format format, bool depth, uint width, uint height, uint texelBytes)
        {
            var step = Math.Max(1u, (width + 959) / 960);
            var outWidth = (int)((width + step - 1) / step);
            var outHeight = (int)((height + step - 1) / step);
            var values = new float[outWidth * outHeight * 3];
            Span<float> rgba = stackalloc float[4];
            var hdr = IsHdr(format, depth);
            var samples = new List<float>(outWidth * outHeight / 16 + 1);
            float min = float.MaxValue, max = float.MinValue;
            long nonzero = 0;
            for (var y = 0; y < outHeight; y++)
            {
                for (var x = 0; x < outWidth; x++)
                {
                    var offset = ((long)y * step * width + (long)x * step) * texelBytes;
                    DecodeTexel(data.Slice((int)offset, (int)texelBytes), format, depth, rgba);
                    var o = (y * outWidth + x) * 3;
                    for (var c = 0; c < 3; c++)
                    {
                        var v = float.IsFinite(rgba[c]) ? rgba[c] : 0;
                        values[o + c] = v;
                        min = Math.Min(min, v);
                        max = Math.Max(max, v);
                    }
                    if (values[o] != 0 || values[o + 1] != 0 || values[o + 2] != 0) nonzero++;
                    if (((x + y) & 15) == 0) samples.Add(Math.Max(values[o], Math.Max(values[o + 1], values[o + 2])));
                }
            }

            var scale = 1f;
            var bias = 0f;
            if (hdr)
            {
                samples.Sort();
                var p99 = samples.Count == 0 ? 1 : samples[(int)(samples.Count * 0.99f)];
                if (depth)
                {
                    bias = min;
                    scale = max > min ? 1 / (max - min) : 1;
                }
                else
                {
                    scale = p99 > 0 ? 1 / p99 : 1;
                }
            }

            var raw = new byte[outHeight * (outWidth * 3 + 1)];
            for (var y = 0; y < outHeight; y++)
            {
                var row = y * (outWidth * 3 + 1);
                raw[row] = 0;
                for (var i = 0; i < outWidth * 3; i++)
                {
                    var v = (values[y * outWidth * 3 + i] - bias) * scale;
                    if (hdr && !depth) v = MathF.Pow(Math.Clamp(v, 0, 1), 1 / 2.2f);
                    raw[row + 1 + i] = (byte)Math.Clamp((int)(v * 255 + 0.5f), 0, 255);
                }
            }

            using var file = File.Create(path);
            file.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
            void Chunk(string type, byte[] payload)
            {
                Span<byte> length = stackalloc byte[4];
                System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(length, payload.Length);
                file.Write(length);
                var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);
                file.Write(typeBytes);
                file.Write(payload);
                var crc = Crc32(typeBytes.Concat(payload).ToArray());
                Span<byte> crcBytes = stackalloc byte[4];
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(crcBytes, crc);
                file.Write(crcBytes);
            }
            var header = new byte[13];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header, outWidth);
            System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), outHeight);
            header[8] = 8;
            header[9] = 2;
            Chunk("IHDR", header);
            using (var compressed = new MemoryStream())
            {
                using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
                    zlib.Write(raw);
                Chunk("IDAT", compressed.ToArray());
            }
            Chunk("IEND", []);
            return FormattableString.Invariant($"min={min:G4} max={max:G4} nonzero={nonzero * 100.0 / (outWidth * outHeight):F1}% scale={scale:G4}");
        }
    }
}
