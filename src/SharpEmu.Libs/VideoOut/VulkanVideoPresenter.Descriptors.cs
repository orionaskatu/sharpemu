// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.Libs.Gpu.Vulkan;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

// This partial binds the resources of one stage: storage buffers, images, samplers and push data.
internal static unsafe partial class VulkanVideoPresenter
{
    private sealed partial class Presenter
    {
        private const ulong NullStorageBufferBytes = 16;
        private const uint MaxMemoryOffsetAdjustment = 256;
        private const uint TransientDataAlignment = 256;
        private const int MaxImageOccurrences = 64;

        private readonly record struct BufferView(VkBuffer Buffer, ulong Offset, ulong Range);

        private sealed class DescriptorScratch
        {
            public readonly RenderScratchPool<TextureResource> Images = new();
            public readonly RenderScratchPool<Sampler> Samplers = new();
            public readonly RenderScratchPool<uint> ShaderData = new();
            public readonly RenderScratchPool<BufferView> Buffers = new();
            public readonly RenderScratchPool<(BufferDescriptorWords Descriptor, ResourceSlotIdentifier Buffer)> Sources = new();
        }

        private DescriptorScratch? _descriptorScratch;
        private DescriptorScratch Scratch => _descriptorScratch ??= new();

        // The host descriptors of one stage in the order its binding layout names them.
        private sealed class StageDescriptors
        {
            public BufferView[] Buffers = [];
            public TextureResource[] Images = [];
            // The distinct instances of Images when several resources share one binding; null otherwise.
            public TextureResource[]? UniqueTextures;
            public Sampler[] Samplers = [];
            public BufferView GlobalDataShare;
            public BufferView FlattenedTable;
            public BufferView ShaderData;
            public BufferView RuntimeTable;
            public BufferView RuntimeMisses;
        }

        private sealed class PreparedStageBindings(ShaderStageResources stage, ShaderProgramInfo program) : IPreparedBindings
        {
            public ShaderStageResources Stage => stage;

            public ShaderProgramInfo Program => program;

            public SpecializedResourceInfo Resources => program.Resources!;

            public BindingLayout Layout => program.Bindings!;

            public StageDescriptors Descriptors { get; } = new();

            public (BufferDescriptorWords Descriptor, ResourceSlotIdentifier Buffer)[] BufferSources { get; set; } = [];

            public uint[] ShaderData { get; set; } = [];

            // The registered runtime-descriptor images the draw samples through the heap; they
            // take the same transitions as the bound images but no binding of their own.
            public List<TextureResource> RuntimeImages { get; } = [];

            public List<RuntimeImageEntry> RuntimeEntries { get; } = [];

            public TextureResource[] Textures => RuntimeImages.Count == 0 ? Descriptors.UniqueTextures ?? Descriptors.Images : [.. Descriptors.UniqueTextures ?? Descriptors.Images, .. RuntimeImages];
        }

        private static ShaderStage StageOf(ShaderProgramInfo program) => program.Stage switch
        {
            ShaderStageKind.Vertex => ShaderStage.Vertex,
            ShaderStageKind.Pixel => ShaderStage.Pixel,
            ShaderStageKind.Compute => ShaderStage.Compute,
            _ => throw SubmissionScheduler.Fatal($"The stage kind is unknown: stage={program.Stage} hash=0x{program.Hash:X16}."),
        };

        private static ShaderProgramInfo RequireProgram(ShaderStageResources stage)
        {
            var program = stage.Program ?? throw SubmissionScheduler.Fatal("The stage has no program.");
            if (program.Resources is null || program.Bindings is null)
            {
                throw SubmissionScheduler.Fatal($"The stage program has no resource plan: stage={program.Stage} hash=0x{program.Hash:X16}.");
            }

            return program;
        }

        private static TextureNumericClass NumericClassOf(ImageResource image) => image.NumericClass switch
        {
            ImageNumericClass.Uint => TextureNumericClass.Uint,
            ImageNumericClass.Sint => TextureNumericClass.Sint,
            _ => TextureNumericClass.Float,
        };

        private static ShaderImageShape ShapeOf(ImageResource image) => new(
            Volume: image.Dimension == ImageDimension.Dim3D,
            Arrayed: image.Cube || image.Dimension is ImageDimension.Dim1DArray or ImageDimension.Dim2DArray or ImageDimension.Dim2DMsaaArray,
            Cube: image.Cube,
            Storage: image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage,
            DynamicMip: image.MipMode == ImageMipMode.DynamicStorage,
            NumericClass: NumericClassOf(image),
            OneDimensional: image.Dimension is ImageDimension.Dim1D or ImageDimension.Dim1DArray,
            R128: image.R128,
            Multisampled: image.Dimension is ImageDimension.Dim2DMsaa or ImageDimension.Dim2DMsaaArray,
            DepthCompare: image.DepthCompare,
            Atomic: image.Atomic);

        private static bool[] FindResidentBindlessImages(ShaderResourceInfo info, uint[] flattenedTable)
        {
            var resident = new bool[info.Images.Count];
            for (var index = 0; index < info.Images.Count; index++)
            {
                if (info.Images[index].IndirectRoot == DescriptorConstants.NoIndex)
                {
                    resident[index] = true;
                }
            }

            for (var rootIndex = 0; rootIndex < info.Images.Count; rootIndex++)
            {
                var root = info.Images[rootIndex];
                if (root.IndirectRoot != (uint)rootIndex || root.IndirectSearchIterations == 0)
                {
                    continue;
                }

                var candidates = root.IndirectResources;
                var offset = root.IndirectMappingOffset;
                if (candidates.Count == 0 || offset >= flattenedTable.Length)
                {
                    // An incomplete mapping is not a valid basis for dropping a
                    // descriptor: preserve the old eager behavior for this table.
                    foreach (var candidate in candidates)
                    {
                        if (candidate < resident.Length)
                        {
                            resident[(int)candidate] = true;
                        }
                    }

                    if (candidates.Count == 0)
                    {
                        resident[rootIndex] = true;
                    }

                    continue;
                }

                var count = flattenedTable[offset];
                if (count == 0 || (ulong)offset + 1 + (ulong)count * 2 > (ulong)flattenedTable.Length)
                {
                    foreach (var candidate in candidates)
                    {
                        if (candidate < resident.Length)
                        {
                            resident[(int)candidate] = true;
                        }
                    }

                    continue;
                }

                for (var entry = 0u; entry < count; entry++)
                {
                    var candidate = flattenedTable[offset + 2 + entry * 2];
                    if (candidate < candidates.Count)
                    {
                        var resource = candidates[(int)candidate];
                        if (resource < resident.Length)
                        {
                            resident[(int)resource] = true;
                        }
                    }
                }
            }

            return resident;
        }

        private TextureResource ResolveNonResidentImage(ImageResource image, uint[] words, ShaderProgramInfo program, int index)
        {
            // Every unreferenced slot of a table binds the same null image: one binding serves the preparation.
            var nullKey = (ShapeOf(image), image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage);
            if (_prepareMemoActive && _nullTextures.TryGetValue(nullKey, out var sharedNull))
            {
                return sharedNull;
            }

            var created = ResolveNonResidentImageCore(image, words);
            if (_prepareMemoActive)
            {
                _nullTextures[nullKey] = created;
            }

            return created;
        }

        private readonly Dictionary<(ShaderImageShape, bool), TextureResource> _nullTextures = new();

        private TextureResource ResolveNonResidentImageCore(ImageResource image, uint[] words)
        {
            var descriptor = new TextureDescriptorWords(words);
            var resolution = ImageRequestBuilders.NullTextureResolution(ShapeOf(image));
            _ = BeginBatchedGuestCommands();
            var request = resolution.Request;
            var imageIdentifier = _imageCache.FindImage(ref request);
            resolution = resolution with { Request = request };
            imageIdentifier = ImageRequestBuilders.ValidateTextureOwner(_imageCache, imageIdentifier, resolution);
            BindImage(imageIdentifier, image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage);
            return new TextureResource
            {
                Address = descriptor.BaseAddress,
                ImageIdentifier = imageIdentifier,
                Request = request,
                IsStorage = image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage,
                IsResident = true,
                Width = descriptor.Width,
                Height = descriptor.Height,
                DestinationSelect = resolution.Swizzle,
            };
        }

        // Render-state discovery for one shader image; the view is acquired later with the draw.
        // TEMP: SHARPEMU_DBG_TEX_REDIRECT=from:to binds the texture at "to" wherever a descriptor names "from".
        private static readonly bool DbgGfxStores = Environment.GetEnvironmentVariable("SHARPEMU_DBG_GFX_STORES") == "1"; // TEMP
        private uint[]? _dbgRedirectWords;
        private static readonly (ulong From, ulong To)? DbgTextureRedirect =
            Environment.GetEnvironmentVariable("SHARPEMU_DBG_TEX_REDIRECT") is { Length: > 0 } dbgRedirectText && dbgRedirectText.Split(':') is { Length: 2 } dbgParts
                ? (Convert.ToUInt64(dbgParts[0], 16), Convert.ToUInt64(dbgParts[1], 16))
                : null;

        private readonly struct TextureBindingKey : IEquatable<TextureBindingKey>
        {
            private readonly ulong _words0, _words1, _words2, _words3;
            private readonly int _length;
            private readonly ShaderImageShape _shape;
            private readonly bool _storage;
            private readonly int _hash;

            public TextureBindingKey(uint[] words, ShaderImageShape shape, bool storage)
            {
                _words0 = Pack(words, 0);
                _words1 = Pack(words, 2);
                _words2 = Pack(words, 4);
                _words3 = Pack(words, 6);
                _length = words.Length;
                _shape = shape;
                _storage = storage;
                var mixed = (_words0 * 0x9E3779B97F4A7C15UL) ^ (_words1 * 0xC2B2AE3D27D4EB4FUL) ^ (_words2 * 0x165667B19E3779F9UL) ^
                            (_words3 * 0x27D4EB2F165667C5UL) ^ (ulong)_length ^ (storage ? 0x8000UL : 0UL);
                _hash = (int)(mixed ^ (mixed >> 32));
            }

            public bool Equals(TextureBindingKey other) =>
                _hash == other._hash && _words0 == other._words0 && _words1 == other._words1 && _words2 == other._words2 &&
                _words3 == other._words3 && _length == other._length && _storage == other._storage && _shape.Equals(other._shape);

            public override bool Equals(object? obj) => obj is TextureBindingKey other && Equals(other);

            public override int GetHashCode() => _hash;

            private static ulong Pack(uint[] words, int index) =>
                (index < words.Length ? words[index] : 0u) | ((ulong)(index + 1 < words.Length ? words[index + 1] : 0u) << 32);
        }

        private sealed class CachedTextureBinding
        {
            public ResourceSlotIdentifier Image;
            public ImageRequest Request;
            // The image-cache version this lookup is valid for and the addresses it depends on.
            public long Version;
            public ulong RangeStart;
            public ulong RangeEnd;
            // The binding built for it during the preparation numbered SharedEpoch.
            public TextureResource? Shared;
            public int SharedEpoch;
        }

        private readonly Dictionary<TextureBindingKey, CachedTextureBinding> _textureBindings = new();
        private const int MaxTextureBindings = 1 << 16;
        private bool _prepareMemoActive;
        private int _prepareEpoch;
        private const int BoundSetMinimumImages = 256;
        private static readonly bool BoundSetsEnabled = Environment.GetEnvironmentVariable("SHARPEMU_BOUND_SETS") != "0";
        private delegate TextureResource ResolveOne(int index);
        private readonly List<int> _setChanged = new();
        private static long _dbgSetDeltas; // TEMP
        private static long _dbgSetHits, _dbgSetMisses, _dbgMissResident, _dbgMissLog, _dbgMissOverlap, _dbgMissContent, _dbgMissNoSet; // TEMP
        private readonly Dictionary<ShaderResourceInfo, BoundImageSet> _boundSets = new(ReferenceEqualityComparer.Instance);

        // The resolved bindings of one bindless table, reusable while the table's descriptors,
        // the resident subset and the image cache's structure are unchanged.
        private (ulong Start, ulong End) TextureRange(TextureResource texture)
        {
            var data = texture.Request.Description.Data;
            var start = data.Address;
            var end = data.Address + Math.Max(data.Size, 1);
            if (_imageCache.TryGetImage(texture.ImageIdentifier, out var image))
            {
                var found = image.Description.Data;
                start = Math.Min(start, found.Address);
                end = Math.Max(end, found.Address + Math.Max(found.Size, 1));
            }

            return (start, end);
        }

        private enum SetMatch { Miss, Hit, Delta }

        // The resolved bindings of one bindless table, reusable while the table's descriptors,
        // the resident subset and the image cache's structure are unchanged. A table that changed
        // in a few entries keeps the bindings of the others.
        private sealed class BoundImageSet(uint[][] words, bool[]? resident, TextureResource[] images, (ulong Start, ulong End)[] ranges, long version)
        {
            public uint[][] Words = (uint[][])words.Clone();
            public bool[]? Resident = (bool[]?)resident?.Clone();
            public TextureResource[] Images = images;
            public (ulong Start, ulong End)[] Ranges = ranges;
            public TextureResource[]? Unique;
            public long Version = version;
            public const int MaxDelta = 1024;

            public SetMatch Match(uint[][] current, bool[]? currentResident, GuestImageCache imageCache, List<int> changed)
            {
                changed.Clear();
                if (current.Length != Words.Length || (Resident is null) != (currentResident is null))
                {
                    return SetMatch.Miss;
                }

                var cachedWords = Words;
                for (var index = 0; index < cachedWords.Length; index++)
                {
                    var same = ReferenceEquals(cachedWords[index], current[index]) ||
                               cachedWords[index].AsSpan().SequenceEqual(current[index]);
                    if (same && Resident is not null && Resident[index] != currentResident![index])
                    {
                        same = false;
                    }

                    if (!same)
                    {
                        if (changed.Count >= MaxDelta)
                        {
                            return SetMatch.Miss;
                        }

                        changed.Add(index);
                    }
                }

                if (imageCache.StructureVersion != Version)
                {
                    // Images came and went since; the bindings hold unless one overlaps a texture of the set.
                    Span<(ulong Start, ulong End)> changes = stackalloc (ulong, ulong)[256];
                    if (!imageCache.TryGetChangesSince(Version, changes, out var changeCount, out var validated))
                    {
                        return SetMatch.Miss;
                    }

                    for (var index = 0; index < Ranges.Length; index++)
                    {
                        var (start, end) = Ranges[index];
                        for (var change = 0; change < changeCount; change++)
                        {
                            if (changes[change].Start < end && start < changes[change].End)
                            {
                                return SetMatch.Miss;
                            }
                        }
                    }

                    Version = validated;
                }

                return changed.Count == 0 ? SetMatch.Hit : SetMatch.Delta;
            }
        }
        private int _textureMark;
        private readonly HashSet<(ResourceSlotIdentifier, bool)> _preparedImages = new();
        private static long _dbgTexHits, _dbgTexMisses, _dbgTexClears, _dbgTexRevalFail, _dbgTexNotStored; // TEMP

        private CachedTextureBinding ResolveTextureBinding(uint[] words, ShaderImageShape shape)
        {
            var resolution = ImageRequestBuilders.Texture(words, shape);
            var binding = new CachedTextureBinding { Request = resolution.Request };
            binding.Image = _imageCache.FindImage(ref binding.Request, resolution.ExactFormat);
            binding.Image = ImageRequestBuilders.ValidateTextureOwner(_imageCache, binding.Image, resolution with { Request = binding.Request });
            return binding;
        }

        [System.Runtime.CompilerServices.SkipLocalsInit]
        private TextureResource ResolveImageBinding(ImageResource image, uint[] words, ShaderProgramInfo program, int index)
        {
            if (words.Length < 4)
            {
                throw SubmissionScheduler.Fatal($"An image descriptor is too short: image={index} words={words.Length} hash=0x{program.Hash:X16}.");
            }

            if (VisibilityFeedback.Enabled && words.Length >= 8)
            {
                // The visibility pass reads the id target the world pass wrote last frame.
                if (VisibilityFeedback.IdTarget != 0 && VisibilityFeedback.IdTargetWords is null && image.ResourceClass != ShaderCompiler.Resources.ImageResourceClass.Storage &&
                    new TextureDescriptorWords(words).BaseAddress == VisibilityFeedback.IdTarget)
                {
                    VisibilityFeedback.IdTargetWords = (uint[])words.Clone();
                }
                else if (program.Hash == VisibilityFeedback.CullProgramHash && index == 0 && VisibilityFeedback.IdTargetWords is { } idWords)
                {
                    words = (uint[])idWords.Clone();
                }
            }

            if (DbgTextureRedirect is { } dbgRedirect && words.Length >= 8) // TEMP
            {
                var dbgBase = new TextureDescriptorWords(words).BaseAddress;
                if (dbgBase == dbgRedirect.To && !image.ResourceClass.Equals(ShaderCompiler.Resources.ImageResourceClass.Storage) && _dbgRedirectWords is null)
                    _dbgRedirectWords = (uint[])words.Clone();
                else if (dbgBase == dbgRedirect.From && _dbgRedirectWords is not null)
                    words = (uint[])_dbgRedirectWords.Clone(); // the whole T# of the real surface, as its other readers use it
            }

            var storage = image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage;
            var shape = ShapeOf(image);
            _ = BeginBatchedGuestCommands();
            // A descriptor resolves to the same image until an image enters or leaves the cache,
            // so the request building and overlap lookup run once per descriptor, not per draw.
            var structureVersion = _imageCache.StructureVersion;
            if (_textureBindings.Count > MaxTextureBindings)
            {
                _dbgTexClears++; // TEMP
                _textureBindings.Clear();
            }

            var key = new TextureBindingKey(words, shape, storage);
            var found = _textureBindings.TryGetValue(key, out var known);
            if (found && known!.Version != structureVersion)
            {
                // Images came and went since; the lookup holds unless one of them overlaps it.
                if (_imageCache.TryRevalidate(known.Version, known.RangeStart, known.RangeEnd, out var validated))
                {
                    known.Version = validated;
                }
                else
                {
                    found = false;
                    _dbgTexRevalFail++; // TEMP
                }
            }

            if (!found)
            {
                _dbgTexMisses++; // TEMP
                known = ResolveTextureBinding(words, shape);
                if (_imageCache.StructureVersion == structureVersion)
                {
                    known.Version = structureVersion;
                    var data = known.Request.Description.Data;
                    var start = data.Address;
                    var end = data.Address + Math.Max(data.Size, 1);
                    if (_imageCache.TryGetImage(known.Image, out var foundImage))
                    {
                        var foundData = foundImage.Description.Data;
                        start = Math.Min(start, foundData.Address);
                        end = Math.Max(end, foundData.Address + Math.Max(foundData.Size, 1));
                    }

                    known.RangeStart = start;
                    known.RangeEnd = end;
                    _textureBindings[key] = known;
                }
                else
                {
                    _dbgTexNotStored++; // TEMP
                }
            }
            else
            {
                _dbgTexHits++; // TEMP
            }

            var imageIdentifier = known.Image;
            if (_prepareMemoActive && !storage && found && known.SharedEpoch == _prepareEpoch && known.Shared is { } sharedHit)
            {
                return sharedHit;
            }

            // Within one preparation a repeated image needs its recency touch and binding mark once.
            if (!_prepareMemoActive || _preparedImages.Add((imageIdentifier, storage)))
            {
                if (found)
                {
                    _imageCache.TouchFoundImage(known.Image);
                }

                BindImage(imageIdentifier, storage);
            }
            if (Diagnostics.DbgTargetWatch.Addresses.Count != 0 && _imageCache.TryGetImage(imageIdentifier, out var dbgImage) && Diagnostics.DbgTargetWatch.Addresses.Contains(dbgImage.Description.Data.Address)) // TEMP
                Diagnostics.DbgTargetWatch.Log($"resolve {program.Hash:X} {index} {storage}", () => $"resolved image 0x{dbgImage.Description.Data.Address:X} stage={program.Stage} hash=0x{program.Hash:X16} slot={index} storage={storage} (runtime={index < 0})");
            var descriptor = new TextureDescriptorWords(words);
            CaptureNoteTexture(program, index, imageIdentifier, known.Request, descriptor.BaseAddress, words); // TEMP
            if (ShouldTraceTextureBindings())
            {
                var cached = _imageCache.GetImage(imageIdentifier);
                var description = cached.Description;
                Console.Error.WriteLine(
                    $"TextureBinding stage={program.Stage} hash=0x{program.Hash:X16} index={index} " +
                    $"address=0x{new TextureDescriptorWords(words).BaseAddress:X16} " +
                    $"descriptor={descriptor.BaseAddress:X16} size={descriptor.Width + 1}x{descriptor.Height + 1} " +
                    $"format={(uint)descriptor.Format} tile={(uint)descriptor.TileMode} " +
                    $"image=0x{description.Data.Address:X16} size=0x{description.Data.Size:X} " +
                    $"extent={description.Extent.Width}x{description.Extent.Height} pitch={description.Pitch} " +
                    $"guestFormat={(uint)description.GuestFormat} imageTile={(uint)description.TileMode} " +
                    $"backing={cached.Backing.Extent.Width}x{cached.Backing.Extent.Height} format={cached.Backing.Format}");
            }
            // Many descriptors of a bindless table name one image view; they share one binding.
            var resolved = new TextureResource
            {
                Address = descriptor.BaseAddress,
                ImageIdentifier = imageIdentifier,
                Request = known.Request,
                IsStorage = storage,
                IsResident = true,
                DestinationSelect = words[3] & 0xFFFu,
                Width = descriptor.Width,
                Height = descriptor.Height,
            };
            if (_prepareMemoActive && !storage)
            {
                known.Shared = resolved;
                known.SharedEpoch = _prepareEpoch;
            }

            return resolved;
        }

        // Compare bits stay only on depth-compare samplers; a forced point sampler drops its filters.
        // A sampler takes the numeric class of the views it samples; integer only when every
        // paired view is integer, since a float view needs a float border and filtering.
        private static bool SamplesIntegerViews(ShaderResourceInfo info, TextureResource[] images, int sampler)
        {
            var paired = false;
            foreach (var pair in info.SampledPairs)
            {
                if (pair.Sampler != sampler || pair.Image >= images.Length || images[pair.Image].IsHostMovie || !images[pair.Image].IsResident)
                {
                    continue;
                }

                if (!ViewFormatRules.IsIntegerFormat(images[pair.Image].Request.View.Format))
                {
                    return false;
                }

                paired = true;
            }

            return paired;
        }

        private Sampler ResolveSampler(SamplerResource sampler, uint[] words, ShaderProgramInfo program, int index, ShaderStageResources stage,
            bool integerView)
        {
            if (words.Length < 4)
            {
                throw SubmissionScheduler.Fatal($"A sampler descriptor is too short: sampler={index} words={words.Length} hash=0x{program.Hash:X16}.");
            }

            Span<uint> native = stackalloc uint[4] { words[0], words[1], words[2], words[3] };
            if (!sampler.DepthCompare)
            {
                native[0] &= ~(0x7u << 12);
            }

            if (sampler.ForcePointFiltering)
            {
                var mipmapped = ((native[2] >> 26) & 0x3u) != 0;
                native[2] &= ~(0xFFu << 20);
                native[2] |= 1u << 24;
                if (mipmapped)
                {
                    native[2] |= 1u << 26;
                }
            }

            var descriptor = new SamplerDescriptorWords(native);
            if (descriptor.MaxAnisotropyRatio > 4 &&
                (descriptor.MagnifyFilter >= (uint)SamplerFilter.AnisotropicPoint ||
                 descriptor.MinifyFilter >= (uint)SamplerFilter.AnisotropicPoint))
            {
                Console.Error.WriteLine(
                    $"[GPU][ERROR] Sampler source: stage={program.Stage} hash=0x{program.Hash:X16} shader=0x{stage.ShaderBase:X16} " +
                    $"sampler={index} source={sampler.Source} pc=0x{sampler.FirstUsePc:X} " +
                    $"descriptor=[{string.Join(",", words.Select(word => $"{word:X8}"))}] " +
                    $"user_data=[{string.Join(",", stage.Resources.UserData.Select(word => $"{word:X8}"))}]");
            }

            return _samplerStore.GetSampler(descriptor, integerView);
        }

        // The guest textures the movie path matches; built only while a decoded frame is active.
        private static List<GuestDrawTexture>? MovieCandidates(SpecializedResourceInfo resources, ResourceSnapshot snapshot)
        {
            var textures = new List<GuestDrawTexture>(resources.Info.Images.Count);
            for (var index = 0; index < resources.Info.Images.Count; index++)
            {
                var words = snapshot.Images[index];
                var storage = resources.Info.Images[index].ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage;
                textures.Add(AgcExports.TryDecodeTextureDescriptor(words, out var descriptor)
                    ? new GuestDrawTexture(descriptor.Address, descriptor.Width, descriptor.Height, descriptor.Format, descriptor.NumberType, [], false, storage, DstSelect: descriptor.DstSelect)
                    : new GuestDrawTexture(0, 1, 1, 0, 0, [], true, storage));
            }

            return textures;
        }

        // TEMP: SHARPEMU_DBG_PREP_STATS=1 prints, every 10 s, the programs that cost the most in PrepareBindings.
        private static readonly bool DbgPrepStats = Environment.GetEnvironmentVariable("SHARPEMU_DBG_PREP_STATS") == "1";
        private static readonly Dictionary<(ulong, ShaderStageKind), (int Count, long Ticks, long Images, long BindTicks)> _dbgPrep = new();
        private static long _dbgPrepLast = Environment.TickCount64;

        public IPreparedBindings PrepareBindings(ShaderStageResources stage)
        {
            if (!DbgPrepStats)
                return PrepareBindingsCore(stage);
            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            var result = PrepareBindingsCore(stage);
            var elapsed = System.Diagnostics.Stopwatch.GetTimestamp() - start;
            var program = ((PreparedStageBindings)result).Program;
            var key = (program.Hash, program.Stage);
            _dbgPrep.TryGetValue(key, out var entry);
            _dbgPrep[key] = (entry.Count + 1, entry.Ticks + elapsed, entry.Images + ((PreparedStageBindings)result).Descriptors.Images.Length, entry.BindTicks);
            if (Environment.TickCount64 - _dbgPrepLast >= 10000)
            {
                _dbgPrepLast = Environment.TickCount64;
                Console.Error.WriteLine($"[DBG][PREP] texbinding hits={_dbgTexHits} misses={_dbgTexMisses} clears={_dbgTexClears} revalfail={_dbgTexRevalFail} notstored={_dbgTexNotStored} entries={_textureBindings.Count}");
                var dbgF = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                Console.Error.WriteLine($"[DBG][PREP] sections(ms) resident={_dbgSections[0] * dbgF:F0} images={_dbgSections[1] * dbgF:F0} unique={_dbgSections[2] * dbgF:F0} tail={_dbgSections[3] * dbgF:F0} buffers={_dbgSections[4] * dbgF:F0} | bindBuffers={_dbgSections[5] * dbgF:F0} devRanges={_dbgSections[6] * dbgF:F0} bindImages={_dbgSections[7] * dbgF:F0} flattened={_dbgSections[8] * dbgF:F0}");
                Array.Clear(_dbgSections);
                Console.Error.WriteLine($"[DBG][PREP] boundsets hits={_dbgSetHits} deltas={_dbgSetDeltas} misses={_dbgSetMisses}");
                _dbgSetHits = _dbgSetMisses = _dbgSetDeltas = 0;
                if (_dbgUniqueOps != 0)
                    Console.Error.WriteLine($"[DBG][PREP] bulk draws={_dbgUniqueOps} avg unique textures={_dbgUniqueSum / _dbgUniqueOps} distinct images={_dbgDistinctSum / _dbgUniqueOps} resident={_dbgResidentSum / _dbgUniqueOps}");
                _dbgUniqueSum = _dbgUniqueOps = _dbgDistinctSum = _dbgResidentSum = 0;
                _dbgTexHits = _dbgTexMisses = _dbgTexClears = _dbgTexRevalFail = _dbgTexNotStored = 0;
                var total = _dbgPrep.Values.Sum(v => v.Ticks + v.BindTicks);
                Console.Error.WriteLine($"[DBG][PREP] programs={_dbgPrep.Count} total_ms={total * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} ops={_dbgPrep.Values.Sum(v => v.Count)}");
                foreach (var (k, v) in _dbgPrep.OrderByDescending(x => x.Value.Ticks + x.Value.BindTicks).Take(14))
                    Console.Error.WriteLine($"[DBG][PREP]   {k.Item2} 0x{k.Item1:X16} n={v.Count} ms={v.Ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} us/op={v.Ticks * 1e6 / System.Diagnostics.Stopwatch.Frequency / v.Count:F0} images/op={v.Images / (double)v.Count:F1} bind_us/op={v.BindTicks * 1e6 / System.Diagnostics.Stopwatch.Frequency / v.Count:F0}");
                _dbgPrep.Clear();
            }

            return result;
        }

        private IPreparedBindings PrepareBindingsCore(ShaderStageResources stage)
        {
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DescriptorPreparation);
            var preparation = RequirePreparation();
            var program = RequireProgram(stage);
            var resources = program.Resources!;
            var layout = program.Bindings!;
            var snapshot = stage.Resources;
            var info = resources.Info;
            if (snapshot.Images.Length != info.Images.Count || snapshot.Samplers.Length != info.Samplers.Count || snapshot.Buffers.Length != info.Buffers.Count)
            {
                throw SubmissionScheduler.Fatal(
                    $"The resource snapshot does not match the program: hash=0x{program.Hash:X16} images={snapshot.Images.Length}/{info.Images.Count} " +
                    $"samplers={snapshot.Samplers.Length}/{info.Samplers.Count} buffers={snapshot.Buffers.Length}/{info.Buffers.Count}.");
            }

            var prepared = new PreparedStageBindings(stage, program);
            // Register ownership before preparation can fail halfway through.
            preparation.Stages.Add(prepared);
            var descriptors = prepared.Descriptors;
            descriptors.Images = Scratch.Images.Rent(info.Images.Count);
            var movieCandidates = _hostMovieFramePixels is null ? null : MovieCandidates(resources, snapshot);
            var hostMovie = movieCandidates is null ? HostMovieTextureBindings.None : FindHostMovieTextureBindings(movieCandidates);
            var dbgS0 = DbgSec(); // TEMP
            var residentImages = layout.UsesBindlessImages ? FindResidentBindlessImages(info, snapshot.FlattenedResourceTable) : null;
            DbgSecEnd(0, ref dbgS0); // TEMP
            _preparedImages.Clear();
            _prepareEpoch++;
            _nullTextures.Clear();
            _prepareMemoActive = info.Images.Count >= 64;
            // A bindless table that names the same textures as the previous preparation of this
            // program keeps that preparation's bindings; only the per-draw marks are repeated.
            var setEligible = BoundSetsEnabled && info.Images.Count >= BoundSetMinimumImages && movieCandidates is null;
            var structureVersionBefore = _imageCache.StructureVersion;
            BoundImageSet? boundSet = null;
            var match = SetMatch.Miss;
            if (setEligible && _boundSets.TryGetValue(info, out var candidateSet))
            {
                match = candidateSet.Match(snapshot.Images, residentImages, _imageCache, _setChanged);
                if (match != SetMatch.Miss)
                {
                    boundSet = candidateSet;
                }
            }

            ResolveOne resolveOne = index => index == hostMovie.Luma
                ? CreateHostMovieTextureResource(movieCandidates![index], plane: 0)
                : index == hostMovie.Chroma
                    ? CreateHostMovieTextureResource(movieCandidates![index], plane: 1)
                    : residentImages is { } && !residentImages[index]
                        ? ResolveNonResidentImage(info.Images[index], snapshot.Images[index], program, index)
                        : ResolveImageBinding(info.Images[index], snapshot.Images[index], program, index);
            if (boundSet is not null)
            {
                descriptors.Images = (TextureResource[])boundSet.Images.Clone();
                if (match == SetMatch.Delta)
                {
                    foreach (var index in _setChanged)
                    {
                        descriptors.Images[index] = resolveOne(index);
                    }

                    if (_imageCache.StructureVersion == structureVersionBefore && !_setChanged.Exists(index => descriptors.Images[index].IsStorage || descriptors.Images[index].IsHostMovie))
                    {
                        boundSet.Words = (uint[][])snapshot.Images.Clone();
                        boundSet.Resident = (bool[]?)residentImages?.Clone();
                        boundSet.Images = (TextureResource[])descriptors.Images.Clone();
                        foreach (var index in _setChanged)
                        {
                            boundSet.Ranges[index] = TextureRange(descriptors.Images[index]);
                        }

                        boundSet.Unique = null;
                        boundSet.Version = structureVersionBefore;
                    }
                    else
                    {
                        _boundSets.Remove(info);
                    }

                    descriptors.UniqueTextures = UniqueTextures(descriptors.Images);
                    _dbgSetDeltas++; // TEMP
                }
                else
                {
                    descriptors.UniqueTextures = boundSet.Unique ??= UniqueTextures(boundSet.Images);
                    _dbgSetHits++; // TEMP
                }

                foreach (var texture in descriptors.UniqueTextures)
                {
                    BindImage(texture.ImageIdentifier, texture.IsStorage);
                }
            }
            else
            {
                if (setEligible) _dbgSetMisses++; // TEMP
                for (var index = 0; index < info.Images.Count; index++)
                {
                    descriptors.Images[index] = resolveOne(index);
                }
            }

            DbgSecEnd(1, ref dbgS0); // TEMP
            if (boundSet is null && _prepareMemoActive)
            {
                descriptors.UniqueTextures = UniqueTextures(descriptors.Images);
                if (setEligible && _imageCache.StructureVersion == structureVersionBefore && !descriptors.UniqueTextures.Any(texture => texture.IsStorage || texture.IsHostMovie))
                {
                    if (_boundSets.Count >= 64)
                    {
                        _boundSets.Clear();
                    }

                    var ranges = new (ulong Start, ulong End)[descriptors.Images.Length];
                    for (var index = 0; index < ranges.Length; index++)
                    {
                        ranges[index] = TextureRange(descriptors.Images[index]);
                    }

                    _boundSets[info] = new BoundImageSet(snapshot.Images, residentImages, (TextureResource[])descriptors.Images.Clone(), ranges, structureVersionBefore)
                    {
                        Unique = descriptors.UniqueTextures,
                    };
                }

                if (DbgPrepStats && info.Images.Count >= 4000)
                {
                    _dbgUniqueSum += descriptors.UniqueTextures.Length;
                    _dbgUniqueOps++;
                    var dbgDistinctImages = new HashSet<ResourceSlotIdentifier>();
                    var dbgResident = 0;
                    for (var i = 0; i < descriptors.Images.Length; i++)
                    {
                        dbgDistinctImages.Add(descriptors.Images[i].ImageIdentifier);
                        if (residentImages is null || residentImages[i]) dbgResident++;
                    }

                    _dbgDistinctSum += dbgDistinctImages.Count;
                    _dbgResidentSum += dbgResident;
                }
            }

            DbgSecEnd(2, ref dbgS0); // TEMP

            _prepareMemoActive = false;
            if (layout.Find(DescriptorBindingKind.RuntimeDescriptorTable) is not null)
            {
                PrepareRuntimeDescriptors(prepared);
            }

            descriptors.Samplers = Scratch.Samplers.Rent(info.Samplers.Count);
            for (var index = 0; index < info.Samplers.Count; index++)
            {
                descriptors.Samplers[index] = ResolveSampler(info.Samplers[index], snapshot.Samplers[index], program, index, stage,
                    SamplesIntegerViews(info, descriptors.Images, index));
            }

            var shaderData = Scratch.ShaderData.Rent(checked((int)layout.ShaderDataDwordCount));
            prepared.ShaderData = shaderData;
            for (var index = 0; index < layout.UserDataRegisters.Count; index++)
            {
                var register = layout.UserDataRegisters[index];
                var userIndex = (int)(register - program.UserDataBase);
                if (register < program.UserDataBase || userIndex >= snapshot.UserData.Length)
                {
                    throw SubmissionScheduler.Fatal($"A user register is outside the draw's user data: register={register} base={program.UserDataBase} count={snapshot.UserData.Length} hash=0x{program.Hash:X16}.");
                }

                shaderData[index] = snapshot.UserData[userIndex];
            }

            if (layout.UsesShaderBase)
            {
                shaderData[layout.ShaderBaseDword] = (uint)stage.ShaderBase;
                shaderData[layout.ShaderBaseDword + 1] = (uint)(stage.ShaderBase >> 32);
            }

            stage.WriteDispatchThreadLimits(shaderData);
            prepared.ShaderData = shaderData;
            if (layout.Find(DescriptorBindingKind.GlobalDataShare) is not null)
            {
                descriptors.GlobalDataShare = new BufferView(_bufferCache.GdsBuffer.Handle, 0, Vk.WholeSize);
            }

            DbgSecEnd(3, ref dbgS0); // TEMP
            FindDeviceAddressBuffers(prepared);
            FindBuffers(prepared);
            DbgSecEnd(4, ref dbgS0); // TEMP
            ValidateDrawImageTypes(prepared);
            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write(
                    $"Bindings prepared stage={program.Stage} hash=0x{program.Hash:X16} buffers={info.Buffers.Count} images={info.Images.Count} " +
                    $"samplers={info.Samplers.Count} userData={snapshot.UserData.Length} flattened={snapshot.FlattenedResourceTable.Length} shaderData={shaderData.Length}");
            }

            return prepared;
        }

        // Reject incompatible draw views before buffer writes or image transitions are recorded.
        // TEMP: SHARPEMU_DBG_IMAGE_USE=addr,addr logs every shader that binds those images.
        private static readonly HashSet<ulong>? DbgImageUseAddresses =
            Environment.GetEnvironmentVariable("SHARPEMU_DBG_IMAGE_USE") is { Length: > 0 } dbgText
                ? dbgText.Split(',').Select(text => Convert.ToUInt64(text.Trim(), 16)).ToHashSet()
                : null;
        private static readonly HashSet<(ulong, ulong, bool, Format)> DbgImageUseSeen = new();
        // TEMP: SHARPEMU_DBG_PROGRAM_IMAGES=hash,... logs every image those programs bind.
        private static readonly bool DbgOnePixel = Environment.GetEnvironmentVariable("SHARPEMU_DBG_ONEPX") == "1"; // TEMP
        private static readonly HashSet<ulong> DbgProgramImageHashes = (Environment.GetEnvironmentVariable("SHARPEMU_DBG_PROGRAM_IMAGES") ?? "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(text => Convert.ToUInt64(text.Replace("0x", ""), 16)).ToHashSet();

        private void ValidateDrawImageTypes(PreparedStageBindings prepared)
        {
            if (prepared.Program.Stage == ShaderStageKind.Compute)
            {
                return;
            }

            foreach (var binding in prepared.Textures)
            {
                if (binding.IsHostMovie || !binding.IsResident || IsStaleImage(binding.ImageIdentifier, out var image) || image is null)
                {
                    continue;
                }

                var view = binding.IsStorage ? binding.Request.View with { LevelCount = 1 } : binding.Request.View;
                if (!image.SupportsViewType(view))
                {
                    throw new DrawImageTypeMismatchException(
                        prepared.Program.Hash, binding.Address, image.Backing.ImageType, view.Type);
                }
            }
        }

        // TEMP: SHARPEMU_DBG_BUF_WRITERS=addr:size,... logs each program (once per binding) that
        // binds a written buffer or device-address range overlapping one of the ranges.
        private static readonly (ulong Address, ulong Size)[] DbgBufferWriterRanges =
            (Environment.GetEnvironmentVariable("SHARPEMU_DBG_BUF_WRITERS") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(item => item.Split(':')).Select(parts => (Convert.ToUInt64(parts[0].Replace("0x", ""), 16), Convert.ToUInt64(parts[1].Replace("0x", ""), 16))).ToArray();
        private static readonly HashSet<(ulong, int, ulong, bool)> DbgBufferWritersSeen = new();

        // TEMP: SHARPEMU_DBG_ARGS_WRITER_VS=vsHash,... reports the last program that wrote the
        // indirect arguments of those vertex shaders' draws.
        internal static readonly HashSet<ulong> DbgArgsWriterVertexHashes =
            (Environment.GetEnvironmentVariable("SHARPEMU_DBG_ARGS_WRITER_VS") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(text => Convert.ToUInt64(text.Replace("0x", ""), 16)).ToHashSet();
        private readonly Dictionary<ulong, (ulong Hash, int Slot, ulong Address, ulong Size, ulong Tick)> _dbgPageWriters = new();
        private readonly Dictionary<(ulong, ulong), int> _dbgArgsReports = new();

        public void DebugReportArgsWriter(ulong vertexHash, ulong argsAddress) // TEMP
        {
            if (!DbgArgsWriterVertexHashes.Contains(vertexHash))
                return;
            var found = _dbgPageWriters.TryGetValue(argsAddress >> 12, out var writer);
            var key = (vertexHash, found ? writer.Hash : 0);
            var seen = _dbgArgsReports.GetValueOrDefault(key);
            _dbgArgsReports[key] = seen + 1;
            if ((seen & (seen - 1)) == 0)
                Console.Error.WriteLine(found
                    ? $"[DBG][ARGSWRITER] vs=0x{vertexHash:X16} args=0x{argsAddress:X} writer=0x{writer.Hash:X16} slot={writer.Slot} range=0x{writer.Address:X}+0x{writer.Size:X} writer_tick={writer.Tick} now={_scheduler.CurrentTick} n={seen + 1}"
                    : $"[DBG][ARGSWRITER] vs=0x{vertexHash:X16} args=0x{argsAddress:X} writer=none n={seen + 1}");
        }

        // TEMP: SHARPEMU_DBG_READS_WRITER=programHash,... reports the last writer of each buffer those programs read.
        private static readonly HashSet<ulong> DbgReadsWriterHashes =
            (Environment.GetEnvironmentVariable("SHARPEMU_DBG_READS_WRITER") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(text => Convert.ToUInt64(text.Replace("0x", ""), 16)).ToHashSet();

        private void DbgNoteBufferWriter(ShaderProgramInfo program, int slot, ulong address, ulong size, bool written)
        {
            if (written && Diagnostics.DbgTargetWatch.PhysicalWatch.Length != 0)
                Diagnostics.DbgTargetWatch.CheckPhysical("buffer", address, size, () => $"hash=0x{program.Hash:X16} slot={slot}");
            if (DbgReadsWriterHashes.Contains(program.Hash))
            {
                var found = _dbgPageWriters.TryGetValue(address >> 12, out var writer);
                var key = (program.Hash, (ulong)(uint)slot * 0x10000 + (found ? writer.Hash & 0xFFFF : 0));
                var seen = _dbgArgsReports.GetValueOrDefault(key);
                _dbgArgsReports[key] = seen + 1;
                if ((seen & (seen - 1)) == 0)
                    Console.Error.WriteLine($"[DBG][READSWRITER] reader=0x{program.Hash:X16} slot={slot} range=0x{address:X}+0x{size:X} written={written} " +
                        (found ? $"writer=0x{writer.Hash:X16} wslot={writer.Slot} wrange=0x{writer.Address:X}+0x{writer.Size:X} wtick={writer.Tick} now={_scheduler.CurrentTick}" : "writer=none") + $" n={seen + 1}");
            }
            if (written && (DbgArgsWriterVertexHashes.Count != 0 || DbgReadsWriterHashes.Count != 0) && size <= 64UL << 20)
            {
                for (var page = address >> 12; page <= (address + size - 1) >> 12; page++)
                    _dbgPageWriters[page] = (program.Hash, slot, address, size, _scheduler.CurrentTick);
            }

            foreach (var (watch, watchSize) in DbgBufferWriterRanges)
            {
                if (address < watch + watchSize && watch < address + size && DbgBufferWritersSeen.Add((program.Hash, slot, address, written)))
                    Console.Error.WriteLine($"[DBG][BUFWRITER] stage={program.Stage} hash=0x{program.Hash:X16} slot={slot} range=0x{address:X}+0x{size:X} written={written} tick={_scheduler.CurrentTick}");
            }
        }

        private void FindDeviceAddressBuffers(PreparedStageBindings prepared)
        {
            foreach (var range in prepared.Stage.Resources.DeviceAddressRanges)
            {
                if ((DbgBufferWriterRanges.Length != 0 || DbgArgsWriterVertexHashes.Count != 0 || DbgReadsWriterHashes.Count != 0 || Diagnostics.DbgTargetWatch.PhysicalWatch.Length != 0) && range.Planned && range.Size != 0) DbgNoteBufferWriter(prepared.Program, -1 - (int)range.Handle, range.Base, range.Size, range.Written); // TEMP
                if (!range.Planned || range.Size == 0 ||
                    range.Base >= PageOwnerTable.AddressSpaceSize || range.Size > PageOwnerTable.AddressSpaceSize - range.Base ||
                    (!range.Written && !_guestMemory.CanRead(range.Base, 1)))
                {
                    continue;
                }

                _ = _bufferCache.FindBuffer(range.Base, ClampMappedSize(range.Base, range.Size));
            }
        }

        // The cache buffer each descriptor's range lives in; a null descriptor has none.
        private void FindBuffers(PreparedStageBindings prepared)
        {
            var program = prepared.Program;
            var snapshot = prepared.Stage.Resources;
            var sources = Scratch.Sources.Rent(prepared.Resources.Info.Buffers.Count);
            prepared.BufferSources = sources;
            for (var index = 0; index < sources.Length; index++)
            {
                var words = snapshot.Buffers[index];
                if (words.Length < 4)
                {
                    throw SubmissionScheduler.Fatal($"A buffer descriptor is too short: buffer={index} words={words.Length} hash=0x{program.Hash:X16}.");
                }

                var descriptor = BufferDescriptorWords.From(words);
                var requested = descriptor.Footprint() ?? throw SubmissionScheduler.Fatal(
                    $"A storage buffer descriptor footprint overflows: buffer={index} stride={descriptor.Stride} records={descriptor.RecordCount} hash=0x{program.Hash:X16}.");
                if (descriptor.Address == 0 || requested == 0)
                {
                    sources[index] = (descriptor, default);
                    continue;
                }

                var size = ClampMappedSize(descriptor.Address, requested, prepared, index);
                sources[index] = (descriptor, _bufferCache.FindBuffer(descriptor.Address, size));
            }

            prepared.BufferSources = sources;
        }

        // Uploads every mapped range into the cache before a device-address draw; the fault pass follows.
        public void PrepareDeviceAddresses()
        {
            var preparation = RequirePreparation();
            ulong vertexProgramHash = 0, pixelProgramHash = 0, computeProgramHash = 0;
            if (BufferUploadProfile.Enabled)
            {
                foreach (var stage in preparation.Stages)
                {
                    if (!stage.Program.UsesDeviceAddresses) continue;
                    switch (stage.Program.Stage)
                    {
                        case ShaderStageKind.Vertex: vertexProgramHash = stage.Program.Hash; break;
                        case ShaderStageKind.Pixel: pixelProgramHash = stage.Program.Hash; break;
                        case ShaderStageKind.Compute: computeProgramHash = stage.Program.Hash; break;
                    }
                }
            }
            using var profileScope = BufferUploadProfile.BeginSweep(vertexProgramHash, pixelProgramHash, computeProgramHash);
            var memory = GuestGpuMemoryHook.Current ?? throw SubmissionScheduler.Fatal("A device-address program needs the guest GPU memory registry.");
            var spans = DeviceAddressSpans(memory);
            var traceAddress = GuestGpuMemoryHook.TraceAddress;
            if (traceAddress != 0)
            {
                foreach (var stage in preparation.Stages)
                {
                    if (!stage.Program.UsesDeviceAddresses) continue;
                    GuestGpuMemoryHook.Trace(traceAddress, 1,
                        $"device-address-program submission_tick={_scheduler.CurrentTick} stage={stage.Program.Stage} hash=0x{stage.Program.Hash:X16} shader=0x{stage.Stage.ShaderBase:X16} ranges={stage.Stage.Resources.DeviceAddressRanges.Length}");
                    foreach (var range in stage.Stage.Resources.DeviceAddressRanges)
                        GuestGpuMemoryHook.Trace(traceAddress, 1,
                            $"device-address-range submission_tick={_scheduler.CurrentTick} hash=0x{stage.Program.Hash:X16} handle={range.Handle} base=0x{range.Base:X16} size=0x{range.Size:X} planned={range.Planned} written={range.Written}");
                }
            }
            if (traceAddress != 0 && !memory.Covers(traceAddress, 1))
                GuestGpuMemoryHook.Trace(traceAddress, 1,
                    $"device-address-mapping-check readable={_guestMemory.CanRead(traceAddress, 1)} backed={_guestBacking.IsBackedView(traceAddress)}");
            // The sweep finds CPU writes since the last one; every device-address draw repeating it
            // within a fraction of a millisecond finds nothing new, and the page-guard locks it takes
            // were a tenth of the render thread. SHARPEMU_BDA_SWEEP_US=0 sweeps for every draw.
            var sweepNow = System.Diagnostics.Stopwatch.GetTimestamp();
            if (BdaSweepIntervalTicks != 0 && _bdaSweepMappingKey == _bdaSpanMapping && sweepNow - _bdaSweepTimestamp < BdaSweepIntervalTicks &&
                _scheduler.CurrentTick == _bdaSweepTick)
            {
                return;
            }

            _bdaSweepMappingKey = _bdaSpanMapping;
            _bdaSweepTimestamp = sweepNow;
            _bdaSweepTick = _scheduler.CurrentTick;
            _bufferCache.PrepareBda(spans, _bdaSpanMapping);
        }

        private static readonly long BdaSweepIntervalTicks = long.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_BDA_SWEEP_US"), out var bdaSweepMicroseconds)
            ? bdaSweepMicroseconds * System.Diagnostics.Stopwatch.Frequency / 1_000_000
            : 500 * System.Diagnostics.Stopwatch.Frequency / 1_000_000;
        private ulong _bdaSweepMappingKey = ulong.MaxValue;
        private long _bdaSweepTimestamp;
        private ulong _bdaSweepTick;

        private List<GuestSpan>? _bdaSpans;
        private GuestGpuMemory? _bdaSpanMemory;
        private long _bdaSpanVersion = -1;
        private ulong _bdaSpanMapping;

        private List<GuestSpan> DeviceAddressSpans(GuestGpuMemory memory)
        {
            var version = memory.SpanVersion;
            if (_bdaSpans is { } cached && ReferenceEquals(_bdaSpanMemory, memory) && _bdaSpanVersion == version)
            {
                return cached;
            }

            var spans = new List<GuestSpan>();
            memory.ForEachSpan((address, size) => spans.Add(new GuestSpan(address, size)));
            _bdaSpans = spans;
            _bdaSpanMemory = memory;
            _bdaSpanVersion = version;
            _bdaSpanMapping = GuestBufferCache.MappingKey(spans);
            return spans;
        }

        public void BindResources(IPreparedBindings prepared)
        {
            if (!DbgPrepStats)
            {
                BindResourcesCore(prepared);
                return;
            }

            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            BindResourcesCore(prepared);
            var elapsed = System.Diagnostics.Stopwatch.GetTimestamp() - start;
            var program = ((PreparedStageBindings)prepared).Program;
            var key = (program.Hash, program.Stage);
            _dbgPrep.TryGetValue(key, out var entry);
            _dbgPrep[key] = (entry.Count, entry.Ticks, entry.Images, entry.BindTicks + elapsed);
        }

        private void BindResourcesCore(IPreparedBindings prepared)
        {
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DescriptorPreparation);
            var stage = (PreparedStageBindings)prepared;
            var dbgS0 = DbgSec(); // TEMP
            BindBuffers(stage);
            DbgSecEnd(5, ref dbgS0); // TEMP
            ObtainDeviceAddressRanges(stage);
            DbgSecEnd(6, ref dbgS0); // TEMP
            BindImages(stage);
            DbgSecEnd(7, ref dbgS0); // TEMP
            if (stage.Layout.Find(DescriptorBindingKind.RuntimeDescriptorTable) is not null)
            {
                BindRuntimeDescriptors(stage);
            }

            BindFlattenedResourceTable(stage);
            DbgSecEnd(8, ref dbgS0); // TEMP
            _preparedTextures.Add(stage.Textures);
        }

        // TEMP: section timers for the heaviest bindless pixel shader (SHARPEMU_DBG_PREP_STATS=1).
        private static readonly long[] _dbgSections = new long[9];
        private static long _dbgSectionOps, _dbgUniqueSum, _dbgUniqueOps, _dbgDistinctSum, _dbgResidentSum;
        private static long DbgSec() => DbgPrepStats ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        private static void DbgSecEnd(int index, ref long start)
        {
            if (!DbgPrepStats) return;
            var now = System.Diagnostics.Stopwatch.GetTimestamp();
            _dbgSections[index] += now - start;
            start = now;
        }

        // Only host movie planes own a staging buffer that must outlive the submission. Keeping
        // every binding until GPU completion promoted each draw's bindings out of gen0.
        private static TextureResource[] TexturesWithStaging(TextureResource[] textures)
        {
            var count = 0;
            foreach (var texture in textures)
            {
                if (texture.StagingBuffer.Handle != 0)
                {
                    count++;
                }
            }

            if (count == 0)
            {
                return [];
            }

            var retained = new TextureResource[count];
            count = 0;
            foreach (var texture in textures)
            {
                if (texture.StagingBuffer.Handle != 0)
                {
                    retained[count++] = texture;
                }
            }

            return retained;
        }

        // Textures bound for the draw being prepared; cleared when its preparation closes.
        private readonly List<TextureResource[]> _preparedTextures = new();

        bool IRenderHost.SamplesDepthAttachment(in DepthAttachmentState depth)
        {
            var image = _imageCache.GetImage(depth.Image);
            var attachmentView = depth.Target.Target.Request.View;
            foreach (var textures in _preparedTextures)
            {
                foreach (var texture in textures)
                {
                    if (!texture.IsHostMovie && ReferenceEquals(texture.CachedImage, image) && ViewsOverlap(texture.Request.View, attachmentView))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        private BufferView NullStorageBuffer() => new(_bufferCache.GetBuffer(GuestBufferCache.NullBufferId).Handle, 0, NullStorageBufferBytes);

        // A storage buffer view on the cache buffer, aligned down with the adjustment carried in the memory offsets.
        private BufferView BindStorageBuffer(
            in BufferDescriptorWords descriptor,
            BufferResource resource,
            ShaderProgramInfo program,
            int slot,
            ResourceSlotIdentifier bufferIdentifier,
            out uint memoryOffset)
        {
            memoryOffset = 0;
            var address = descriptor.Address;
            var requested = descriptor.Footprint() ?? throw SubmissionScheduler.Fatal($"A storage buffer descriptor footprint overflows: buffer={slot} hash=0x{program.Hash:X16}.");
            if (address == 0 || requested == 0)
            {
                return NullStorageBuffer();
            }

            var size = ClampMappedSize(address, requested);
            var alignment = _minStorageBufferOffsetAlignment;
            var maxRange = _deviceInfo.MaxStorageBufferRange;
            if (alignment == 0 || size > maxRange)
            {
                throw SubmissionScheduler.Fatal($"A storage buffer range or the device alignment is unsupported: buffer={slot} size=0x{size:X} alignment={alignment} hash=0x{program.Hash:X16}.");
            }

            DbgNoteBufferWriter(program, slot, address, size, resource.Written); // TEMP
            var (buffer, offset) = _bufferCache.ObtainBuffer(address, size, resource.Written, isTexelBuffer: resource.Formatted, bufferIdentifier);
            var alignedOffset = offset - offset % alignment;
            var adjustment = offset - alignedOffset;
            // A buffer read only through byte-assembled accesses may start mid-dword.
            if ((adjustment % sizeof(uint) != 0 && resource.DwordAddressed) || adjustment >= MaxMemoryOffsetAdjustment || size > maxRange - adjustment)
            {
                throw SubmissionScheduler.Fatal($"A storage buffer offset adjustment is unsupported: buffer={slot} adjustment={adjustment} hash=0x{program.Hash:X16}.");
            }

            memoryOffset = (uint)adjustment;

            if (resource.Formatted && resource.Written)
            {
                _imageCache.InvalidateMemoryFromGpu(address, size);
            }

            return new BufferView(buffer.Handle, alignedOffset, size + adjustment);
        }

        private BufferView UploadDwords(uint[] data, string label, ShaderProgramInfo program)
        {
            if (data.Length == 0)
            {
                throw SubmissionScheduler.Fatal($"The {label} upload is empty: hash=0x{program.Hash:X16}.");
            }

            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(data);
            var binding = UploadTransient(bytes, Math.Max(TransientDataAlignment, (uint)_minStorageBufferOffsetAlignment));
            return new BufferView(new VkBuffer(binding.Handle), binding.Offset, (ulong)bytes.Length);
        }

        private void BindBuffers(PreparedStageBindings prepared)
        {
            var program = prepared.Program;
            var layout = prepared.Layout;
            var info = prepared.Resources.Info;
            var snapshot = prepared.Stage.Resources;
            var shaderData = prepared.ShaderData;
            if (prepared.BufferSources.Length != info.Buffers.Count || shaderData.Length != layout.ShaderDataDwordCount)
            {
                throw SubmissionScheduler.Fatal($"The prepared bindings are stale: hash=0x{program.Hash:X16}.");
            }

            // Reset only packed buffer offsets; dispatch limits follow them in shader data.
            Array.Clear(shaderData, (int)layout.MemoryOffsetDword, (int)((layout.MemoryOffsetCount + 3) / 4));
            var views = prepared.Descriptors.Buffers;
            if (views.Length != info.Buffers.Count)
            {
                Scratch.Buffers.Return(views);
                views = Scratch.Buffers.Rent(info.Buffers.Count);
                prepared.Descriptors.Buffers = views;
            }
            for (var index = 0; index < views.Length; index++)
            {
                var (descriptor, bufferIdentifier) = prepared.BufferSources[index];
                views[index] = BindStorageBuffer(in descriptor, info.Buffers[index], program, index, bufferIdentifier, out var memoryOffset);
                var dword = layout.MemoryOffsetDword + (uint)index / 4;
                var shift = ((uint)index % 4) * 8;
                shaderData[dword] |= memoryOffset << (int)shift;
            }

            if (layout.UsesRuntimeBufferStrides)
            {
                Array.Clear(shaderData, (int)layout.BufferStrideDword, (int)layout.BufferStrideDwordCount);
                for (var index = 0; index < views.Length; index++)
                {
                    var stride = prepared.BufferSources[index].Descriptor.Stride;
                    shaderData[layout.BufferStrideDword + (uint)index / 2] |= stride << ((index % 2) * 16);
                }
            }

            prepared.Descriptors.Buffers = views;
            if (layout.Find(DescriptorBindingKind.ShaderData) is not null)
            {
                prepared.Descriptors.ShaderData = UploadDwords(shaderData, "shader data", program);
            }
        }

        private void BindFlattenedResourceTable(PreparedStageBindings prepared)
        {
            var layout = prepared.Layout;
            if (layout.Find(DescriptorBindingKind.FlattenedResourceTable) is null)
            {
                return;
            }

            var program = prepared.Program;
            var snapshot = prepared.Stage.Resources;
            if (!layout.UsesBindlessImages)
            {
                var patched = PatchTableOnCpu(snapshot, 0, snapshot.FlattenedResourceTable, out var deviceRuns);
                prepared.Descriptors.FlattenedTable = UploadDwords(patched, "flattened resource table", program);
                CopyTableRuns(prepared.Descriptors.FlattenedTable, deviceRuns);
                return;
            }

            var slotCount = (int)BindingLayout.ImageSlotTableDwordCount(prepared.Resources.Info);
            var table = new uint[checked(slotCount + snapshot.FlattenedResourceTable.Length)];
            var slot = 0;
            foreach (var binding in layout.Descriptors)
            {
                if (ImageDescriptorBinding.ResourceClass(binding.Kind) == ShaderCompiler.Resources.ImageResourceClass.None)
                {
                    continue;
                }

                var occurrences = new uint[prepared.Descriptors.Images.Length];
                foreach (var resource in binding.Resources)
                {
                    var texture = prepared.Descriptors.Images[resource];
                    var occurrence = occurrences[resource]++;
                    if (!texture.IsResident)
                    {
                        table[slot++] = 0;
                        continue;
                    }

                    var view = texture.MipViews.Length == 0
                        ? texture.View
                        : occurrence < (uint)texture.MipViews.Length ? texture.MipViews[occurrence] : default;
                    if (view.Handle == 0)
                    {
                        throw SubmissionScheduler.Fatal($"A bindless image has no view: image={resource} hash=0x{program.Hash:X16}.");
                    }

                    table[slot++] = _bindlessImageHeap!.GetOrCreateSlot(
                        binding.Kind, prepared.Stage.Resources.Images[(int)resource], view, texture.Layout);
                }
            }

            snapshot.FlattenedResourceTable.AsSpan().CopyTo(table.AsSpan(slotCount));
            PatchTableOnCpu(snapshot, slotCount, table, out var runs);
            prepared.Descriptors.FlattenedTable = UploadDwords(table, "bindless image slot table", program);
            CopyTableRuns(prepared.Descriptors.FlattenedTable, runs);
        }

        // Table words the materializer left to the device: runs whose bytes a GPU buffer still
        // owns are copied in queue order after upload; any other word is read on the CPU now.
        private uint[] PatchTableOnCpu(ResourceSnapshot snapshot, int baseDword, uint[] table,
            out List<(ulong Address, uint Dword, uint Count)> deviceRuns)
        {
            deviceRuns = [];
            var patches = snapshot.TablePatches;
            if (patches.Length == 0)
            {
                return table;
            }

            if (ReferenceEquals(table, snapshot.FlattenedResourceTable))
            {
                table = (uint[])table.Clone();
            }

            foreach (var patch in patches)
            {
                var dword = (uint)baseDword + patch.FlatOffset;
                if (_bufferCache.IsGpuOwnedWord(patch.Address) && !_imageCache.HasGpuModifiedImageBytes(patch.Address, sizeof(uint)))
                {
                    if (deviceRuns.Count != 0 && deviceRuns[^1] is var last &&
                        last.Address + last.Count * sizeof(uint) == patch.Address && last.Dword + last.Count == dword)
                    {
                        deviceRuns[^1] = last with { Count = last.Count + 1 };
                    }
                    else
                    {
                        deviceRuns.Add((patch.Address, dword, 1));
                    }

                    continue;
                }

                if (!TryReadGuestWord(patch.Address, out table[dword]))
                {
                    table[dword] = 0;
                }
            }

            return table;
        }

        private void CopyTableRuns(BufferView table, List<(ulong Address, uint Dword, uint Count)> runs)
        {
            foreach (var run in runs)
            {
                if (_bufferCache.TryCopyGpuOwned(run.Address, run.Count * sizeof(uint), table.Buffer.Handle,
                    table.Offset + run.Dword * sizeof(uint)))
                {
                    continue;
                }

                // PatchTableOnCpu checked ownership on this thread just before the upload.
                throw SubmissionScheduler.Fatal($"A GPU-owned table run could not be copied: address=0x{run.Address:X16} dwords={run.Count}.");
            }
        }

        // Device-address reads need persistent page-table entries before the shader runs.
        private void ObtainDeviceAddressRanges(PreparedStageBindings prepared)
        {
            var program = prepared.Program;
            foreach (var range in prepared.Stage.Resources.DeviceAddressRanges)
            {
                if (!range.Planned)
                {
                    if (!range.Written) continue;
                    throw SubmissionScheduler.Fatal($"A written device-address range cannot be planned: handle={range.Handle} hash=0x{program.Hash:X16}.");
                }

                if (range.Size == 0)
                {
                    continue;
                }

                if (range.Base >= PageOwnerTable.AddressSpaceSize || range.Size > PageOwnerTable.AddressSpaceSize - range.Base)
                {
                    throw SubmissionScheduler.Fatal($"A device-address range is outside the cache: handle={range.Handle} base=0x{range.Base:X16} size=0x{range.Size:X} hash=0x{program.Hash:X16}.");
                }

                // Unmapped read-only pointers resolve to zero through the page table.
                if (!range.Written && !_guestMemory.CanRead(range.Base, 1))
                {
                    continue;
                }

                var size = ClampMappedSize(range.Base, range.Size);

                if (range.Written)
                {
                    _ = _bufferCache.ObtainBuffer(range.Base, size, isWritten: true);
                }
                else
                {
                    // Stream buffers do not populate the device-address page table.
                    _ = _bufferCache.FindBuffer(range.Base, size);
                    _bufferCache.SynchronizeBuffersInRange(range.Base, size);
                }
            }
        }

        // Views for every image after the targets are bound; a stale image is found again first.
        private readonly Dictionary<(ResourceSlotIdentifier, ImageViewDescription, bool), ImageView> _acquiredViews = new();
        private readonly Dictionary<TextureResource, TextureResource> _replacedTextures = new();

        private TextureResource[] UniqueTextures(TextureResource[] images)
        {
            var mark = ++_textureMark;
            var unique = new List<TextureResource>(Math.Min(images.Length, 1024));
            foreach (var texture in images)
            {
                if (texture.PassMark != mark)
                {
                    texture.PassMark = mark;
                    unique.Add(texture);
                }
            }

            return [.. unique];
        }

        private void BindImages(PreparedStageBindings prepared)
        {
            var program = prepared.Program;
            var info = prepared.Resources.Info;
            var snapshot = prepared.Stage.Resources;
            var images = prepared.Descriptors.Images;
            _acquiredViews.Clear();
            _replacedTextures.Clear();
            var staleMark = ++_textureMark;
            var replacedAny = false;
            for (var index = 0; index < images.Length; index++)
            {
                var binding = images[index];
                if (binding.IsHostMovie || !binding.IsResident)
                {
                    continue;
                }

                if (binding.PassMark == staleMark)
                {
                    // A shared binding was already checked: follow its replacement.
                    if (replacedAny && _replacedTextures.TryGetValue(binding, out var replacement))
                    {
                        images[index] = replacement;
                    }

                    continue;
                }

                binding.PassMark = staleMark;
                if (IsStaleImage(binding.ImageIdentifier, out var stale))
                {
                    if (stale is not null)
                    {
                        stale.Binding = default;
                    }

                    var fresh = ResolveImageBinding(info.Images[index], snapshot.Images[index], program, index);
                    _replacedTextures[binding] = fresh;
                    images[index] = binding = fresh;
                    binding.PassMark = staleMark;
                    replacedAny = true;
                }
            }

            if (replacedAny)
            {
                _boundSets.Remove(prepared.Resources.Info);
                if (prepared.Descriptors.UniqueTextures is not null)
                {
                    prepared.Descriptors.UniqueTextures = UniqueTextures(images);
                }
            }

            var bindMark = ++_textureMark;
            for (var index = 0; index < images.Length; index++)
            {
                var binding = images[index];
                if (binding.IsHostMovie || !binding.IsResident || binding.PassMark == bindMark)
                {
                    continue;
                }

                binding.PassMark = bindMark;

                var resource = info.Images[index];
                var view = binding.Request.View;
                var image = _imageCache.GetImage(binding.ImageIdentifier);
                if (binding.IsStorage && image.Description.HasStencil && ViewFormatRules.IsStencilViewFormat(view.Format))
                {
                    image = AcquireStencilStorage(binding, image);
                    binding.Request = binding.Request with { View = view with { LevelCount = 1 } };
                    binding.View = image.GetOrCreateView(binding.Request.View with { Aspect = ImageAspectFlags.ColorBit });
                    binding.MipViews = [];
                }
                else if (resource.MipMode == ImageMipMode.DynamicStorage)
                {
                    if (resource.MipCount == 0 || resource.MipCount != view.LevelCount)
                    {
                        throw SubmissionScheduler.Fatal(
                            $"A storage image's mip count does not match its view: image={index} mips={resource.MipCount} levels={view.LevelCount} hash=0x{program.Hash:X16}.");
                    }

                    var mipViews = new ImageView[resource.MipCount];
                    for (var mip = 0u; mip < resource.MipCount; mip++)
                    {
                        var mipRequest = binding.Request with { View = view with { BaseLevel = view.BaseLevel + mip, LevelCount = 1 } };
                        mipViews[mip] = _imageCache.AcquireTextureView(binding.ImageIdentifier, mipRequest);
                    }

                    binding.MipViews = mipViews;
                    binding.View = mipViews[0];
                }
                else if (!binding.IsStorage && binding.View.Handle != 0 && binding.MipViews.Length == 0 &&
                         ReferenceEquals(binding.CachedImage, image) && binding.Image.Handle == image.Backing.Handle.Handle &&
                         image.Registered && !image.DepthOwner.IsValid && !image.Binding.NeedsRebind &&
                         !image.IsMaybeCpuDirty && !image.IsDefinitelyCpuDirty && !image.IsBufferModified &&
                         image.WatchBegin == image.Description.Data.Address && image.WatchEnd == image.Description.Data.End)
                {
                    // The view acquired for this binding earlier still stands: nothing to refresh or watch.
                }
                else
                {
                    if (binding.IsStorage)
                    {
                        binding.Request = binding.Request with { View = view with { LevelCount = 1 } };
                    }

                    // Bindless tables name the same image through many descriptors; acquire its view once.
                    var viewKey = (binding.ImageIdentifier, binding.Request.View, binding.IsStorage);
                    if (!_acquiredViews.TryGetValue(viewKey, out var acquiredView))
                    {
                        acquiredView = _imageCache.AcquireTextureView(binding.ImageIdentifier, binding.Request);
                        _acquiredViews[viewKey] = acquiredView;
                    }

                    binding.View = acquiredView;
                    binding.MipViews = [];
                }

                var descriptor = new TextureDescriptorWords(snapshot.Images[index]);
                // The culling pass reads the previous frame's visibility images after the game fast-cleared their DCC
                // metadata; applying that clear leaves it nothing to read (no static world). It reads their pixels.
                if (descriptor.MetadataCompress && !(VisibilityFeedback.KeepVisibilityImages && program.Hash == VisibilityFeedback.CullProgramHash))
                {
                    _imageCache.ApplyPendingDccClear(binding.ImageIdentifier, descriptor.MetadataAddress << 8);
                }

                if (DbgImageUseAddresses is { } dbgA && dbgA.Contains(image.Description.Data.Address) && Diagnostics.DbgTargetWatch.Addresses.Count != 0) // TEMP
                {
                    var dbgWords = snapshot.Images[index];
                    Diagnostics.DbgTargetWatch.Log($"tsharp {program.Hash:X} {index} {string.Join(",", dbgWords.Select(w => w.ToString("X8")))}", () =>
                        $"tsharp hash=0x{program.Hash:X16} slot={index} words=[{string.Join(",", dbgWords.Select(w => w.ToString("X8")))}] metaCompress={descriptor.MetadataCompress} meta=0x{descriptor.MetadataAddress << 8:X} cleared={_imageCache.IsMetadataCleared(descriptor.MetadataAddress << 8, 0, out var dbgFill)} fill=0x{dbgFill:X}");
                }
                if ((DbgImageUseAddresses is { } dbgAddresses && dbgAddresses.Contains(image.Description.Data.Address)) || DbgProgramImageHashes.Contains(program.Hash)) // TEMP
                {
                    var dbgKey = (program.Hash ^ image.Backing.Handle.Handle ^ (DbgProgramImageHashes.Contains(program.Hash) ? (_scheduler.CurrentTick / 2000) << 40 : 0), image.Description.Data.Address, binding.IsStorage, view.Format);
                    lock (DbgImageUseSeen)
                    {
                        if (DbgImageUseSeen.Add(dbgKey))
                            Console.Error.WriteLine($"[DBG][IMGUSE] words=[{string.Join(",", snapshot.Images[index].Select(w => w.ToString("X8")))}] size=0x{image.Description.Data.Size:X} hash=0x{program.Hash:X16} slot={index} addr=0x{image.Description.Data.Address:X} extent={image.Description.Extent.Width}x{image.Description.Extent.Height} storage={binding.IsStorage} written={resource.Written} view={view.Format} backing={image.Description.PixelFormat} mip={view.BaseLevel} image=0x{image.Backing.Handle.Handle:X}");
                    }
                }

                if (DbgOnePixel && image.Backing.Extent.Width == 1 && image.Backing.Extent.Height == 1) // TEMP
                {
                    var dbgOneKey = (program.Hash, image.Description.Data.Address, binding.IsStorage, view.Format);
                    lock (DbgImageUseSeen)
                    {
                        if (DbgImageUseSeen.Add(dbgOneKey))
                            Console.Error.WriteLine($"[DBG][ONEPX] hash=0x{program.Hash:X16} stage={prepared.Program.Stage} slot={index} addr=0x{image.Description.Data.Address:X} storage={binding.IsStorage} written={resource.Written} fmt={view.Format} words=[{string.Join(",", snapshot.Images[index].Select(w => w.ToString("X8")))}]");
                    }
                }

                if (binding.IsStorage && resource.Written && DbgGfxStores && prepared.Program.Stage != ShaderStageKind.Compute) // TEMP
                    Diagnostics.DbgTargetWatch.Log($"gfxstore {program.Hash:X} {image.Description.Data.Address:X}", () => $"gfxstore stage={prepared.Program.Stage} hash=0x{program.Hash:X16} slot={index} image=0x{image.Description.Data.Address:X} extent={image.Description.Extent.Width}x{image.Description.Extent.Height} fmt={view.Format}");
                if (binding.IsStorage && resource.Written && Diagnostics.DbgTargetWatch.PhysicalWatch.Length != 0) // TEMP
                    Diagnostics.DbgTargetWatch.CheckPhysical("storage", image.Description.Data.Address, image.Description.Data.Size, () => $"hash=0x{program.Hash:X16} slot={index} fmt={view.Format}");
                if (binding.IsStorage && resource.Written)
                    CaptureNoteStorage(image, binding.Request.View, image.Description.Data.Address, program.Hash, index); // TEMP

                image.Uses.Storage |= binding.IsStorage;
                image.Uses.Texture |= !binding.IsStorage;
                binding.CachedImage = image;
                binding.Image = image.Backing.Handle;
            }
        }

        private void DestroyStageBindings(PreparedStageBindings stage)
        {
            foreach (var texture in stage.Textures)
            {
                if (texture is { StagingBuffer.Handle: not 0 })
                {
                    RecycleHostBuffer(texture.StagingBuffer, texture.StagingMemory);
                }
            }
        }

        private void ReturnStageScratch(PreparedStageBindings stage)
        {
            // CommitBindings copies texture references into submission-owned
            // storage. These CPU descriptions are no longer referenced by GPU work.
            Scratch.Images.Return(stage.Descriptors.Images);
            Scratch.Samplers.Return(stage.Descriptors.Samplers);
            Scratch.Buffers.Return(stage.Descriptors.Buffers);
            Scratch.Sources.Return(stage.BufferSources);
            Scratch.ShaderData.Return(stage.ShaderData);
            stage.Descriptors.Images = [];
            stage.Descriptors.Samplers = [];
            stage.Descriptors.Buffers = [];
            stage.BufferSources = [];
            stage.ShaderData = [];
        }

        private static DescriptorImageInfo ImageInfo(TextureResource texture, uint element, ShaderProgramInfo program, int index)
        {
            var view = texture.MipViews.Length == 0
                ? (element == 0 ? texture.View : default)
                : (element < (uint)texture.MipViews.Length ? texture.MipViews[element] : default);
            if (view.Handle == 0 || texture.Layout == ImageLayout.Undefined)
            {
                throw SubmissionScheduler.Fatal($"An image binding has no view for its element: image={index} element={element} hash=0x{program.Hash:X16}.");
            }

            return new DescriptorImageInfo { ImageView = view, ImageLayout = texture.Layout };
        }

        // Joins every stage's writes, records the image work and binds the set by push or from the heap.
        public void CommitBindings(PipelineBindPoint bindPoint, in PipelineHandle pipeline, ReadOnlySpan<IPreparedBindings> stages)
        {
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DescriptorCommit);
            var preparation = RequirePreparation();
            var entry = RequirePipelineEntry(in pipeline);
            foreach (var prepared in stages)
            {
                var stage = (PreparedStageBindings)prepared;
                if (stage.Layout.Find(DescriptorBindingKind.RuntimeDescriptorTable) is not null)
                    CommitRuntimeDescriptorBuffers(stage);
            }
            var command = BeginBatchedGuestCommands();
            _commandBuffer = command;
            if (entry.UsesBindlessImages)
            {
                var globalSet = _bindlessImageHeap!.Set;
                _vk.CmdBindDescriptorSets(command, bindPoint, entry.Layout, 0, 1, &globalSet, 0, null);
            }
            var writeCount = 0;
            var bufferCount = 0;
            var imageCount = 0;
            var textureCount = 0;
            var maxStageImages = 0;
            foreach (var prepared in stages)
            {
                var stage = (PreparedStageBindings)prepared;
                var stageFlag = DescriptorWriter.ShaderStageFlag(StageOf(stage.Program));
                if ((bindPoint == PipelineBindPoint.Graphics && (stageFlag & (ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit)) == 0) ||
                    (bindPoint == PipelineBindPoint.Compute && stageFlag != ShaderStageFlags.ComputeBit))
                {
                    throw SubmissionScheduler.Fatal($"A stage does not belong to the bind point: stage={stage.Program.Stage} bindPoint={bindPoint}.");
                }

                foreach (var binding in stage.Layout.Descriptors)
                {
                    if (stage.Layout.UsesBindlessImages && ImageDescriptorBinding.ResourceClass(binding.Kind) != ShaderCompiler.Resources.ImageResourceClass.None)
                    {
                        continue;
                    }

                    writeCount++;
                    var count = (int)DescriptorWriter.DescriptorCount(binding);
                    if (ImageDescriptorBinding.ResourceClass(binding.Kind) != ShaderCompiler.Resources.ImageResourceClass.None || binding.Kind == DescriptorBindingKind.Samplers)
                    {
                        imageCount += count;
                    }
                    else
                    {
                        bufferCount += count;
                    }
                }

                textureCount += stage.Textures.Length;
                maxStageImages = Math.Max(maxStageImages, stage.Descriptors.Images.Length);
            }

            // All stages that read the same stencil bytes share the shader's working image.
            if (preparation.StencilStorageImages is { } stencilImages)
            {
                foreach (var prepared in stages)
                {
                    foreach (var texture in ((PreparedStageBindings)prepared).Textures)
                    {
                        if (texture.CachedImage is { } attachment && ViewFormatRules.IsStencilViewFormat(texture.Request.View.Format) &&
                            stencilImages.TryGetValue(attachment, out var storage))
                        {
                            texture.CachedImage = storage;
                            texture.Image = storage.Backing.Handle;
                            texture.View = storage.GetOrCreateView(texture.Request.View with { Aspect = ImageAspectFlags.ColorBit });
                            texture.MipViews = [];
                        }
                    }
                }
            }

            // Vulkan consumes these host-side descriptions during the call.
            // Only the resources themselves remain alive until GPU completion.
            // Bound stack use for unusually large descriptor layouts.
            const int StackDescriptorLimit = 128;
            Span<DescriptorBufferInfo> bufferInfos = bufferCount <= StackDescriptorLimit
                ? stackalloc DescriptorBufferInfo[bufferCount] : new DescriptorBufferInfo[bufferCount];
            Span<DescriptorImageInfo> imageInfos = imageCount <= StackDescriptorLimit
                ? stackalloc DescriptorImageInfo[imageCount] : new DescriptorImageInfo[imageCount];
            Span<WriteDescriptorSet> writes = writeCount <= StackDescriptorLimit
                ? stackalloc WriteDescriptorSet[writeCount] : new WriteDescriptorSet[writeCount];
            Span<uint> occurrenceScratch = maxStageImages <= StackDescriptorLimit
                ? stackalloc uint[maxStageImages] : new uint[maxStageImages];
            var pushData = stackalloc uint[(int)PushData.DwordCount];
            var hasPushData = false;
            var bufferIndex = 0;
            var imageIndex = 0;
            var writeIndex = 0;
            var uploadStage = bindPoint == PipelineBindPoint.Compute ? PipelineStageFlags.ComputeShaderBit : PipelineStageFlags.FragmentShaderBit;
            TextureResource[] textures = textureCount == 0 ? [] : new TextureResource[textureCount];
            var textureIndex = 0;
            fixed (DescriptorBufferInfo* bufferInfoPointer = bufferInfos)
            fixed (DescriptorImageInfo* imageInfoPointer = imageInfos)
            {
                foreach (var prepared in stages)
                {
                    var stage = (PreparedStageBindings)prepared;
                    var program = stage.Program;
                    var descriptors = stage.Descriptors;
                    var shaderStage = StageOf(program);
                    var stageFlag = DescriptorWriter.ShaderStageFlag(shaderStage);
                    if (descriptors.GlobalDataShare.Buffer.Handle != 0)
                    {
                        // The host and every earlier queue write to the data share complete before the shader reads it.
                        EndRendering();
                        command = BeginBatchedGuestCommands();
                        var barrier = GlobalDataShareBarrier.Make(descriptors.GlobalDataShare.Buffer);
                        VulkanSynchronization.PipelineBarrier(_vk,command, GlobalDataShareBarrier.SourceStages, DescriptorWriter.PipelineStageFlag(stageFlag), 0, 0, null, 1, &barrier, 0, null);
                    }

                    RecordStageTextureTransitions(stage.Textures);
                    foreach (var texture in stage.Textures)
                    {
                        if (texture.IsHostMovie && texture.NeedsUpload)
                        {
                            EndRendering();
                            break;
                        }
                    }

                    RecordHostMovieUploads(stage.Textures, uploadStage);
                    foreach (var texture in stage.Textures)
                    {
                        textures[textureIndex++] = texture;
                    }

                    var occurrences = occurrenceScratch[..descriptors.Images.Length];
                    occurrences.Clear();
                    foreach (var binding in stage.Layout.Descriptors)
                    {
                        if (stage.Layout.UsesBindlessImages && ImageDescriptorBinding.ResourceClass(binding.Kind) != ShaderCompiler.Resources.ImageResourceClass.None)
                        {
                            continue;
                        }

                        var write = new WriteDescriptorSet
                        {
                            SType = StructureType.WriteDescriptorSet,
                            DstBinding = BindingLayout.NativeBindingIndex(shaderStage, binding.Kind),
                            DescriptorType = DescriptorWriter.DescriptorType(binding.Kind),
                            DescriptorCount = DescriptorWriter.DescriptorCount(binding),
                        };
                        var bufferStart = bufferIndex;
                        var imageStart = imageIndex;
                        if (ImageDescriptorBinding.ResourceClass(binding.Kind) != ShaderCompiler.Resources.ImageResourceClass.None)
                        {
                            foreach (var resource in binding.Resources)
                            {
                                imageInfos[imageIndex++] = ImageInfo(descriptors.Images[(int)resource], occurrences[(int)resource]++, program, (int)resource);
                            }
                        }
                        else
                        {
                            switch (binding.Kind)
                            {
                                case DescriptorBindingKind.Buffers:
                                    foreach (var resource in binding.Resources)
                                    {
                                        var view = descriptors.Buffers[(int)resource];
                                        if (view.Buffer.Handle == 0)
                                        {
                                            throw SubmissionScheduler.Fatal($"A storage buffer binding has no buffer: buffer={resource} hash=0x{program.Hash:X16}.");
                                        }

                                        bufferInfos[bufferIndex++] = new DescriptorBufferInfo { Buffer = view.Buffer, Offset = view.Offset, Range = view.Range };
                                    }

                                    break;
                                case DescriptorBindingKind.DeviceAddressPageTable:
                                case DescriptorBindingKind.FaultBuffer:
                                {
                                    var shared = binding.Kind == DescriptorBindingKind.DeviceAddressPageTable ? _bufferCache.BdaPageTableBuffer : _bufferCache.FaultBuffer;
                                    bufferInfos[bufferIndex++] = new DescriptorBufferInfo { Buffer = shared.Handle, Offset = 0, Range = shared.Size };
                                    break;
                                }

                                case DescriptorBindingKind.FlattenedResourceTable:
                                case DescriptorBindingKind.ShaderData:
                                case DescriptorBindingKind.GlobalDataShare:
                                case DescriptorBindingKind.RuntimeDescriptorTable:
                                case DescriptorBindingKind.RuntimeDescriptorMisses:
                                {
                                    var view = binding.Kind switch
                                    {
                                        DescriptorBindingKind.FlattenedResourceTable => descriptors.FlattenedTable,
                                        DescriptorBindingKind.ShaderData => descriptors.ShaderData,
                                        DescriptorBindingKind.RuntimeDescriptorTable => descriptors.RuntimeTable,
                                        DescriptorBindingKind.RuntimeDescriptorMisses => descriptors.RuntimeMisses,
                                        _ => descriptors.GlobalDataShare,
                                    };
                                    if (view.Buffer.Handle == 0)
                                    {
                                        throw SubmissionScheduler.Fatal($"A shared buffer binding has no buffer: kind={binding.Kind} hash=0x{program.Hash:X16}.");
                                    }

                                    bufferInfos[bufferIndex++] = new DescriptorBufferInfo { Buffer = view.Buffer, Offset = view.Offset, Range = view.Range };
                                    break;
                                }

                                case DescriptorBindingKind.Samplers:
                                    foreach (var resource in binding.Resources)
                                    {
                                        var sampler = descriptors.Samplers[(int)resource];
                                        if (sampler.Handle == 0)
                                        {
                                            throw SubmissionScheduler.Fatal($"A sampler binding has no sampler: sampler={resource} hash=0x{program.Hash:X16}.");
                                        }

                                        imageInfos[imageIndex++] = new DescriptorImageInfo { Sampler = sampler, ImageLayout = ImageLayout.Undefined };
                                    }

                                    break;
                                default:
                                    throw SubmissionScheduler.Fatal($"The descriptor binding kind is invalid: kind={binding.Kind}.");
                            }
                        }

                        if (bufferIndex != bufferStart)
                        {
                            write.PBufferInfo = bufferInfoPointer + bufferStart;
                        }

                        if (imageIndex != imageStart)
                        {
                            write.PImageInfo = imageInfoPointer + imageStart;
                        }

                        writes[writeIndex++] = write;
                    }

                    for (var index = 0; index < descriptors.Images.Length && !stage.Layout.UsesBindlessImages; index++)
                    {
                        var expected = descriptors.Images[index].MipViews.Length == 0 ? 1u : (uint)descriptors.Images[index].MipViews.Length;
                        if (occurrences[index] != expected)
                        {
                            throw SubmissionScheduler.Fatal($"An image is bound a different number of times than its views: image={index} occurrences={occurrences[index]} views={expected} hash=0x{program.Hash:X16}.");
                        }
                    }

                    if (stage.ShaderData.Length != stage.Layout.ShaderDataDwordCount)
                    {
                        throw SubmissionScheduler.Fatal($"The shader data does not match the layout: dwords={stage.ShaderData.Length} layout={stage.Layout.ShaderDataDwordCount} hash=0x{program.Hash:X16}.");
                    }

                    if (stage.Layout.UsesPushData)
                    {
                        for (var index = 0; index < stage.ShaderData.Length; index++)
                        {
                            pushData[stage.Layout.PushDataStartDword + index] = stage.ShaderData[index];
                        }

                        hasPushData = true;
                    }
                }

                if (hasPushData)
                {
                    var pushStages = bindPoint == PipelineBindPoint.Graphics ? ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit : ShaderStageFlags.ComputeBit;
                    _vk.CmdPushConstants(command, entry.Layout, pushStages, 0, PushData.ByteSize, pushData);
                }

                if (writeIndex != 0)
                {
                    fixed (WriteDescriptorSet* writePointer = writes)
                    {
                        if (entry.UsesPushDescriptors)
                        {
                            _pushDescriptorApi.CmdPushDescriptorSet(command, bindPoint, entry.Layout, entry.UsesBindlessImages ? 1u : 0u, (uint)writeIndex, writePointer);
                            ShaderCacheCounters.CountPushSet();
                        }
                        else
                        {
                            var set = _descriptorHeap.Commit(entry.SetLayout, in entry.Demand);
                            for (var index = 0; index < writeIndex; index++)
                            {
                                writes[index].DstSet = set;
                            }

                            _vk.UpdateDescriptorSets(_device, (uint)writeIndex, writePointer, 0, null);
                            var setIndex = entry.UsesBindlessImages ? 1u : 0u;
                            _vk.CmdBindDescriptorSets(command, bindPoint, entry.Layout, setIndex, 1, &set, 0, null);
                            ShaderCacheCounters.CountHeapSet();
                        }
                    }
                }
            }

            _batchResources.Add(new SubmissionUploadResources
            {
                DebugName = bindPoint == PipelineBindPoint.Compute ? "SharpEmu dispatch" : "SharpEmu draw",
                Textures = TexturesWithStaging(textures),
                FeedbackSnapshots = preparation.FeedbackSnapshots?.ToArray() ?? [],
                OverflowBuffers = preparation.OverflowBuffers.Count == 0 ? null : preparation.OverflowBuffers.ToArray(),
            });
            preparation.OverflowBuffers.Clear();
            preparation.Committed = true;
            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write($"Bindings committed bindPoint={bindPoint} pipeline={pipeline.Pipeline} writes={writeIndex} push={entry.UsesPushDescriptors} pushData={hasPushData}");
            }
        }
    }
}
