// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

using System.Buffers;
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
            public readonly RenderScratchPool<byte> SamplerClasses = new();
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
            ShaderStageKind.Mesh => ShaderStage.Mesh,
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
            public uint[]? Words;
            public ulong Generation;
            public TextureRequestResolution Resolution;
            public TextureRequestResolution OriginalResolution;
            public ResourceSlotIdentifier Found;
            public uint MetadataBaseLayer;
            public bool CompletesLookup;

            public readonly bool Matches(uint[] words, ulong generation) =>
                Words is not null && Generation == generation && words.AsSpan().SequenceEqual(Words);
        }

        private Dictionary<ShaderProgramInfo, ImageBindingMemo[]>? _imageBindingMemos;
        private ShaderProgramInfo? _lastMemoProgram;
        private ImageBindingMemo[] _lastMemos = [];

        // A draw's images resolve one after another for one program, so the last program's memos
        // are kept at hand.
        private ref ImageBindingMemo ImageBindingMemoOf(ShaderProgramInfo program, int index)
        {
            if (!ReferenceEquals(program, _lastMemoProgram) || index >= _lastMemos.Length)
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

        private readonly HashSet<string> _imageLookupTraceEntries = new(StringComparer.Ordinal);

        private void TraceImageLookup(ImageResource image, uint[] words, ShaderProgramInfo program, int index, in ImageRequest request)
        {
            if (!ImageClearTrace.Enabled ||
                !ImageTraceRange.Overlaps(request.Description.Data.Address, 1) ||
                _imageLookupTraceEntries.Count >= 256)
                return;

            var description = request.Description;
            var message = $"ImageLookup shader=0x{program.Hash:X16} stage={program.Stage} image={index} indirectRoot={image.IndirectRoot} " +
                $"address=0x{description.Data.Address:X16} size=0x{description.Data.Size:X} type={description.Type} tile={description.TileMode} " +
                $"extent={description.Extent.Width}x{description.Extent.Height}x{description.Extent.Depth} levels={description.Resources.Levels} layers={description.Resources.Layers} format={description.PixelFormat} " +
                $"view={request.View} words={string.Join(',', words.Select(word => word.ToString("X8")))}";
            if (!_imageLookupTraceEntries.Add(message)) return;
            Console.Error.WriteLine($"[GPU][TRACE] {message}");
            if (_imageLookupTraceEntries.Count == 256)
                Console.Error.WriteLine("[GPU][TRACE] ImageLookup limit=256. Further entries are omitted.");
        }

        // Render-state discovery for one shader image; the view is acquired later with the draw.
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private TextureResource ResolveImageBinding(ImageResource image, uint[] words, ShaderProgramInfo program, int index)
        {
            if (words.Length < 4)
            {
                throw SubmissionScheduler.Fatal($"An image descriptor is too short: image={index} words={words.Length} hash=0x{program.Hash:X16}.");
            }

            var storage = image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage;
            _ = BeginBatchedGuestCommands();
            var originalResolution = memo.Matches(words, _imageCache.LookupGeneration)
                ? memo.OriginalResolution
                : ImageRequestBuilders.Texture(words, ShapeOf(image), program.Hash, index);
            TraceImageLookup(image, words, program, index, originalResolution.Request);
            CompleteOverlappingPublications(originalResolution.Request.Description.Data.Address,
                originalResolution.Request.Description.Data.Size, includeLabels: false);
            _imageCache.SynchronizeColorMetadata(originalResolution.Request);
            _ = BeginBatchedGuestCommands();
            TextureRequestResolution resolution;
            ResourceSlotIdentifier imageIdentifier;
            if (memo.Matches(words, _imageCache.LookupGeneration) && _imageCache.TryTouchRemembered(memo.Found))
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

            var storage = image.ResourceClass == ShaderCompiler.Resources.ImageResourceClass.Storage;
            var shape = ShapeOf(image);
            _ = BeginBatchedGuestCommands();
            // A descriptor resolves to the same image until an image enters or leaves the cache,
            // so the request building and overlap lookup run once per descriptor, not per draw.
            var structureVersion = _imageCache.StructureVersion;
            if (_textureBindings.Count > MaxTextureBindings)
            {
                resolution = originalResolution;
                var request = resolution.Request;
                var metadataBaseLayer = request.View.BaseLayer;
                imageIdentifier = _imageCache.FindImage(ref request, resolution.ExactFormat);
                resolution = resolution with { Request = request };
                memo = !ImageDescription.IsEmptyRange(request.Description.Data)
                    ? new ImageBindingMemo
                    {
                        Words = (uint[])words.Clone(),
                        Generation = _imageCache.LookupGeneration,
                        Resolution = resolution,
                        OriginalResolution = originalResolution,
                        Found = imageIdentifier,
                        MetadataBaseLayer = metadataBaseLayer,
                        CompletesLookup = request.Description.Metadata.Kind == MetadataKind.Dcc ||
                            _imageCache.GetImage(imageIdentifier).Description.DccSliceSize != 0,
                    }
                    : default;
            }

            _imageCache.TraceTextureMetadata(resolution.Request);
            if (Gpu.Images.ImageClearTrace.Enabled)
            {
                Gpu.Images.ImageTraceRange.NoteFollowedImage(program.Hash, index,
                    resolution.Request.Description.Data.Address, resolution.Request.Description.Data.Size);
                _imageCache.TraceTextureBinding(program.Hash, index, words, resolution.Request);
            }
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("image-binding", $"stage={program.Stage} hash=0x{program.Hash:X16} slot={index} storage={storage} address=0x{resolution.Request.Description.Data.Address:X16} size=0x{resolution.Request.Description.Data.Size:X} view={resolution.Request.View} words={string.Join(",", words.Select(value => value.ToString("X8")))}");
            imageIdentifier = ImageRequestBuilders.ValidateTextureOwner(_imageCache, imageIdentifier, resolution);
            _imageCache.ApplyNativeColorClear(imageIdentifier, resolution.Request);
            if (Gpu.Images.ImageClearTrace.Enabled)
                _imageCache.TraceVolumeClear("resolved-shader-image", resolution.Request, _imageCache.GetImage(imageIdentifier),
                    $"stage={program.Stage} hash=0x{program.Hash:X16} slot={index}");
            BindImage(imageIdentifier, storage);
            var descriptor = new TextureDescriptorWords(words);
            if (ShouldTraceTextureBindings() &&
                (!_traceVolumeTextureBindingsOnly || image.Dimension == ImageDimension.Dim3D))
            {
                var cached = _imageCache.GetImage(imageIdentifier);
                var description = cached.Description;
                var line =
                    $"TextureBinding stage={program.Stage} hash=0x{program.Hash:X16} index={index} " +
                    $"address=0x{new TextureDescriptorWords(words).BaseAddress:X16} " +
                    $"descriptor={descriptor.BaseAddress:X16} size={descriptor.Width + 1}x{descriptor.Height + 1} " +
                    $"format={(uint)descriptor.Format} tile={(uint)descriptor.TileMode} " +
                    $"image=0x{description.Data.Address:X16} size=0x{description.Data.Size:X} " +
                    $"extent={description.Extent.Width}x{description.Extent.Height} pitch={description.Pitch} " +
                    $"guestFormat={(uint)description.GuestFormat} imageTile={(uint)description.TileMode} " +
                    $"backing={cached.Backing.Extent.Width}x{cached.Backing.Extent.Height} format={cached.Backing.Format} " +
                    $"depth={descriptor.Depth + 1} type={descriptor.Type} dcc={descriptor.MetadataCompress} " +
                    $"metadata=0x{descriptor.MetadataAddress << 8:X} metaKind={description.Metadata.Kind} " +
                    $"words={string.Join(',', words.Select(static word => word.ToString("X8")))}";
                if (!_traceVolumeTextureBindingsOnly || _tracedTextureBindings.TryAdd(line, 0))
                {
                    Console.Error.WriteLine(line);
                }
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

        // Binding descriptions are large (an image request carries its whole mip layout) and one is made
        // per shader image per draw, so they are reused once the submission that read them has retired.
        // Only the render thread rents and returns them.
        // Sized for the bindings of every draw still in flight (a few frames of a thousand draws each);
        // a smaller cap drops returns during busy scenes and the pool allocates again.
        private const int MaxPooledTextureResources = 32768;
        private Stack<TextureResource>? _texturePool;

        private TextureResource RentTextureResource()
        {
            if (_texturePool is not null && _texturePool.TryPop(out var texture))
            {
                texture.InPool = false;
                return texture;
            }

            return new TextureResource { Pooled = true };
        }

        private void ReturnTextureResource(TextureResource texture)
        {
            _texturePool ??= new Stack<TextureResource>();
            if (!texture.Pooled || texture.InPool || _texturePool.Count >= MaxPooledTextureResources)
            {
                return;
            }

            texture.Reset();
            texture.InPool = true;
            _texturePool.Push(texture);
        }

        // A float view overrides integer views. Unpaired and movie-only samplers stay unclassified.
        private static void ClassifySamplerViews(ShaderResourceInfo info, TextureResource[] images, byte[] classes)
        {
            Array.Clear(classes);
            if (classes.Length == 0) return;
            foreach (var pair in info.SampledPairs)
            {
                if (pair.Sampler >= info.Samplers.Count || pair.Image >= images.Length || images[pair.Image].IsHostMovie)
                {
                    continue;
                }

                classes[pair.Sampler] |= ViewFormatRules.IsIntegerFormat(images[pair.Image].Request.View.Format)
                    ? (byte)1 : (byte)2;
            }
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

        public IPreparedBindings PrepareBindings(ShaderStageResources stage)
        {
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("stage-binding", $"stage={stage.Program?.Stage} hash=0x{stage.Program?.Hash:X16} shaderBase=0x{stage.ShaderBase:X16} userData={string.Join(",", stage.Resources.UserData.Select(value => value.ToString("X8")))}");
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DescriptorPreparation);
            var preparation = RequirePreparation();
            var program = RequireProgram(stage);
            var resources = program.Resources!;
            var layout = program.Bindings!;
            var snapshot = stage.Resources;
            var info = resources.Info;
            using var stallProfile = DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.Prepare, program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count);
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
            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.ResolveImages, program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
            for (var index = 0; index < info.Images.Count; index++)
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
                }
                else
                {
                    descriptors.UniqueTextures = boundSet.Unique ??= UniqueTextures(boundSet.Images);
                }

                foreach (var texture in descriptors.UniqueTextures)
                {
                    BindImage(texture.ImageIdentifier, texture.IsStorage);
                }
            }
            else
            {
                for (var index = 0; index < info.Images.Count; index++)
                {
                    descriptors.Images[index] = resolveOne(index);
                }
            }

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
            }


            _prepareMemoActive = false;
            if (layout.Find(DescriptorBindingKind.RuntimeDescriptorTable) is not null)
            {
                PrepareRuntimeDescriptors(prepared);
            }

            descriptors.Samplers = Scratch.Samplers.Rent(info.Samplers.Count);
            var samplerClasses = Scratch.SamplerClasses.Rent(info.Samplers.Count);
            try
            {
                using var samplerProfile = DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.ResolveSamplers, program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count);
                ClassifySamplerViews(info, descriptors.Images, samplerClasses);
                for (var index = 0; index < info.Samplers.Count; index++)
                {
                    descriptors.Samplers[index] = ResolveSampler(info.Samplers[index], snapshot.Samplers[index], program, index, stage,
                        samplerClasses[index] == 1);
                }
            }
            finally
            {
                Scratch.SamplerClasses.Return(samplerClasses);
            }

            var shaderData = Scratch.ShaderData.Rent(checked((int)layout.ShaderDataDwordCount));
            prepared.ShaderData = shaderData;
            var userData = stage.Resources.UserData;
            for (var index = 0; index < layout.UserDataRegisters.Count; index++)
            {
                var register = layout.UserDataRegisters[index];
                var userIndex = (int)(register - program.UserDataBase);
                if (register < program.UserDataBase || userIndex >= userData.Length)
                {
                    throw SubmissionScheduler.Fatal($"A user register is outside the draw's user data: register={register} base={program.UserDataBase} count={userData.Length} hash=0x{program.Hash:X16}.");
                }

                shaderData[index] = userData[userIndex];
            }

            if (layout.UsesShaderBase)
            {
                shaderData[layout.ShaderBaseDword] = (uint)stage.ShaderBase;
                shaderData[layout.ShaderBaseDword + 1] = (uint)(stage.ShaderBase >> 32);
            }

            stage.WriteDispatchThreadLimits(shaderData);
            stage.WriteMeshDrawParameters(shaderData);
            prepared.ShaderData = shaderData;
            if (layout.Find(DescriptorBindingKind.GlobalDataShare) is not null)
            {
                descriptors.GlobalDataShare = new BufferView(_bufferCache.GdsBuffer.Handle, 0, Vk.WholeSize);
            }

            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.FindAddressBuffers, program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
                FindDeviceAddressBuffers(prepared);
            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.FindBuffers, program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
                FindBuffers(prepared);
            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.ValidateImages, program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
                ValidateDrawImageTypes(prepared);
            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write(
                    $"Bindings prepared stage={program.Stage} hash=0x{program.Hash:X16} buffers={info.Buffers.Count} images={info.Images.Count} " +
                    $"samplers={info.Samplers.Count} userData={snapshot.UserData.Length} flattened={snapshot.FlattenedResourceTable.Length} shaderData={shaderData.Length}");
            }

            return prepared;
        }

        public void UpdateTessellationData(IPreparedBindings prepared, uint[] data)
        {
            var stage = (PreparedStageBindings)prepared;
            (stage.Stage with { TessellationData = data }).WriteTessellationData(stage.ShaderData);
        }

        // Reject incompatible draw views before buffer writes or image transitions are recorded.
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

        private void FindDeviceAddressBuffers(PreparedStageBindings prepared)
        {
            foreach (var range in prepared.Stage.Resources.DeviceAddressRanges)
            {
                if (!range.Planned || range.Size == 0 ||
                    range.Base >= PageOwnerTable.AddressSpaceSize || range.Size > PageOwnerTable.AddressSpaceSize - range.Base ||
                    (!range.Written && !_guestMemory.CanRead(range.Base, 1)))
                {
                    continue;
                }

                _ = _bufferCache.FindBuffer(range.Base, ClampMappedSize(range.Base, range.Size));
            }
        }

        private static int _unresidentPrtBufferLines;

        private bool IsUnresidentPrtPage(ulong address)
        {
            if (_guestMemory.CanRead(address, 1) || !Kernel.KernelRuntimeCompatExports.ContainsPrtRange(address, 1))
            {
                return false;
            }

            if (Interlocked.Exchange(ref _unresidentPrtBufferLines, 1) == 0)
            {
                Console.Error.WriteLine($"[GPU][INFO] A buffer starts in an unresident PRT page and binds as a null buffer: address=0x{address:X16}.");
            }

            return true;
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
                if (descriptor.Address == 0 || requested == 0 || IsUnresidentPrtPage(descriptor.Address))
                {
                    sources[index] = (descriptor, default);
                    continue;
                }

                var size = ClampMappedSize(descriptor.Address, requested, prepared, index);
                var resource = prepared.Resources.Info.Buffers[index];
                if (resource.Formatted && resource.Written)
                {
                    // Image preservation can grow a buffer beyond its descriptor's range.
                    // All stages prepare here before any stage captures buffer handles.
                    _imageCache.InvalidateMemoryFromGpu(descriptor.Address, size);
                }
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
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.DescriptorPreparation);
            var stage = (PreparedStageBindings)prepared;
            var info = stage.Resources.Info;
            using var stallProfile = DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.Bind, stage.Program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count);
            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.BindBuffers, stage.Program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
                BindBuffers(stage);
            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.ObtainAddressRanges, stage.Program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
                ObtainDeviceAddressRanges(stage);
            using (DescriptorStallProfile.Measure(DescriptorStallProfile.Operation.BindImages, stage.Program.Hash, info.Buffers.Count, info.Images.Count, info.Samplers.Count))
                BindImages(stage);
            _preparedTextures.Add(stage.Textures);
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

        ImageLayout? IRenderHost.DepthAttachmentLayout(in DepthAttachmentState depth)
        {
            var view = depth.Target.Target.Request.View;
            return _imageCache.GetImage(depth.Image).UniformLayout(new SubresourceRange(view.BaseLevel, view.LevelCount, view.BaseLayer, view.LayerCount));
        }

        private BufferView NullStorageBuffer() => new(_bufferCache.GetBuffer(GuestBufferCache.NullBufferId).Handle, 0, NullStorageBufferBytes);

        // A storage buffer view on the cache buffer, aligned down with the adjustment carried in the memory offsets.
        private BufferView BindStorageBuffer(
            in BufferDescriptorWords descriptor,
            BufferResource resource,
            ShaderProgramInfo program,
            int slot,
            ResourceSlotIdentifier bufferIdentifier,
            out uint memoryOffset,
            out uint tailPadding)
        {
            memoryOffset = 0;
            tailPadding = 0;
            var address = descriptor.Address;
            var requested = descriptor.Footprint() ?? throw SubmissionScheduler.Fatal($"A storage buffer descriptor footprint overflows: buffer={slot} hash=0x{program.Hash:X16}.");
            if (address == 0 || requested == 0 || IsUnresidentPrtPage(address))
            {
                return NullStorageBuffer();
            }

            var size = ClampMappedSize(address, requested);
            if (MeshDrawTrace.Active) MeshDrawTrace.Write("buffer-binding", $"stage={program.Stage} hash=0x{program.Hash:X16} slot={slot} address=0x{address:X16} requested=0x{requested:X} size=0x{size:X} written={resource.Written} formatted={resource.Formatted}");
            var alignment = _minStorageBufferOffsetAlignment;
            var maxRange = _deviceInfo.MaxStorageBufferRange;
            if (alignment == 0 || size > maxRange)
            {
                throw SubmissionScheduler.Fatal($"A storage buffer range or the device alignment is unsupported: buffer={slot} size=0x{size:X} alignment={alignment} hash=0x{program.Hash:X16}.");
            }

            var (buffer, offset) = _bufferCache.ObtainBuffer(address, size, resource.Written, isTexelBuffer: resource.Formatted, bufferIdentifier, program.Hash);
            TraceBufferParameter(program.Hash, slot, address, size, buffer.Handle.Handle, offset, resource.Written);
            var alignedOffset = offset - offset % alignment;
            var adjustment = offset - alignedOffset;
            // A buffer read only through byte-assembled accesses may start mid-dword.
            if ((adjustment % sizeof(uint) != 0 && resource.DwordAddressed) || adjustment >= MaxMemoryOffsetAdjustment || size > maxRange - adjustment)
            {
                throw SubmissionScheduler.Fatal($"A storage buffer offset adjustment is unsupported: buffer={slot} adjustment={adjustment} hash=0x{program.Hash:X16}.");
            }

            memoryOffset = (uint)adjustment;
            var range = size + adjustment;
            // The shader accesses uint[], and Vulkan's runtime array length truncates a partial
            // final word. The hardware still reads and writes a word that starts inside the
            // range, so the view is padded to whole words; the byte limit for byte and
            // formatted elements travels in the unused high bits of the runtime stride.
            if (program.Bindings!.UsesRuntimeBufferStrides)
            {
                tailPadding = (uint)((0UL - range) & 3);
                // Cache buffers span whole pages, so the padded word stays inside the VkBuffer.
                if (range > maxRange - tailPadding || alignedOffset + range + tailPadding > buffer.Size)
                {
                    throw SubmissionScheduler.Fatal($"A padded storage buffer exceeds the device limit or its cache buffer: buffer={slot} range=0x{range:X} offset=0x{alignedOffset:X} cache_size=0x{buffer.Size:X}.");
                }
                range += tailPadding;
            }
            else if (resource.Written)
            {
                _imageCache.InvalidateMemoryCopiesFromGpu(address, size);
            }

            if (resource.Written && !resource.Formatted)
            {
                _imageCache.InvalidateMemoryCopiesFromGpu(address, size);
            }

            return new BufferView(buffer.Handle, alignedOffset, range);
        }

        private BufferView UploadDwords(uint[] data, string label, ShaderProgramInfo program) =>
            UploadDwords(data.AsSpan(), label, program);

        private BufferView UploadDwords(ReadOnlySpan<uint> data, string label, ShaderProgramInfo program)
        {
            if (data.Length == 0)
            {
                throw SubmissionScheduler.Fatal($"The {label} upload is empty: hash=0x{program.Hash:X16}.");
            }

            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data);
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
            if (layout.UsesRuntimeBufferStrides)
            {
                Array.Clear(shaderData, (int)layout.BufferStrideDword, (int)layout.BufferStrideDwordCount);
            }
            if (views.Length != info.Buffers.Count)
            {
                Scratch.Buffers.Return(views);
                views = Scratch.Buffers.Rent(info.Buffers.Count);
                prepared.Descriptors.Buffers = views;
            }
            for (var index = 0; index < views.Length; index++)
            {
                var (descriptor, bufferIdentifier) = prepared.BufferSources[index];
                views[index] = BindStorageBuffer(in descriptor, info.Buffers[index], program, index, bufferIdentifier, out var memoryOffset, out var tailPadding);
                var dword = layout.MemoryOffsetDword + (uint)index / 4;
                var shift = ((uint)index % 4) * 8;
                shaderData[dword] |= memoryOffset << (int)shift;
                if (layout.UsesRuntimeBufferStrides)
                {
                    var stride = descriptor.Stride | (tailPadding << 14);
                    shaderData[layout.BufferStrideDword + (uint)index / 2] |= stride << ((index % 2) * 16);
                }
            }

            prepared.Descriptors.Buffers = views;
            if (layout.MipStatisticsCount != 0)
            {
                LodStatistics().EnsureInitialized();
                WriteMipStatisticsFields(shaderData, layout, info, snapshot.Images);
            }

            if (layout.Find(DescriptorBindingKind.ShaderData) is not null)
            {
                prepared.Descriptors.ShaderData = UploadDwords(shaderData, "shader data", program);
            }
        }

        private static readonly Dictionary<ulong, (long Calls, long Dwords, long Patches, int Max)> _dbgTable = new(); // TEMP
        private static long _dbgTableNext; // TEMP
        private static readonly bool _dbgTableStats = Environment.GetEnvironmentVariable("SHARPEMU_DBG_TABLESTAT2") == "1"; // TEMP
        private static void DbgTableStat(ulong hash, int dwords, int patches) // TEMP
        {
            if (!_dbgTableStats) return;
            lock (_dbgTable)
            {
                _dbgTable.TryGetValue(hash, out var e);
                _dbgTable[hash] = (e.Calls + 1, e.Dwords + dwords, e.Patches + patches, Math.Max(e.Max, dwords));
                var now = System.Diagnostics.Stopwatch.GetTimestamp();
                if (now < _dbgTableNext) return;
                _dbgTableNext = now + 15 * System.Diagnostics.Stopwatch.Frequency;
                foreach (var pair in _dbgTable.OrderByDescending(p => p.Value.Dwords).Take(12))
                    Console.Error.WriteLine($"[DBG][TABLE2] hash=0x{pair.Key:X16} calls={pair.Value.Calls} avgDw={pair.Value.Dwords / pair.Value.Calls} maxDw={pair.Value.Max} avgPatches={pair.Value.Patches / pair.Value.Calls} totalMB={pair.Value.Dwords * 4 / 1048576}");
                _dbgTable.Clear();
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
            DbgTableStat(program.Hash, snapshot.FlattenedResourceTable.Length, snapshot.TablePatches.Length); // TEMP
            if (!layout.UsesBindlessImages)
            {
                if (snapshot.TablePatches.Length == 0)
                {
                    prepared.Descriptors.FlattenedTable = UploadDwords(snapshot.FlattenedResourceTable, "flattened resource table", program);
                    return;
                }

                var flat = ArrayPool<uint>.Shared.Rent(snapshot.FlattenedResourceTable.Length);
                try
                {
                    snapshot.FlattenedResourceTable.AsSpan().CopyTo(flat);
                    PatchTableOnCpu(snapshot, 0, flat, out var deviceRuns);
                    prepared.Descriptors.FlattenedTable = UploadDwords(flat.AsSpan(0, snapshot.FlattenedResourceTable.Length), "flattened resource table", program);
                    CopyTableRuns(prepared.Descriptors.FlattenedTable, deviceRuns);
                }
                finally
                {
                    ArrayPool<uint>.Shared.Return(flat);
                }

                return;
            }

            // The slot table and the flattened table go to the device in one transient upload: a pooled array
            // holds them, because a table per draw is large (one slot per image the shader declares).
            var slotCount = (int)BindingLayout.ImageSlotTableDwordCount(prepared.Resources.Info);
            var tableLength = checked(slotCount + snapshot.FlattenedResourceTable.Length);
            var table = ArrayPool<uint>.Shared.Rent(tableLength);
            var imageCount = prepared.Descriptors.Images.Length;
            var occurrences = ArrayPool<uint>.Shared.Rent(imageCount);
            try
            {
                Array.Clear(table, 0, slotCount);
                Array.Clear(occurrences, 0, imageCount);
                var slot = 0;
                foreach (var binding in layout.Descriptors)
                {
                    if (ImageDescriptorBinding.ResourceClass(binding.Kind) == ShaderCompiler.Resources.ImageResourceClass.None)
                    {
                        continue;
                    }

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

                    // The occurrence count restarts for every binding.
                    foreach (var resource in binding.Resources)
                    {
                        occurrences[resource] = 0;
                    }
                }

                snapshot.FlattenedResourceTable.AsSpan().CopyTo(table.AsSpan(slotCount));
                PatchTableOnCpu(snapshot, slotCount, table, out var runs);
                prepared.Descriptors.FlattenedTable = UploadDwords(table.AsSpan(0, tableLength), "bindless image slot table", program);
                CopyTableRuns(prepared.Descriptors.FlattenedTable, runs);
            }
            finally
            {
                ArrayPool<uint>.Shared.Return(table);
                ArrayPool<uint>.Shared.Return(occurrences);
            }
        }

        // Table words the materializer left to the device: runs whose bytes a GPU buffer still
        // owns are copied in queue order after upload; any other word is read on the CPU now.
        private void PatchTableOnCpu(ResourceSnapshot snapshot, int baseDword, uint[] table,
            out List<(ulong Address, uint Dword, uint Count)> deviceRuns)
        {
            deviceRuns = [];
            var patches = snapshot.TablePatches;
            if (patches.Length == 0)
            {
                return;
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
                    _ = _bufferCache.ObtainBuffer(range.Base, size, isWritten: true, traceShaderHash: program.Hash);
                    _imageCache.InvalidateMemoryCopiesFromGpu(range.Base, size);
                }
                else
                {
                    // Stream buffers do not populate the device-address page table.
                    _ = _bufferCache.FindBuffer(range.Base, size);
                    _bufferCache.SynchronizeDeviceAddressRange(range.Base, size);
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
                if (binding.IsStorage && HizPolicy.ShouldPrefill(program.Hash, index) && image.Backing.Format == Format.R32Sfloat)
                {
                    _imageCache.FillImage(binding.ImageIdentifier, 1e7f);
                }

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

                if (binding.IsStorage) CaptureNoteStorage(image, binding.Request.View, image.Description.Data.Address, program.Hash, index); // TEMP
                var descriptor = new TextureDescriptorWords(snapshot.Images[index]);
                CaptureNoteTexture(program, index, binding.ImageIdentifier, binding.Request, descriptor.BaseAddress, snapshot.Images[index]); // TEMP
                // The culling pass reads the previous frame's visibility images after the game fast-cleared their DCC
                // metadata; applying that clear leaves it nothing to read (no static world). It reads their pixels.
                if (descriptor.MetadataCompress && !(VisibilityFeedback.KeepVisibilityImages && program.Hash == VisibilityFeedback.CullProgramHash))
                {
                    _imageCache.ApplyPendingDccClear(binding.ImageIdentifier, descriptor.MetadataAddress << 8);
                }

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
            var command = BeginBatchedGuestStateCommands();
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
                if ((bindPoint == PipelineBindPoint.Graphics && (stageFlag & (ShaderStageFlags.VertexBit | ShaderStageFlags.MeshBitExt | ShaderStageFlags.FragmentBit)) == 0) ||
                    (bindPoint == PipelineBindPoint.Compute && stageFlag != ShaderStageFlags.ComputeBit))
                {
                    throw SubmissionScheduler.Fatal($"A stage does not belong to the bind point: stage={stage.Program.Stage} bindPoint={bindPoint}.");
                }

                // Indexed loops over the layout's lists: a foreach on IReadOnlyList allocates an enumerator per draw.
                var countedBindings = stage.Layout.Descriptors;
                for (var bindingIndex = 0; bindingIndex < countedBindings.Count; bindingIndex++)
                {
                    var binding = countedBindings[bindingIndex];
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
            var uploadStage = bindPoint == PipelineBindPoint.Compute ? PipelineStageFlags.ComputeShaderBit : PipelineStageFlags.AllGraphicsBit;
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
                    var bindings = stage.Layout.Descriptors;
                    for (var bindingIndex = 0; bindingIndex < bindings.Count; bindingIndex++)
                    {
                        var binding = bindings[bindingIndex];
                        if (stage.Layout.UsesBindlessImages && ImageDescriptorBinding.ResourceClass(binding.Kind) != ShaderCompiler.Resources.ImageResourceClass.None)
                        {
                            continue;
                        }

                        var resources = binding.Resources;
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
                            for (var resourceIndex = 0; resourceIndex < resources.Count; resourceIndex++)
                            {
                                var resource = resources[resourceIndex];
                                imageInfos[imageIndex++] = ImageInfo(descriptors.Images[(int)resource], occurrences[(int)resource]++, program, (int)resource);
                            }
                        }
                        else
                        {
                            switch (binding.Kind)
                            {
                                case DescriptorBindingKind.Buffers:
                                    for (var resourceIndex = 0; resourceIndex < resources.Count; resourceIndex++)
                                    {
                                        var resource = resources[resourceIndex];
                                        var view = descriptors.Buffers[(int)resource];
                                        if (view.Buffer.Handle == 0)
                                        {
                                            throw SubmissionScheduler.Fatal($"A storage buffer binding has no buffer: buffer={resource} hash=0x{program.Hash:X16}.");
                                        }

                                        bufferInfos[bufferIndex++] = new DescriptorBufferInfo { Buffer = view.Buffer, Offset = view.Offset, Range = view.Range };
                                    }

                                    break;
                                case DescriptorBindingKind.MipStatistics:
                                    bufferInfos[bufferIndex++] = MipStatisticsDescriptor();
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
                                    for (var resourceIndex = 0; resourceIndex < resources.Count; resourceIndex++)
                                    {
                                        var resource = resources[resourceIndex];
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
                        var resources = stage.Resources.Info;
                        var canonical = resources.GetCanonicalImageBinding((uint)index);
                        if (canonical >= descriptors.Images.Length || resources.GetCanonicalImageBinding(canonical) != canonical)
                        {
                            throw SubmissionScheduler.Fatal($"An image alias has no canonical binding: image={index} canonical={canonical} hash=0x{program.Hash:X16}.");
                        }

                        var expected = canonical == index
                            ? Math.Max(1u, (uint)descriptors.Images[index].MipViews.Length)
                            : 0u;
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
                    var pushStages = bindPoint == PipelineBindPoint.Graphics
                        ? ((entry.Description?.VertexStage.Stage == ShaderStageKind.Mesh
                            ? ShaderStageFlags.MeshBitExt : ShaderStageFlags.VertexBit) |
                           ShaderStageFlags.FragmentBit)
                        : ShaderStageFlags.ComputeBit;
                    PushConstantsShadowed(command, entry.Layout, pushStages, pushData);
                }

                if (writeIndex != 0 && !DescriptorsAlreadyBound(command, bindPoint, entry.Layout, writes[..writeIndex]))
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
