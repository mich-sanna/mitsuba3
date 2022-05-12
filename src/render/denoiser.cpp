#include <mitsuba/render/denoiser.h>

#include <mitsuba/render/optix_api.h>
#include <drjit-core/optix.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float>
ref<Bitmap> denoise(const Bitmap &noisy_, Bitmap * /*albedo*/, Bitmap * /*normals*/) {
    if constexpr (dr::is_cuda_array_v<Float>) {
        ref<Bitmap> noisy = noisy_.convert(Bitmap::PixelFormat::RGB, Struct::Type::Float32, false);

        OptixDeviceContext context = jit_optix_context();

        OptixDenoiser denoiser = nullptr;
        OptixDenoiserOptions options = {};
        options.guideAlbedo = false;
        options.guideNormal = false;
        OptixDenoiserModelKind modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;

        jit_optix_check(optixDenoiserCreate(context, modelKind, &options, &denoiser));

        OptixDenoiserSizes sizes = {};
        jit_optix_check(optixDenoiserComputeMemoryResources(
            denoiser, noisy->width(), noisy->height(), &sizes));

        CUstream stream = jit_cuda_stream();

        uint32_t state_size   = sizes.stateSizeInBytes;
        CUdeviceptr state     = jit_malloc(AllocType::Device, state_size);
        uint32_t scratch_size = sizes.withoutOverlapScratchSizeInBytes;
        CUdeviceptr scratch   = jit_malloc(AllocType::Device, scratch_size);
        jit_optix_check(optixDenoiserSetup(denoiser, stream, noisy->width(),
                                       noisy->height(), state, state_size,
                                       scratch, scratch_size));

        OptixDenoiserLayer layers = {};

        OptixImage2D &input = layers.input;
        input.width = noisy->width();
        input.height = noisy->height();
        input.rowStrideInBytes = noisy->width() * noisy->bytes_per_pixel();
        input.pixelStrideInBytes = noisy->bytes_per_pixel();
        input.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        input.data = jit_malloc(AllocType::Device, noisy->buffer_size());
        jit_memcpy_async(JitBackend::CUDA, input.data, noisy->data(),
                         noisy->buffer_size());

        OptixImage2D &output = layers.output;
        output.width = noisy->width();
        output.height = noisy->height();
        output.rowStrideInBytes = noisy->width() * noisy->bytes_per_pixel();
        output.pixelStrideInBytes = noisy->bytes_per_pixel();
        output.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        output.data = jit_malloc(AllocType::Device, noisy->buffer_size());

        OptixDenoiserParams params = {};
        params.denoiseAlpha = 0;
        params.hdrIntensity = jit_malloc(AllocType::Device, sizeof(float));
        jit_optix_check(optixDenoiserComputeIntensity(denoiser, stream, &input,
                                                      params.hdrIntensity,
                                                      scratch, scratch_size));
        params.blendFactor = 0.0f;
        params.hdrAverageColor = nullptr;

        OptixDenoiserGuideLayer guide_layer = {};
        uint32_t num_layers = 1;
        jit_optix_check(optixDenoiserInvoke(
            denoiser, stream, &params, state, state_size, &guide_layer, &layers,
            num_layers, 0, 0, scratch, scratch_size));

        void *denoised_data =
            jit_malloc_migrate(output.data, AllocType::Host, false);
        jit_sync_thread();

        Bitmap *denoised = new Bitmap(
            noisy->pixel_format(), noisy->component_format(), noisy->size(),
            noisy->channel_count(), {}, (uint8_t *) denoised_data);

        jit_optix_check(optixDenoiserDestroy(denoiser));
        jit_free(input.data);
        jit_free(params.hdrIntensity);
        jit_free(output.data);
        jit_free(state);
        jit_free(scratch);

        return denoised;
    }
    jit_raise("You should not be calling this without a CUDA (AD) variant.");

    return new Bitmap(noisy_) /* unused */;
}

template <typename Float> ref<Bitmap> denoise(const Bitmap &noisy) {
    return denoise<Float>(noisy, nullptr, nullptr);
}

template MI_EXPORT_LIB ref<Bitmap>
denoise<float>(const Bitmap &noisy, Bitmap *albedo, Bitmap *normals);
template MI_EXPORT_LIB ref<Bitmap> denoise<float>(const Bitmap &noisy);
template MI_EXPORT_LIB ref<Bitmap>
denoise<dr::DiffArray<dr::CUDAArray<float>>>(const Bitmap &noisy,
                                             Bitmap *albedo, Bitmap *normals);
template MI_EXPORT_LIB ref<Bitmap>
denoise<dr::DiffArray<dr::CUDAArray<float>>>(const Bitmap &noisy);

NAMESPACE_END(mitsuba)
