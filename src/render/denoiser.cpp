#include <mitsuba/render/denoiser.h>

#include <drjit-core/optix.h>
#include <mitsuba/render/optix_api.h>

NAMESPACE_BEGIN(mitsuba)

void buildOptixImage2DfromBitmap(const Bitmap *bitmap, OptixImage2D *optixImage,
                                 bool copy_data = true) {
    optixImage->width = bitmap->width();
    optixImage->height = bitmap->height();
    optixImage->rowStrideInBytes = bitmap->width() * bitmap->bytes_per_pixel();
    optixImage->pixelStrideInBytes = bitmap->bytes_per_pixel();
    optixImage->format = OPTIX_PIXEL_FORMAT_FLOAT3; // TODO: Better
    optixImage->data = jit_malloc(AllocType::Device, bitmap->buffer_size());
    if (copy_data)
        jit_memcpy_async(JitBackend::CUDA, optixImage->data, bitmap->data(),
                         bitmap->buffer_size());
}

ref<Bitmap> denoise(const Bitmap &noisy_, const Bitmap *albedo_,
                    const Bitmap *normals_) {
    ref<Bitmap> noisy =
        noisy_.convert(Bitmap::PixelFormat::RGB, Struct::Type::Float32, false);

    OptixDeviceContext context = jit_optix_context();

    OptixDenoiser denoiser = nullptr;
    OptixDenoiserOptions options = {};
    options.guideAlbedo = albedo_ != nullptr;
    options.guideNormal = normals_ != nullptr;
    OptixDenoiserModelKind modelKind = OPTIX_DENOISER_MODEL_KIND_HDR;

    jit_optix_check(
        optixDenoiserCreate(context, modelKind, &options, &denoiser));

    OptixDenoiserSizes sizes = {};
    jit_optix_check(optixDenoiserComputeMemoryResources(
        denoiser, noisy->width(), noisy->height(), &sizes));

    CUstream stream = jit_cuda_stream();

    uint32_t state_size = sizes.stateSizeInBytes;
    CUdeviceptr state = jit_malloc(AllocType::Device, state_size);
    uint32_t scratch_size = sizes.withoutOverlapScratchSizeInBytes;
    CUdeviceptr scratch = jit_malloc(AllocType::Device, scratch_size);
    jit_optix_check(optixDenoiserSetup(denoiser, stream, noisy->width(),
                                       noisy->height(), state, state_size,
                                       scratch, scratch_size));

    OptixDenoiserLayer layers = {};
    buildOptixImage2DfromBitmap(noisy.get(), &layers.input);
    buildOptixImage2DfromBitmap(noisy.get(), &layers.output, false);

    OptixDenoiserParams params = {};
    params.denoiseAlpha = 0;
    params.hdrIntensity = jit_malloc(AllocType::Device, sizeof(float));
    jit_optix_check(optixDenoiserComputeIntensity(
        denoiser, stream, &layers.input, params.hdrIntensity, scratch,
        scratch_size));
    params.blendFactor = 0.0f;
    params.hdrAverageColor = nullptr;

    OptixDenoiserGuideLayer guide_layer = {};
    if (options.guideAlbedo) {
    }

    if (options.guideNormal) {
        ref<Bitmap> normals = normals_->convert(Bitmap::PixelFormat::XYZ,
                                                Struct::Type::Float32, false);

        // Flip from left-handed coordinate system to right-handed (y is up)
        float *normals_data = (float *) normals->data();
        size_t width = (size_t) normals->width();
        size_t height = (size_t) normals->height();
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t index = y * width * 3 + x * 3;
                normals_data[index + 0] = -normals_data[index + 0];
                normals_data[index + 2] = -normals_data[index + 2];
            }
        }

        buildOptixImage2DfromBitmap(normals.get(), &guide_layer.normal);
    }

    uint32_t num_layers = 1;
    jit_optix_check(optixDenoiserInvoke(
        denoiser, stream, &params, state, state_size, &guide_layer, &layers,
        num_layers, 0, 0, scratch, scratch_size));

    void *denoised_data =
        jit_malloc_migrate(layers.output.data, AllocType::Host, false);
    jit_sync_thread();

    Bitmap *denoised = new Bitmap(
        noisy->pixel_format(), noisy->component_format(), noisy->size(),
        noisy->channel_count(), {}, (uint8_t *) denoised_data);

    jit_optix_check(optixDenoiserDestroy(denoiser));
    if (options.guideAlbedo)
        jit_free(guide_layer.albedo.data);
    if (options.guideNormal)
        jit_free(guide_layer.normal.data);
    jit_free(layers.input.data);
    jit_free(params.hdrIntensity);
    jit_free(layers.output.data);
    jit_free(state);
    jit_free(scratch);

    return denoised;
}

ref<Bitmap> denoise(const Bitmap &noisy, const std::string &albedo_ch_name,
                    const std::string &normals_ch_name,
                    const std::string &noisy_ch_name) {
    if (noisy.pixel_format() != Bitmap::PixelFormat::MultiChannel)
        return denoise(noisy, nullptr, nullptr);

    const Bitmap *albedo = nullptr;
    const Bitmap *normals = nullptr;
    const Bitmap *image = nullptr;

    bool found_albedo = albedo_ch_name == "";
    bool found_normals = normals_ch_name == "";

    std::vector<std::pair<std::string, ref<Bitmap>>> res = noisy.split();
    for (const auto &pair : res) {
        if (found_albedo && found_normals && image != nullptr)
            break;
        if (!found_albedo && pair.first == albedo_ch_name) {
            found_albedo = true;
            albedo = pair.second.get();
        }
        if (!found_normals && pair.first == normals_ch_name) {
            found_normals = true;
            normals = pair.second.get();
        }
        if (image == nullptr && pair.first == noisy_ch_name)
            image = pair.second.get();
    }

    if (image == nullptr)
        Throw("Could not find rendered image with channel names '%s' in:\n%s",
              noisy_ch_name, noisy.to_string());

    const Bitmap &image_ = *image;

    return denoise(image_, albedo, normals);
}

NAMESPACE_END(mitsuba)
