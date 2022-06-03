#include <mitsuba/render/denoiser.h>
#include <mitsuba/render/optix_api.h>
#include <drjit-core/optix.h>

NAMESPACE_BEGIN(mitsuba)

static void buildOptixImage2DfromBitmap(const Bitmap &bitmap,
                                        OptixPixelFormat pixel_format,
                                        OptixImage2D *optix_image,
                                        bool copy_data = true) {
    optix_image->width = bitmap.width();
    optix_image->height = bitmap.height();
    optix_image->rowStrideInBytes = bitmap.width() * bitmap.bytes_per_pixel();
    optix_image->pixelStrideInBytes = bitmap.bytes_per_pixel();
    optix_image->format = pixel_format; // TODO: Better

    if (copy_data)
        jit_memcpy_async(JitBackend::CUDA, optix_image->data, bitmap.data(),
                         bitmap.buffer_size());
}

MI_VARIANT Denoiser<Float, Spectrum>::Denoiser(const ScalarVector2u &input_size,
                                               bool albedo, bool normals,
                                               bool temporal)
    : m_temporal(temporal) {
    if (normals && !albedo)
        Throw("The denoiser cannot use normals to guide its process without "
              "also providing albedo information!");

    optix_initialize();

    OptixDeviceContext context = jit_optix_context();
    m_denoiser = nullptr;
    m_options = {};
    m_options.guideAlbedo = albedo;
    m_options.guideNormal = normals;
    OptixDenoiserModelKind model_kind = temporal
                                            ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL
                                            : OPTIX_DENOISER_MODEL_KIND_HDR;
    jit_optix_check(
        optixDenoiserCreate(context, model_kind, &m_options, &m_denoiser));

    OptixDenoiserSizes sizes = {};
    jit_optix_check(optixDenoiserComputeMemoryResources(
        m_denoiser, input_size.x(), input_size.y(), &sizes));
    CUstream stream = jit_cuda_stream();
    m_state_size = sizes.stateSizeInBytes;
    m_state = jit_malloc(AllocType::Device, m_state_size);
    m_scratch_size = sizes.withoutOverlapScratchSizeInBytes;
    m_scratch = jit_malloc(AllocType::Device, m_scratch_size);
    jit_optix_check(optixDenoiserSetup(m_denoiser, stream, input_size.x(),
                                       input_size.y(), m_state, m_state_size,
                                       m_scratch, m_scratch_size));

    size_t pixel_count = input_size.x() * input_size.y();
    m_input_data =
        jit_malloc(AllocType::Device, pixel_count * 4 * sizeof(float));
    m_hdr_intensity = jit_malloc(AllocType::Device, sizeof(float));
    if (albedo)
        m_albedo_data =
            jit_malloc(AllocType::Device, pixel_count * 3 * sizeof(float));
    if (normals)
        m_normal_data =
            jit_malloc(AllocType::Device, pixel_count * 3 * sizeof(float));
    if (temporal) {
        m_flow_data =
            jit_malloc(AllocType::Device, pixel_count * 2 * sizeof(float));
        m_previous_output_data =
            jit_malloc(AllocType::Device, pixel_count * 4 * sizeof(float));
    }
    m_output_data =
        jit_malloc(AllocType::Device, pixel_count * 4 * sizeof(float));
}

MI_VARIANT Denoiser<Float, Spectrum>::~Denoiser() {
    jit_optix_check(optixDenoiserDestroy(m_denoiser));
    if (m_options.guideAlbedo)
        jit_free(m_albedo_data);
    if (m_options.guideNormal)
        jit_free(m_normal_data);
    if (m_temporal)
        jit_free(m_flow_data);
    jit_free(m_input_data);
    jit_free(m_hdr_intensity);
    jit_free(m_output_data);
    jit_free(m_state);
    jit_free(m_scratch);
}

MI_VARIANT
ref<Bitmap> Denoiser<Float, Spectrum>::denoise(const Bitmap &noisy,
                                               const Bitmap *albedo,
                                               const Bitmap *normals,
                                               const Bitmap *previous_denoised,
                                               const Bitmap *flow) {

    // TODO check that input matches current state

    OptixDenoiserLayer layers = {};
    layers.input.data = m_input_data;
    buildOptixImage2DfromBitmap(noisy, OPTIX_PIXEL_FORMAT_FLOAT4,
                                &layers.input);
    layers.output.data = m_output_data;
    buildOptixImage2DfromBitmap(noisy, OPTIX_PIXEL_FORMAT_FLOAT4,
                                &layers.output, false);

    CUstream stream = jit_cuda_stream();

    OptixDenoiserParams params = {};
    params.hdrIntensity = m_hdr_intensity;
    params.denoiseAlpha = true;
    jit_optix_check(optixDenoiserComputeIntensity(
        m_denoiser, stream, &layers.input, m_hdr_intensity, m_scratch,
        m_scratch_size));
    params.blendFactor = 0.0f;
    params.hdrAverageColor = nullptr;

    OptixDenoiserGuideLayer guide_layer = {};

    if (m_options.guideAlbedo) {
        guide_layer.albedo.data = m_albedo_data;
        buildOptixImage2DfromBitmap(*albedo, OPTIX_PIXEL_FORMAT_FLOAT3,
                                    &guide_layer.albedo);
    }
    if (m_options.guideNormal) {
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

        guide_layer.normal.data = m_normal_data;
        buildOptixImage2DfromBitmap(*normals, OPTIX_PIXEL_FORMAT_FLOAT3,
                                    &guide_layer.normal);
    }
    if (m_temporal) {
        guide_layer.flow.data = m_flow_data;
        buildOptixImage2DfromBitmap(*flow, OPTIX_PIXEL_FORMAT_FLOAT2,
                                    &guide_layer.flow);
        layers.previousOutput.data = m_previous_output_data;
        buildOptixImage2DfromBitmap(*previous_denoised,
                                    OPTIX_PIXEL_FORMAT_FLOAT4,
                                    &layers.previousOutput);
    }

    jit_optix_check(optixDenoiserInvoke(m_denoiser, stream, &params, m_state,
                                        m_state_size, &guide_layer, &layers, 1,
                                        0, 0, m_scratch, m_scratch_size));

    void *denoised_data =
        jit_malloc_migrate(layers.output.data, AllocType::Host, false);
    jit_sync_thread();

    Bitmap *denoised =
        new Bitmap(noisy.pixel_format(), noisy.component_format(), noisy.size(),
                   noisy.channel_count(), {}, (uint8_t *) denoised_data);
    return denoised;
}

MI_VARIANT
ref<Bitmap> Denoiser<Float, Spectrum>::denoise(const Bitmap &noisy_,
                                               const std::string &albedo_ch,
                                               const std::string &normals_ch,
                                               const std::string &noisy_ch) {
    if (noisy_.pixel_format() != Bitmap::PixelFormat::MultiChannel)
        return denoise(noisy_, nullptr, nullptr, nullptr, nullptr);

    const Bitmap *albedo = nullptr;
    const Bitmap *normals = nullptr;
    const Bitmap *noisy = nullptr;

    bool found_albedo = albedo_ch == "";
    bool found_normals = normals_ch == "";

    std::vector<std::pair<std::string, ref<Bitmap>>> res = noisy_.split();
    for (const auto &pair : res) {
        if (found_albedo && found_normals && noisy != nullptr)
            break;
        if (!found_albedo && pair.first == albedo_ch) {
            found_albedo = true;
            albedo = pair.second.get();
        }
        if (!found_normals && pair.first == normals_ch) {
            found_normals = true;
            normals = pair.second.get();
        }
        if (noisy == nullptr && pair.first == noisy_ch)
            noisy = pair.second.get();
    }

    const auto throw_missing_channel = [&](const std::string &channel) {
        Throw("Could not find rendered image with channel name '%s' in:\n%s",
              channel, noisy_.to_string());
    };
    if (noisy == nullptr)
        throw_missing_channel(noisy_ch);
    if (!found_albedo)
        throw_missing_channel(albedo_ch);
    if (!found_normals)
        throw_missing_channel(normals_ch);

    return denoise(*noisy, albedo, normals);
}

MI_VARIANT
std::string Denoiser<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "Denoiser[" << std::endl
        << "  albedo = " << m_options.guideAlbedo << "," << std::endl
        << "  normals = " << m_options.guideNormal << "," << std::endl
        << "  temporal = " << m_temporal << std::endl
        << "]";
    return oss.str();
}

MI_IMPLEMENT_CLASS_VARIANT(Denoiser, Object, "denoiser")
MI_INSTANTIATE_CLASS(Denoiser)

NAMESPACE_END(mitsuba)
