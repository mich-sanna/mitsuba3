#pragma once

#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/optix_api.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * OptiX Denoiser
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Denoiser : public Object {
public:
    MI_IMPORT_TYPES()

    Denoiser(const ScalarVector2u &input_size, bool albedo, bool normals,
             bool temporal);

    ~Denoiser();

    ref<Bitmap> denoise(const Bitmap &noisy, const Bitmap *albedo = nullptr,
                        const Bitmap *normals = nullptr,
                        const Bitmap *previous_denoised = nullptr,
                        const Bitmap *flow = nullptr);

    ref<Bitmap> denoise(const Bitmap &noisy, const std::string &albedo_ch = "",
                        const std::string &normals_ch = "",
                        const std::string &noisy_ch = "<root>");

    virtual std::string to_string() const override;

    MI_DECLARE_CLASS()
private:
    CUdeviceptr m_state;
    uint32_t m_state_size;
    CUdeviceptr m_scratch;
    uint32_t m_scratch_size;
    OptixDenoiserOptions m_options;
    bool m_temporal;
    OptixDenoiser m_denoiser;
    CUdeviceptr m_input_data;
    CUdeviceptr m_albedo_data;
    CUdeviceptr m_normal_data;
    CUdeviceptr m_hdr_intensity;
    CUdeviceptr m_previous_output_data;
    CUdeviceptr m_flow_data;
    CUdeviceptr m_output_data;
};

MI_EXTERN_CLASS(Denoiser)

NAMESPACE_END(mitsuba)
