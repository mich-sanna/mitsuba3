#pragma once

#include <mitsuba/core/bitmap.h>

NAMESPACE_BEGIN(mitsuba)

extern MI_EXPORT_LIB ref<Bitmap> denoise_temporal(
    const Bitmap &noisy, const Bitmap &flow, const Bitmap &previous_denoised,
    const Bitmap *albedo = nullptr, const Bitmap *normals = nullptr);

extern MI_EXPORT_LIB ref<Bitmap>
denoise(const Bitmap &noisy, const Bitmap *albedo = nullptr, const Bitmap *normals=nullptr);

extern MI_EXPORT_LIB ref<Bitmap>
denoise(const Bitmap &noisy, const std::string &albedo_ch_name = "",
        const std::string &normals_ch_name = "",
        const std::string &noisy_ch_name = "<root>");

NAMESPACE_END(mitsuba)
