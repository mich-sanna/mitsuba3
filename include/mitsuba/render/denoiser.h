#pragma once

#include <mitsuba/core/bitmap.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float>
ref<Bitmap> denoise(const Bitmap &noisy, Bitmap *albedo, Bitmap *normals);

template <typename Float> ref<Bitmap> denoise(const Bitmap &noisy);

extern template MI_EXPORT_LIB ref<Bitmap>
denoise<float>(const Bitmap &noisy, Bitmap *albedo, Bitmap *normals);
extern template MI_EXPORT_LIB ref<Bitmap> denoise<float>(const Bitmap &noisy);
extern template MI_EXPORT_LIB ref<Bitmap>
denoise<dr::DiffArray<dr::CUDAArray<float>>>(const Bitmap &noisy,
                                             Bitmap *albedo, Bitmap *normals);
extern template MI_EXPORT_LIB ref<Bitmap>
denoise<dr::DiffArray<dr::CUDAArray<float>>>(const Bitmap &noisy);

NAMESPACE_END(mitsuba)
