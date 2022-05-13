#include <mitsuba/render/denoiser.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(denoiser) {
    MI_PY_IMPORT_TYPES()
    m.def("denoise",
        py::overload_cast<const Bitmap &, const std::string&, const std::string&, const std::string&>(&denoise),
        "noisy"_a, "albedo_ch_name"_a = "", "normals_ch_name"_a = "", "noisy_ch_name"_a = "<root>", D(denoise))
    .def("denoise",
        py::overload_cast<const Bitmap &, const Bitmap *, const Bitmap *>(&denoise),
        "noisy"_a, "albedo"_a = nullptr, "normals"_a = nullptr, D(denoise, 2));
}
