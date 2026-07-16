#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#define _USE_MATH_DEFINES
#include <cmath>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "closed_form.hpp"
#include "filter.hpp"
#include "log.hpp"
#include "math.hpp"
#include "monte_carlo.hpp"

namespace nb = nanobind;

// Checks if a each nanobind array is allocated on the given device
template<typename Array, typename... Arrays>
void check_on_device(int device_type, int device_id, Array&& a, Arrays&&... arrays)
{
    if (a.device_type() != device_type || a.device_id() != device_id)
        throw std::invalid_argument("Input arrays must be on the same device");

    if constexpr (sizeof...(arrays) > 0)
        check_on_device(device_type, device_id, std::forward<Arrays>(arrays)...);
}

template<typename Float>
void validate_common_voxelize_arguments(nb::ndarray<Float, nb::c_contig> const&    vertices,
                                        nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                                        nb::ndarray<Float, nb::c_contig> const&    occupancy)
{
    unsigned int dim = static_cast<unsigned int>(occupancy.ndim());

    if (dim != 2 && dim != 3)
        throw std::invalid_argument(dvx::format_message("Expected occupancy grid with shape (h,w) or (d,h,w) but array has %d dimensions", occupancy.ndim()));

    if ((occupancy.shape(0) == 0) || (occupancy.shape(1) == 0) || ((dim > 2) && (occupancy.shape(2) == 0)))
        throw std::invalid_argument("All dimensions of the ccupancy grid must be > 0");

    if (vertices.ndim() != 2)
        throw std::invalid_argument(dvx::format_message("Expected vertices with shape (n,%d) but array has %d dimensions", dim, vertices.ndim()));

    if (vertices.shape(1) != dim)
        throw std::invalid_argument(dvx::format_message("Expected vertices with shape (n,%d) but array has shape (n,%d)", dim, vertices.shape(1)));

    if (simplices.ndim() != 2)
        throw std::invalid_argument(dvx::format_message("Expected simplices with shape (n,%d) but array has %d dimensions", dim, simplices.ndim()));

    if (simplices.shape(1) != dim)
        throw std::invalid_argument(dvx::format_message("Expected simplices with shape (n,%d) but array has shape (n,%d)", dim, simplices.shape(1)));

    check_on_device(vertices.device_type(), vertices.device_id(), vertices, simplices, occupancy);
}

template<typename Float>
void validate_common_differential_voxelize_arguments(nb::ndarray<Float, nb::c_contig> const&    vertices,
                                                     nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                                                     nb::ndarray<Float, nb::c_contig> const&    occupancy,
                                                     nb::ndarray<Float, nb::c_contig> const&    d_vertices,
                                                     nb::ndarray<Float, nb::c_contig> const&    d_occupancy)
{
    validate_common_voxelize_arguments(vertices, simplices, occupancy);

    check_on_device(vertices.device_type(), vertices.device_id(), d_vertices, d_occupancy);
    validate_common_voxelize_arguments(d_vertices, simplices, d_occupancy);

    if (occupancy.ndim() != d_occupancy.ndim())
        throw std::invalid_argument(dvx::format_message("Dimension mismatch between `occupancy` (dim=%d) and `d_occupancy` (dim=%d)", occupancy.ndim(), d_occupancy.ndim()));

    for (size_t i = 0; i < occupancy.ndim(); ++i)
    {
        if (occupancy.shape(i) != d_occupancy.shape(i))
            throw std::invalid_argument(dvx::format_message("Shape mismatch in dimension `%d` between `occupancy` (=%d) and `d_occupancy` (=%d)", i, occupancy.shape(i), d_occupancy.shape(i)));
    }
}

template<typename Float>
void voxelize_mc(nb::ndarray<Float, nb::c_contig> const&    vertices,
                 nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                 nb::ndarray<Float, nb::c_contig>&          occupancy,
                 uint32_t                                   num_samples_per_voxel,
                 dvx::SamplingFlags                         sampling_flags,
                 Float                                      filter_radius)
{
    validate_common_voxelize_arguments(vertices, simplices, occupancy);

    unsigned int dim = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    dvx::MonteCarloParameters mc_params{
        .num_samples    = num_samples_per_voxel,
        .sampling_flags = sampling_flags};

    if (dim == 2)
        dvx::voxelize_mc_2d<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                   occupancy.data(), occupancy.shape(0), occupancy.shape(1), 
                                   mc_params, filter);
    if (dim == 3)
        dvx::voxelize_mc_3d<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                   occupancy.data(),occupancy.shape(0), occupancy.shape(1), occupancy.shape(2), 
                                   mc_params, filter);
}

template<typename Float>
void voxelize_forward_mc(nb::ndarray<Float, nb::c_contig> const&    vertices,
                         nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                         nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                         nb::ndarray<Float, nb::c_contig> const&    d_vertices,
                         nb::ndarray<Float, nb::c_contig>&          d_occupancy,
                         uint32_t                                   num_samples_per_simplex,
                         dvx::SamplingFlags                         sampling_flags,
                         Float                                      filter_radius)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    unsigned int dim = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    dvx::MonteCarloParameters mc_params{
        .num_samples    = num_samples_per_simplex,
        .sampling_flags = sampling_flags};

    if (dim == 2)
        dvx::d_voxelize_mc_2d<Float, dvx::DifferentiationMode::Forward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                        occupancy.data(), occupancy.shape(0), occupancy.shape(1),
                                                                        d_vertices.data(), d_occupancy.data(),
                                                                        mc_params, filter);
    if (dim == 3)
        dvx::d_voxelize_mc_3d<Float, dvx::DifferentiationMode::Forward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                        occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                                                        d_vertices.data(), d_occupancy.data(),
                                                                        num_samples_per_simplex, filter);
}

template<typename Float>
void voxelize_backward_mc(nb::ndarray<Float, nb::c_contig> const&    vertices,
                          nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                          nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                          nb::ndarray<Float, nb::c_contig>&          d_vertices,
                          nb::ndarray<Float, nb::c_contig> const&    d_occupancy,
                          uint32_t                                   num_samples_per_simplex,
                          dvx::SamplingFlags                         sampling_flags,
                          Float                                      filter_radius)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    unsigned int dim = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    dvx::MonteCarloParameters mc_params{
        .num_samples    = num_samples_per_simplex,
        .sampling_flags = sampling_flags};

    if (dim == 2)
        dvx::d_voxelize_mc_2d<Float, dvx::DifferentiationMode::Backward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                         occupancy.data(), occupancy.shape(0), occupancy.shape(1),
                                                                         d_vertices.data(), d_occupancy.data(),
                                                                         mc_params, filter);
    if (dim == 3)
        dvx::d_voxelize_mc_3d<Float, dvx::DifferentiationMode::Backward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                         occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                                                         d_vertices.data(), d_occupancy.data(),
                                                                         num_samples_per_simplex, filter);
}

template<typename Float>
void voxelize_cf(nb::ndarray<Float, nb::c_contig> const&    vertices,
                       nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                       nb::ndarray<Float, nb::c_contig>&          occupancy)
{
    validate_common_voxelize_arguments(vertices, simplices, occupancy);

    unsigned int dim = vertices.shape(1);

    if (dim == 2)
        dvx::voxelize_cf_2d<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                         occupancy.data(), occupancy.shape(0), occupancy.shape(1));
    if (dim == 3)
        dvx::voxelize_cf<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                      occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2));
}

template<typename Float>
void voxelize_forward_cf(nb::ndarray<Float, nb::c_contig> const&    vertices,
                               nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                               nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                               nb::ndarray<Float, nb::c_contig> const&    d_vertices,
                               nb::ndarray<Float, nb::c_contig>&          d_occupancy)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    unsigned int dim = vertices.shape(1);

    if (dim == 2)
        dvx::voxelize_cf_2d_forward<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                  occupancy.data(), occupancy.shape(0), occupancy.shape(1),
                                                  d_vertices.data(), d_occupancy.data());
    if (dim == 3)
        dvx::voxelize_cf_forward<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                              occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                              d_vertices.data(), d_occupancy.data());
}


template<typename Float>
void voxelize_backward_cf(nb::ndarray<Float, nb::c_contig> const&    vertices,
                                nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                                nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                                nb::ndarray<Float, nb::c_contig>&          d_vertices,
                                nb::ndarray<Float, nb::c_contig> const&    d_occupancy)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    unsigned int dim = vertices.shape(1);

    if (dim == 2)
        dvx::voxelize_cf_2d_backward<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                   occupancy.data(), occupancy.shape(0), occupancy.shape(1),
                                                   d_vertices.data(), d_occupancy.data());
    if (dim == 3)
        dvx::voxelize_cf_backward<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                               occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                               d_vertices.data(), d_occupancy.data());
}

NB_MODULE(dvx_ext, m)
{
    m.attr("__version__") = DVX_VERSION;

    m.def("build_type", [=]()
          {
#if NDEBUG
              return "Release";
#else
              return "Debug";
#endif
          });

    nb::enum_<dvx::SamplingFlagBits>(m, "SamplingFlags", nb::is_arithmetic())
        .value("Empty", dvx::SamplingFlagBits::None)
        .value("Adaptive", dvx::SamplingFlagBits::Adaptive)
        .value("Stratified", dvx::SamplingFlagBits::Stratified);

    m.attr("DefaultSamplingFlags") = dvx::DefaultSamplingFlags;

#define BIND_FUNCTIONS(type, tag)                                                                                                                                                                                                                                                                       \
    m.def("voxelize_mc" tag, voxelize_mc<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("num_samples_per_voxel"), nb::arg("sampling_flags") = dvx::DefaultSamplingFlags, nb::arg("filter_radius"));                                                                    \
    m.def("voxelize_forward_mc" tag, voxelize_forward_mc<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"), nb::arg("num_samples_per_simplex"), nb::arg("sampling_flags") = dvx::DefaultSamplingFlags, nb::arg("filter_radius"));   \
    m.def("voxelize_backward_mc" tag, voxelize_backward_mc<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"), nb::arg("num_samples_per_simplex"), nb::arg("sampling_flags") = dvx::DefaultSamplingFlags, nb::arg("filter_radius")); \
    m.def("voxelize_cf" tag, voxelize_cf<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"));                                                                                                                                                                                       \
    m.def("voxelize_forward_cf" tag, voxelize_forward_cf<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"));                                                                                                                        \
    m.def("voxelize_backward_cf" tag, voxelize_backward_cf<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"));

    BIND_FUNCTIONS(float, "_f32")
    BIND_FUNCTIONS(double, "_f64")

    nb::enum_<dvx::LogLevel>(m, "LogLevel")
        .value("Trace", dvx::LogLevel::Trace)
        .value("Debug", dvx::LogLevel::Debug)
        .value("Info", dvx::LogLevel::Info)
        .value("Warn", dvx::LogLevel::Warn)
        .value("Error", dvx::LogLevel::Error);

    m.def("set_log_level", dvx::set_log_level, nb::arg("log_level"));

    // Redirect log output to Python
    dvx::set_log_output_function([](char const* str){ nb::print(str); });
}
