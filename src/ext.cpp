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
#include "explicit.hpp"
#include "filter.hpp"
#include "log.hpp"
#include "math.hpp"
#include "monte_carlo.hpp"

namespace nb = nanobind;

static bool suppress_warnings = false;

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
        throw std::invalid_argument(format_message("Expected occupancy grid with shape (h,w) or (d,h,w) but array has %d dimensions", occupancy.ndim()));

    if ((occupancy.shape(0) == 0) || (occupancy.shape(1) == 0) || ((dim > 2) && (occupancy.shape(2) == 0)))
        throw std::invalid_argument("All dimensions of the ccupancy grid must be > 0");

    if (vertices.ndim() != 2)
        throw std::invalid_argument(format_message("Expected vertices with shape (n,%d) but array has %d dimensions", dim, vertices.ndim()));

    if (vertices.shape(1) != dim)
        throw std::invalid_argument(format_message("Expected vertices with shape (n,%d) but array has shape (n,%d)", dim, vertices.shape(1)));

    if (simplices.ndim() != 2)
        throw std::invalid_argument(format_message("Expected simplices with shape (n,%d) but array has %d dimensions", dim, simplices.ndim()));

    if (simplices.shape(1) != dim)
        throw std::invalid_argument(format_message("Expected simplices with shape (n,%d) but array has shape (n,%d)", dim, simplices.shape(1)));

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
        throw std::invalid_argument(format_message("Dimension mismatch between `occupancy` (dim=%d) and `d_occupancy` (dim=%d)", occupancy.ndim(), d_occupancy.ndim()));

    for (size_t i = 0; i < occupancy.ndim(); ++i)
    {
        if (occupancy.shape(i) != d_occupancy.shape(i))
            throw std::invalid_argument(format_message("Shape mismatch in dimension `%d` between `occupancy` (=%d) and `d_occupancy` (=%d)", i, occupancy.shape(i), d_occupancy.shape(i)));
    }
}

template<typename Float>
void voxelize_mc(nb::ndarray<Float, nb::c_contig> const&    vertices,
                 nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                 nb::ndarray<Float, nb::c_contig>&          occupancy,
                 uint32_t num_samples_per_voxel, Float filter_radius)
{
    validate_common_voxelize_arguments(vertices, simplices, occupancy);

    unsigned int dim = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    if (!suppress_warnings && num_samples_per_voxel < 16)
    {
        for (int i = 0; i < dim; ++i)
        {
            Float voxel_spacing = Float(2) / occupancy.shape(i);
            if (filter.radius < voxel_spacing)
            {
                nb::print(format_message("Warning: The filter radius (%f) is smaller than the space between voxels in dimension %d (%f).\n"
                                         "This can result in voxels without valid samples. Consider increasing the sample count or the filter radius.",
                                         filter.radius, i, voxel_spacing));
            }
        }
    }

    if (dim == 2)
        dvx::voxelize_mc_2d<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                   occupancy.data(), occupancy.shape(0), occupancy.shape(1), num_samples_per_voxel, filter);
    if (dim == 3)
        dvx::voxelize_mc_3d<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                   occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2), num_samples_per_voxel, filter);
}

template<typename Float>
void voxelize_forward_mc(nb::ndarray<Float, nb::c_contig> const&    vertices,
                         nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                         nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                         nb::ndarray<Float, nb::c_contig> const&    d_vertices,
                         nb::ndarray<Float, nb::c_contig>&          d_occupancy,
                         uint32_t num_samples_per_simplex, Float filter_radius)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    unsigned int dim = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    if (dim == 2)
        dvx::d_voxelize_mc_2d<Float, dvx::DifferentiationMode::Forward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                        occupancy.data(), occupancy.shape(0), occupancy.shape(1),
                                                                        d_vertices.data(), d_occupancy.data(),
                                                                        num_samples_per_simplex, filter);
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
                          uint32_t num_samples_per_simplex, Float filter_radius)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    unsigned int dim = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    if (dim == 2)
        dvx::d_voxelize_mc_2d<Float, dvx::DifferentiationMode::Backward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                         occupancy.data(), occupancy.shape(0), occupancy.shape(1),
                                                                         d_vertices.data(), d_occupancy.data(),
                                                                         num_samples_per_simplex, filter);
    if (dim == 3)
        dvx::d_voxelize_mc_3d<Float, dvx::DifferentiationMode::Backward>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                                                         occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                                                         d_vertices.data(), d_occupancy.data(),
                                                                         num_samples_per_simplex, filter);
}

template<typename Float>
void voxelize_explicit(nb::ndarray<Float, nb::c_contig> const&    vertices,
                       nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                       nb::ndarray<Float, nb::c_contig>&          occupancy)
{
    validate_common_voxelize_arguments(vertices, simplices, occupancy);

    dvx::voxelize_explicit<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                  occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2));
}

template<typename Float>
void voxelize_forward_explicit(nb::ndarray<Float, nb::c_contig> const&    vertices,
                               nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                               nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                               nb::ndarray<Float, nb::c_contig> const&    d_vertices,
                               nb::ndarray<Float, nb::c_contig>&          d_occupancy)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    dvx::voxelize_explicit_forward<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                          occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                          d_vertices.data(), d_occupancy.data());
}


template<typename Float>
void voxelize_backward_explicit(nb::ndarray<Float, nb::c_contig> const&    vertices,
                                nb::ndarray<uint32_t, nb::c_contig> const& simplices,
                                nb::ndarray<Float, nb::c_contig>&          occupancy /*unused*/,
                                nb::ndarray<Float, nb::c_contig>&          d_vertices,
                                nb::ndarray<Float, nb::c_contig> const&    d_occupancy)
{
    validate_common_differential_voxelize_arguments(vertices, simplices, occupancy, d_vertices, d_occupancy);

    throw std::invalid_argument("Not implemented.");
}

NB_MODULE(dvx_ext, m)
{
#if NDEBUG
    std::string build_type = "Release";
#else
    std::string build_type = "Debug";
#endif

    m.def("build_type", [=]()
          { return build_type; });
    m.def("mute", [=]()
          { suppress_warnings = true; });
    m.def("unmute", [=]()
          { suppress_warnings = false; });

#define BIND_FUNCTIONS(type, tag)                                                                                                                                                                                                                \
    m.def("voxelize_mc" tag, voxelize_mc<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("num_samples_per_voxel"), nb::arg("filter_radius"));                                                                    \
    m.def("voxelize_forward_mc" tag, voxelize_forward_mc<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"), nb::arg("num_samples_per_simplex"), nb::arg("filter_radius"));   \
    m.def("voxelize_backward_mc" tag, voxelize_backward_mc<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"), nb::arg("num_samples_per_simplex"), nb::arg("filter_radius")); \
    m.def("voxelize_explicit" tag, voxelize_explicit<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"));                                                                                                                    \
    m.def("voxelize_forward_explicit" tag, voxelize_forward_explicit<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"));                                                     \
    m.def("voxelize_backward_explicit" tag, voxelize_backward_explicit<type>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"));

    BIND_FUNCTIONS(float, "_f32")
    BIND_FUNCTIONS(double, "_f64")
}
