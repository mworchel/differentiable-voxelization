#include <algorithm>
#include <limits>
#include <functional>
#include <random>
#include <stdexcept>
#include <cstdint>
#define _USE_MATH_DEFINES
#include <cmath>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "log.hpp"
#include "math.hpp"

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

namespace dvx
{

enum class FilterType
{
    Box = 0,
    Gaussian
};

template<typename Float>
struct Filter
{
    FilterType type;
    Float      radius; // Filter radius in world space

    template<unsigned int N>
    DEVICE Float eval(Vector<Float, N> const& difference)
    {
        switch (type)
        {
        case FilterType::Box:
        {
            for (unsigned int i = 0; i < N; ++i)
                if (MAYBE_STD(abs)(difference[i]) < radius)
                    return true;
            return false;
        }
        case FilterType::Gaussian:
        {
            Float const distance = norm(difference);
            constexpr Float const norm = Float(0.5 * M_SQRT1_2 * M_2_SQRTPI);
            return Float(distance < radius) * norm * MAYBE_STD(exp)(-Float(0.5) * (distance * distance) / (2 * gaussian.stddev * gaussian.stddev));
        }
        default:
            return 0;
        }
    }

    union
    {
        struct
        {
        } box;
        struct
        {
            Float stddev;
        } gaussian;
    };
};

template<typename Float, unsigned int N>
Float generalized_winding_number(Float const* vertices, uint32_t num_vertices,
                                 uint32_t const* simplices, uint32_t num_simplices,
                                 Point<Float, N> const& x)
{
    Float occupancy = 0.f;
    for (uint32_t s = 0; s < num_simplices; ++s)
    {
        // Load the vertices and compute `e` vectors
        Vector<Float, N> e[N];
        for (int i = 0; i < N; ++i)
        {
            Point<Float, N> vi;
            for (int d = 0; d < N; ++d)
                vi[d] = vertices[N * simplices[N*s + i] + d];

            e[i] = normalize(vi - x);
        }

        Float const alpha = [&e]() {
            if constexpr (N == 2)
                return e[0].x() * e[1].y() - e[0].y() * e[1].x();
            else // N == 3
                return dot(cross(e[0], e[1]), e[2]);
        }();
        Float const beta  = 1 + ((N == 2) ? dot(e[0], e[1]) : dot(e[0], e[1]) + dot(e[1], e[2]) + dot(e[2], e[0]));
        occupancy += atan2(alpha, beta);
    }

    Float normalization = (N == 2) ? 2 * Float(3.14159265358979323) : 4 * Float(3.14159265358979323);

    return occupancy / normalization;
}

template<typename Float>
void voxelize_2d_cpu(Float const* vertices, uint32_t num_vertices,
                     uint32_t const* edges, uint32_t num_edges,
                     Float* occupancy, uint32_t height, uint32_t width,
                     uint32_t num_samples_per_voxel, 
                     Filter<Float> filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector2<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height
    };

    // TODO: Avoid re-allocation
    Float* weight = new Float[height*width];
    for (uint64_t i = 0; i < height*width; ++i)
        weight[i] = Float(0);

    uint64_t num_samples = num_samples_per_voxel * height * width;
    for (uint64_t sample_index = 0; sample_index < num_samples; ++sample_index)
    {
        // Generate sample position in voxel-local space
        Point2<Float> const sample_local{
            distribution(engine),
            distribution(engine),
        };

        // Transform sample to global space
        uint64_t const voxel_index = sample_index / num_samples_per_voxel;

        // Linear indexing: voxel_index = height*width*z + width*y + x
        uint32_t const y = voxel_index / width;
        uint32_t const x = voxel_index % width;

        Point2<Float> const voxel_origin{
            x * voxel_size[0] - Float(1),
            y * voxel_size[1] - Float(1)
        };

        Point2<Float> const sample = voxel_origin + voxel_size * sample_local;

        Float sample_occupancy = generalized_winding_number<Float, 2>(vertices, num_vertices, 
                                                                      edges, num_edges, 
                                                                      sample);

        // Splat the sample into the grid
        Point2<Float> const sample_grid_coord = Point2<Float>(Float(x), Float(y)) + sample_local;

        Float filter_support[2] = {
            filter.radius / voxel_size[0],
            filter.radius / voxel_size[1]
        };

        int32_t const min_x = floor(sample_grid_coord[0] - filter_support[0]);
        int32_t const max_x = floor(sample_grid_coord[0] + filter_support[0]);
        int32_t const min_y = floor(sample_grid_coord[1] - filter_support[1]);
        int32_t const max_y = floor(sample_grid_coord[1] + filter_support[1]);

        for (int32_t ny = std::max(min_y, 0); ny < std::min(static_cast<int32_t>(height), max_y + 1); ++ny)
        {
            for (int32_t nx = std::max(min_x, 0); nx < std::min(static_cast<int32_t>(width), max_x + 1); ++nx)
            {
                // Compute the weight using the distance from the sample to the voxel center
                Point2<Float> const n_center{
                    (nx + Float(0.5)) * voxel_size[0] - Float(1),
                    (ny + Float(0.5)) * voxel_size[1] - Float(1)};

                Float const filter_weight = filter.eval(n_center - sample);
                if (filter_weight > 0)
                {
                    uint64_t const n_voxel_index = width * ny + nx;
                    occupancy[n_voxel_index] += filter_weight * sample_occupancy;
                    ++weight[n_voxel_index];
                }
            }
        }
    }

    // Normalize occupancy grid
    for (uint64_t i = 0; i < height*width; ++i)
    {
        Float w = weight[i];
        if (w != Float(0))
            occupancy[i] *= (2 / w);
    }

    delete[] weight;
}

template<typename Float>
void voxelize_3d_cpu(Float const* vertices, uint32_t num_vertices,
                     uint32_t const* faces, uint32_t num_faces,
                     Float* grid, uint32_t depth, uint32_t height, uint32_t width,
                     uint32_t num_samples_per_voxel)
{
    // TODO: Fixme

    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // Filter radius in world space
    Float const filter_radius = 0.1f;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector3<Float> const voxel_size{
        Float(2) / depth,
        Float(2) / height,
        Float(2) / width
    };

    uint64_t num_samples = num_samples_per_voxel * depth * height * width;
    for (uint64_t sample_index = 0; sample_index < num_samples; ++sample_index)
    {
        // Generate sample position in voxel-local space
        Point3<Float> const sample_local{
            distribution(engine),
            distribution(engine),
            distribution(engine),
        };

        // Transform sample to global space
        uint64_t const voxel_index = sample_index / num_samples_per_voxel;

        // Linear indexing: voxel_index = height*width*z + width*y + x
        uint32_t const z = voxel_index / (height * width);
        uint32_t const y = (voxel_index % (height * width)) / width;
        uint32_t const x = (voxel_index % (height * width)) % width;

        Point3<Float> const voxel_origin = {
            z * voxel_size[0],
            y * voxel_size[1],
            x * voxel_size[2]
        };

        Point3<Float> const sample = voxel_origin + voxel_size * sample_local;

        // TODO: Determine 'occupancy' at the sample
        Float occupancy = 0.f;
        for (uint32_t f = 0; f < num_faces; ++f)
        {
            // Load the vertices and immediately compute `e` vectors
            Vector3<Float> e[3];
            for (int i = 0; i < 3; ++i)
                e[i] = normalize(Point3<Float>{
                                     vertices[3 * faces[3*f + i] + 0],
                                     vertices[3 * faces[3*f + i] + 1],
                                     vertices[3 * faces[3*f + i] + 2],
                                 } -
                                 sample);

            Float const alpha = dot(cross(e[0], e[1]), e[2]);
            Float const beta  = 1 + dot(e[0], e[1]) + dot(e[1], e[2]) + dot(e[2], e[0]);
            occupancy += atan2(alpha, beta);
        }

        // Splat the sample into the grid
        uint32_t filter_support[3] = {
            static_cast<uint32_t>(ceil(filter_radius / voxel_size[0])),
            static_cast<uint32_t>(ceil(filter_radius / voxel_size[1])),
            static_cast<uint32_t>(ceil(filter_radius / voxel_size[2])),
        };

        int32_t const min_z = static_cast<int32_t>(z) - filter_support[0];
        int32_t const max_z = static_cast<int32_t>(z) + filter_support[0];
        int32_t const min_y = static_cast<int32_t>(y) - filter_support[1];
        int32_t const max_y = static_cast<int32_t>(y) + filter_support[1];
        int32_t const min_x = static_cast<int32_t>(x) - filter_support[2];
        int32_t const max_x = static_cast<int32_t>(x) + filter_support[2];

        for (int32_t nz = std::max(min_z, 0); nz < std::min(static_cast<int32_t>(depth), max_z + 1); ++nz)
        {
            for (int32_t ny = std::max(min_y, 0); ny < std::min(static_cast<int32_t>(height), max_y + 1); ++ny)
            {
                for (int32_t nx = std::max(min_x, 0); nx < std::min(static_cast<int32_t>(width), max_x + 1); ++nx)
                {
                    uint64_t n_voxel_index = width * (height * nz + ny) + nx;
                    grid[n_voxel_index] += occupancy;
                }
            }
        }
    }
}

} // namespace dvx

template <typename Float>
void voxelize(nb::ndarray<Float, nb::c_contig> const &vertices,
              nb::ndarray<uint32_t, nb::c_contig> const &simplices,
              nb::ndarray<Float, nb::c_contig> &occupancy,
              uint32_t num_samples_per_voxel, Float filter_radius)
{
    if (vertices.ndim() != 2 || (vertices.shape(1) != 2 && vertices.shape(1) != 3))
        throw std::invalid_argument(format_message("Expected vertices with shape (n,2) or (n,3) but array has %d dimensions", vertices.ndim()));

    if (simplices.ndim() != 2 || (simplices.shape(1) != 2 && simplices.shape(1) != 3))
        throw std::invalid_argument(format_message("Expected simplices with shape (n,2) or (n,3) but array has %d dimensions", simplices.ndim()));

    if (vertices.shape(1) != simplices.shape(1))
        throw std::invalid_argument(format_message("Expected vertices (dim=%d) and simplices (dim=%d) with same dimension", vertices.shape(1), simplices.shape(1)));

    if (occupancy.ndim() != 2 && occupancy.ndim() != 3)
        throw std::invalid_argument(format_message("Expected occupancy grid with shape (h,w) or (d,h,w) but array has %d dimensions", occupancy.ndim()));

    check_on_device(vertices.device_type(), vertices.device_id(), vertices, simplices, occupancy);

    unsigned int N = vertices.shape(1);

    dvx::Filter<Float> filter{
        .type   = dvx::FilterType::Box,
        .radius = filter_radius};

    if (!suppress_warnings && num_samples_per_voxel < 16)
    {
        for (int i = 0; i < N; ++i)
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

    if (vertices.shape(1) == 2)
        dvx::voxelize_2d_cpu<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                    occupancy.data(), occupancy.shape(0), occupancy.shape(1), num_samples_per_voxel, filter);
    if (vertices.shape(1) == 3)
        dvx::voxelize_3d_cpu<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                    occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2), num_samples_per_voxel);
}

NB_MODULE(dvx_ext, m)
{
#if NDEBUG
    std::string build_type = "Release";
#else
    std::string build_type = "Debug";
#endif

    m.def("build_type", [=]() { return build_type; });
    m.def("mute", [=]() { suppress_warnings = true; });
    m.def("unmute", [=]() { suppress_warnings = false; });

    m.def("voxelize_f32", voxelize<float>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("num_samples_per_voxel"), nb::arg("filter_radius"));
}