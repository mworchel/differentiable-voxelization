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
            bool is_inside = true;
            for (unsigned int i = 0; i < N; ++i)
                is_inside &= MAYBE_STD(abs)(difference[i]) < radius;
            return Float(is_inside) / volume<N>();
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

    template<unsigned int N>
    DEVICE Float volume()
    {        switch (type)
        {
        case FilterType::Box:
        {
            if constexpr (N == 2)
                return 4 * radius * radius;
            if constexpr (N == 3)
                return 8 * radius * radius * radius;
            return 0;
        }
        case FilterType::Gaussian:
        {
            return Float(1);
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
        Float const beta  = ((N == 2) ? dot(e[0], e[1]) : 1 + dot(e[0], e[1]) + dot(e[1], e[2]) + dot(e[2], e[0]));
        occupancy += atan2(alpha, beta);
    }

    Float normalization = (N == 2) ? 2 * M_PI : 4 * M_PI;

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

    Float const voxel_volume = voxel_size[0] * voxel_size[1];

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
                uint64_t const n_voxel_index = width * ny + nx;
                if (filter_weight > 0)
                    occupancy[n_voxel_index] += filter_weight * sample_occupancy * voxel_volume / num_samples_per_voxel;
            }
        }
    }
}

template<typename Float, unsigned int N>
Point<Float, N> point_to_grid_coord(Point<Float, N> const& x, Vector<Float, N> const& voxel_size)
{
    Point<Float, N> const grid_origin(-1);
    return (x - grid_origin) / voxel_size + Point<Float,N>(0);
}

// Forward-mode derivatives of the smooth indicator function
template<typename Float>
void voxelize_2d_forward_cpu(Float const* vertices, uint32_t num_vertices,
                             uint32_t const* edges, uint32_t num_edges,
                             Float* occupancy, uint32_t const height, uint32_t const width,
                             Float const* d_vertices,
                             Float* d_occupancy,
                             uint32_t num_samples_per_simplex, 
                             Filter<Float> filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector2<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height
    };

    // Compute the boundary term (integration over edges)
    for (uint32_t e = 0; e < num_edges; ++e)
    {
        uint32_t const v0_idx = edges[2*e + 0];
        uint32_t const v1_idx = edges[2*e + 1];

        Point2<Float> const v0{
            vertices[2*v0_idx + 0],
            vertices[2*v0_idx + 1],
        };

        Point2<Float> const v1{
            vertices[2*v1_idx + 0],
            vertices[2*v1_idx + 1],
        };

        Vector2<Float> const v0v1 = v1 - v0;

        // 90 degree rotation to the right
        Vector2<Float> const normal = normalize(Vector2<Float>{v0v1[1], -v0v1[0]});

        Float const area = norm(v0v1);

        // Monte Carlo integration over the edge
        // TODO: Stratified sampling?
        // TODO: Quadrature?
        for (uint32_t sample_index = 0; sample_index < num_samples_per_simplex; ++sample_index)
        {
            Float const u = distribution(engine);
            // x_b = (1-u) * v0 + u * v1 = v0 + u * (v1 - v0)
            Vector2<Float> const d_v0{d_vertices[2 * v0_idx + 0], d_vertices[2 * v0_idx + 1]};
            Vector2<Float> const d_v1{d_vertices[2 * v1_idx + 0], d_vertices[2 * v1_idx + 1]};

            Float const xb0 = dot(d_v0, normal * (1 - u));
            Float const xb1 = dot(d_v1, normal * u);

            Point2<Float> const sample            = v0 + u * v0v1;
            Point2<Float> const sample_grid_coord = point_to_grid_coord(sample, voxel_size);

            // TODO: Make function
            Float const filter_support[2] = {
                filter.radius / voxel_size[0],
                filter.radius / voxel_size[1]};

            int32_t const min_x = floor(sample_grid_coord[0] - filter_support[0]);
            int32_t const max_x = floor(sample_grid_coord[0] + filter_support[0]);
            int32_t const min_y = floor(sample_grid_coord[1] - filter_support[1]);
            int32_t const max_y = floor(sample_grid_coord[1] + filter_support[1]);

            for (int32_t ny = std::max(min_y, 0); ny < std::min(static_cast<int32_t const>(height), max_y + 1); ++ny)
            {
                for (int32_t nx = std::max(min_x, 0); nx < std::min(static_cast<int32_t const>(width), max_x + 1); ++nx)
                {
                    // Compute the weight using the distance from the sample to the voxel center
                    Point2<Float> const n_center{
                        (nx + Float(0.5)) * voxel_size[0] - Float(1),
                        (ny + Float(0.5)) * voxel_size[1] - Float(1)};

                    Float const filter_weight = filter.eval(n_center - sample);
                    uint64_t const n_voxel_index = width * ny + nx;
                    if (filter_weight != Float(0))
                    {
                        d_occupancy[n_voxel_index] += filter_weight * (xb0 + xb1) * area / Float(num_samples_per_simplex); // TODO: the normalization is incorrect
                    }
                }
            }
        }
    }
}

template<typename Float>
void voxelize_3d_cpu(Float const* vertices, uint32_t num_vertices,
                     uint32_t const* faces, uint32_t num_faces,
                     Float* occupancy, uint32_t depth, uint32_t height, uint32_t width,
                     uint32_t num_samples_per_voxel, 
                     Filter<Float> filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector3<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height,
        Float(2) / depth
    };

    Float const voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];

    uint64_t num_samples = num_samples_per_voxel * depth * height * width;
    for (uint64_t sample_index = 0; sample_index < num_samples; ++sample_index)
    {
        // Generate sample position in voxel-local space
        Point3<Float> const sample_local{
            distribution(engine),
            distribution(engine),
            distribution(engine)
        };

        // Transform sample to global space
        uint64_t const voxel_index = sample_index / num_samples_per_voxel;

        // Linear indexing: voxel_index = height*width*z + width*y + x
        uint32_t const z = voxel_index / (height * width);
        uint32_t const y = (voxel_index % (height * width)) / width;
        uint32_t const x = (voxel_index % (height * width)) % width;

        Point3<Float> const voxel_origin = {
            x * voxel_size[0] - Float(1),
            y * voxel_size[1] - Float(1),
            z * voxel_size[2] - Float(1)
        };

        Point3<Float> const sample = voxel_origin + voxel_size * sample_local;
        
        Float sample_occupancy = generalized_winding_number<Float, 3>(vertices, num_vertices, 
                                                                      faces, num_faces, 
                                                                      sample);
            
        // Splat the sample into the grid
        Point3<Float> const sample_grid_coord = Point3<Float>(Float(x), Float(y), Float(z)) + sample_local;

        Float filter_support[] = {
            filter.radius / voxel_size[0],
            filter.radius / voxel_size[1],
            filter.radius / voxel_size[2],
        };

        int32_t const min_x = floor(sample_grid_coord[0] - filter_support[0]);
        int32_t const max_x = floor(sample_grid_coord[0] + filter_support[0]);
        int32_t const min_y = floor(sample_grid_coord[1] - filter_support[1]);
        int32_t const max_y = floor(sample_grid_coord[1] + filter_support[1]);
        int32_t const min_z = floor(sample_grid_coord[2] - filter_support[2]);
        int32_t const max_z = floor(sample_grid_coord[2] + filter_support[2]);

        for (int32_t nz = std::max(min_z, 0); nz < std::min(static_cast<int32_t const>(depth), max_z + 1); ++nz)
        {
            for (int32_t ny = std::max(min_y, 0); ny < std::min(static_cast<int32_t const>(height), max_y + 1); ++ny)
            {
                for (int32_t nx = std::max(min_x, 0); nx < std::min(static_cast<int32_t const>(width), max_x + 1); ++nx)
                {
                    // Compute the weight using the distance from the sample to the voxel center
                    Point3<Float> const n_center{
                        (nx + Float(0.5)) * voxel_size[0] - Float(1),
                        (ny + Float(0.5)) * voxel_size[1] - Float(1),
                        (nz + Float(0.5)) * voxel_size[2] - Float(1)
                    };

                    Float const filter_weight = filter.eval(n_center - sample);
                    uint64_t const n_voxel_index = width * (height * nz + ny) + nx;
                    if (filter_weight > 0)
                        occupancy[n_voxel_index] += filter_weight * sample_occupancy * voxel_volume / num_samples_per_voxel;
                }
            }
        }
    }
}

// Forward-mode derivatives of the smooth indicator function
template<typename Float>
void voxelize_3d_forward_cpu(Float const* vertices, uint32_t num_vertices,
                             uint32_t const* faces, uint32_t num_faces,
                             Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                             Float const* d_vertices,
                             Float* d_occupancy,
                             uint32_t num_samples_per_simplex, 
                             Filter<Float> filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector3<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height,
        Float(2) / depth
    };

    // Compute the boundary term (integration over edges)
    for (uint32_t f = 0; f < num_faces; ++f)
    {
        uint32_t const v0_idx = faces[3*f + 0];
        uint32_t const v1_idx = faces[3*f + 1];
        uint32_t const v2_idx = faces[3*f + 2];

        Point3<Float> const v0{
            vertices[3*v0_idx + 0],
            vertices[3*v0_idx + 1],
            vertices[3*v0_idx + 2],
        };

        Point3<Float> const v1{
            vertices[3*v1_idx + 0],
            vertices[3*v1_idx + 1],
            vertices[3*v1_idx + 2],
        };

        Point3<Float> const v2{
            vertices[3*v2_idx + 0],
            vertices[3*v2_idx + 1],
            vertices[3*v2_idx + 2],
        };

        Vector3<Float> const v0v1 = v1 - v0;
        Vector3<Float> const v0v2 = v2 - v0;

        Vector3<Float> const cross_e1e2 = cross(v0v1, v0v2);
        Vector3<Float> const normal     = normalize(cross_e1e2);
        Float const          area       = Float(0.5) * norm(cross_e1e2);

        // Monte Carlo integration over the edge
        // TODO: Stratified sampling?
        // TODO: Quadrature?
        for (uint32_t sample_index = 0; sample_index < num_samples_per_simplex; ++sample_index)
        {
            Float u = distribution(engine);
            Float v = distribution(engine);
            if (u + v > Float(1))
            {
                u = 1 - u;
                v = 1 - v;
            }
            // x_b = (1-u-v) * v0 + u * v1 + v * v2 = v0 + u * (v1 - v0) + v * (v2 - v0)
            Vector3<Float> const d_v0{d_vertices[3 * v0_idx + 0], d_vertices[3 * v0_idx + 1], d_vertices[3 * v0_idx + 2]};
            Vector3<Float> const d_v1{d_vertices[3 * v1_idx + 0], d_vertices[3 * v1_idx + 1], d_vertices[3 * v1_idx + 2]};
            Vector3<Float> const d_v2{d_vertices[3 * v2_idx + 0], d_vertices[3 * v2_idx + 1], d_vertices[3 * v2_idx + 2]};

            Float const xb0 = dot(d_v0, normal * (1 - u - v));
            Float const xb1 = dot(d_v1, normal * u);
            Float const xb2 = dot(d_v2, normal * v);

            Point3<Float> const sample            = v0 + u * v0v1 + v * v0v2;
            Point3<Float> const sample_grid_coord = point_to_grid_coord(sample, voxel_size);

            // TODO: Make function
            Float const filter_support[3] = {
                filter.radius / voxel_size[0],
                filter.radius / voxel_size[1],
                filter.radius / voxel_size[2]
            };

            int32_t const min_x = floor(sample_grid_coord[0] - filter_support[0]);
            int32_t const max_x = floor(sample_grid_coord[0] + filter_support[0]);
            int32_t const min_y = floor(sample_grid_coord[1] - filter_support[1]);
            int32_t const max_y = floor(sample_grid_coord[1] + filter_support[1]);
            int32_t const min_z = floor(sample_grid_coord[2] - filter_support[2]);
            int32_t const max_z = floor(sample_grid_coord[2] + filter_support[2]);

            for (int32_t nz = std::max(min_z, 0); nz < std::min(static_cast<int32_t const>(depth), max_z + 1); ++nz)
            {
                for (int32_t ny = std::max(min_y, 0); ny < std::min(static_cast<int32_t const>(height), max_y + 1); ++ny)
                {
                    for (int32_t nx = std::max(min_x, 0); nx < std::min(static_cast<int32_t const>(width), max_x + 1); ++nx)
                    {
                        // Compute the weight using the distance from the sample to the voxel center
                        Point3<Float> const n_center{
                            (nx + Float(0.5)) * voxel_size[0] - Float(1),
                            (ny + Float(0.5)) * voxel_size[1] - Float(1),
                            (nz + Float(0.5)) * voxel_size[2] - Float(1)
                        };

                        Float const filter_weight = filter.eval(n_center - sample);
                        uint64_t const n_voxel_index = width * (height * nz + ny) + nx;
                        if (filter_weight != Float(0))
                        {
                            d_occupancy[n_voxel_index] += filter_weight * (xb0 + xb1 + xb2) * area / Float(num_samples_per_simplex);
                        }
                    }
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
                                    occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2), num_samples_per_voxel, filter);
}

template <typename Float>
void voxelize_forward(nb::ndarray<Float, nb::c_contig> const &vertices,
                      nb::ndarray<uint32_t, nb::c_contig> const &simplices,
                      nb::ndarray<Float, nb::c_contig> &occupancy,
                      nb::ndarray<Float, nb::c_contig> const &d_vertices,
                      nb::ndarray<Float, nb::c_contig> &d_occupancy,
                      uint32_t num_samples_per_simplex, Float filter_radius)
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

    if (vertices.shape(1) == 2)
        dvx::voxelize_2d_forward_cpu<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                            occupancy.data(), occupancy.shape(0), occupancy.shape(1), 
                                            d_vertices.data(), d_occupancy.data(),
                                            num_samples_per_simplex, filter);
    if (vertices.shape(1) == 3)
        dvx::voxelize_3d_forward_cpu<Float>(vertices.data(), vertices.shape(0), simplices.data(), simplices.shape(0),
                                            occupancy.data(), occupancy.shape(0), occupancy.shape(1), occupancy.shape(2),
                                            d_vertices.data(), d_occupancy.data(),
                                            num_samples_per_simplex, filter);
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
    m.def("voxelize_forward_f32", voxelize_forward<float>, nb::arg("vertices"), nb::arg("simplices"), nb::arg("occupancy"), nb::arg("d_vertices"), nb::arg("d_occupancy"), nb::arg("num_samples_per_simplex"), nb::arg("filter_radius"));
}