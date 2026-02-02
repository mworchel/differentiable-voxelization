#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <type_traits>

#include "aabb.hpp"
#include "bitset.hpp"
#include "common.hpp"
#include "coverage.hpp"
#include "differentiation.hpp"
#include "flags.hpp"
#include "filter.hpp"
#include "grid.hpp"
#include "winding_number.hpp"

namespace dvx
{


enum class SamplingFlagBits : uint8_t
{
    None       = 0,
    Adaptive   = 1 << 0,
    Stratified = 1 << 1,
};

using SamplingFlags = typename std::underlying_type_t<SamplingFlagBits>;

DVX_DECLARE_FLAG_OPERATORS(SamplingFlags, SamplingFlagBits);

constexpr SamplingFlags DefaultSamplingFlags = SamplingFlagBits::Adaptive | SamplingFlagBits::Stratified;

struct MonteCarloParameters
{
    uint32_t      num_samples;
    SamplingFlags sampling_flags = DefaultSamplingFlags;
};

template<typename Float>
void voxelize_mc_2d(Float const* vertices, uint32_t const num_vertices,
                    uint32_t const* edges, uint32_t const num_edges,
                    Float* occupancy, Extent<2> const& grid,
                    MonteCarloParameters const& mc_params,
                    Filter<Float> const& filter)
{
    // TODO: Inject into this function
    Allocator& allocator = DefaultAllocator::instance();

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector2<Float> const voxel_size = get_voxel_size<Float>(grid);

    // Tag near-surface voxels
    Bitset mask;
    if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Adaptive))
    {
        create_bitset(&mask, grid.num_elements(), allocator);
        mark_boundary_voxels<Float, 2>(vertices, edges, num_edges,
                                       grid, filter.radius, &mask);
    }

    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    Float const voxel_volume = volume(voxel_size);

    uint32_t num_samples_per_voxel = mc_params.num_samples;

    uint32_t       num_samples_sqrt; // Used only by stratified sampling
    Vector2<Float> stratum_size;     //
    if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Stratified))
    {
        // Round to next square root (TODO: do somewhere else?)
        num_samples_sqrt      = MAYBE_STD(ceil)(MAYBE_STD(sqrt)(num_samples_per_voxel));
        num_samples_per_voxel = num_samples_sqrt * num_samples_sqrt;
        stratum_size          = voxel_size / num_samples_sqrt;
    }

    uint64_t num_samples_total = num_samples_per_voxel * grid.num_elements();
    for (uint64_t sample_index = 0; sample_index < num_samples_total; ++sample_index)
    {
        uint64_t const voxel_index = sample_index / num_samples_per_voxel;

        // Override the number of samples for this voxel to 1, if it is constant (not near the surface)
        uint32_t num_samples_this_voxel = num_samples_per_voxel;
        if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Adaptive))
        {
            bool const is_constant_voxel = !mask.is_set(voxel_index);
            if (is_constant_voxel)
            {
                num_samples_this_voxel = 1u;

                // Skip all but sample 0 for this voxel
                uint64_t const sample_lane = sample_index % num_samples_per_voxel;
                if (sample_lane > 0)
                    continue;
            }
        }

        Float const sample_weight = voxel_volume / num_samples_this_voxel;

        // Generate sample position in voxel-local space
        Point2<Float> const sample_local{
            distribution(engine),
            distribution(engine),
        };

        Point2<uint32_t> const voxel = linear_index_to_coords(voxel_index, grid);

        // Transform sample to global space
        // TODO: Optimize these expressions:
        // Point2<Float> const voxel_origin = Point2<Float>(voxel) * voxel_size - Float(1);
        Point2<Float> const voxel_origin{
            voxel.x() * voxel_size[0] - Float(1),
            voxel.y() * voxel_size[1] - Float(1)};

        Point2<Float> sample;
        if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Stratified) &&
            num_samples_this_voxel > 1 /* Disable stratified sampling for 1 sample voxels */)
        {
            uint64_t const         sample_lane = sample_index % num_samples_per_voxel;
            Point2<uint32_t> const stratum(
                /*x=*/sample_lane % num_samples_sqrt,
                /*y=*/sample_lane / num_samples_sqrt);
            // TODO: Optimize these expressions:
            // sample = voxel_origin + (stratum + sample_local) * stratum_size;
            sample = Point2<Float>{
                voxel_origin[0] + (stratum[0] + sample_local[0]) * stratum_size[0],
                voxel_origin[1] + (stratum[1] + sample_local[1]) * stratum_size[1],
            };
        }
        else
            sample = voxel_origin + voxel_size * sample_local;

        Float sample_occupancy = generalized_winding_number<Float, 2>(vertices, num_vertices,
                                                                      edges, num_edges,
                                                                      sample);

        // Splat the sample into the grid
        Point2<Float> const sample_grid_coord = Point2<Float>(voxel) + sample_local;

        int32_t min_coord[2];
        int32_t max_coord[2];
        get_grid_support<Float, 2>(sample_grid_coord, filter.radius / voxel_size, min_coord, max_coord);

        for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t>(grid.height()), max_coord[1] + 1); ++ny)
        {
            for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t>(grid.width()), max_coord[0] + 1); ++nx)
            {
                // Compute the weight using the distance from the sample to the voxel center
                Point2<int32_t> n_voxel(nx, ny);

                // TODO: Optimize such expression:
                // Point2<Float> const n_center = (Point2<Float>(n_voxel) + Float(0.5)) * voxel_size - Float(1);
                Point2<Float> const n_center{
                        (nx + Float(0.5)) * voxel_size[0] - Float(1),
                        (ny + Float(0.5)) * voxel_size[1] - Float(1)};

                Float const    filter_weight = filter.eval(n_center - sample);
                uint64_t const n_voxel_index = coords_to_linear_index(n_voxel, grid);
                if (filter_weight > 0)
                    occupancy[n_voxel_index] += filter_weight * sample_occupancy * sample_weight;
            }
        }
    }

    if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Adaptive))
        free_bitset(&mask, allocator);
}

// Forward-mode derivatives of the smooth indicator function
template<typename Float, DifferentiationMode Mode>
void d_voxelize_mc_2d(Float const* vertices, uint32_t const num_vertices,
                      uint32_t const* edges, uint32_t const num_edges,
                      Float* occupancy, uint32_t const height, uint32_t const width,
                      dIn<Float, Mode>*   d_vertices,
                      dOut<Float, Mode>*  d_occupancy,
                      uint32_t const      num_samples_per_simplex,
                      Filter<Float> const filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector2<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height};

    // Compute the boundary term (integration over edges)
    for (uint32_t e = 0; e < num_edges; ++e)
    {
        uint32_t const v0_idx = edges[2 * e + 0];
        uint32_t const v1_idx = edges[2 * e + 1];

        Point2<Float> const v0(&vertices[2 * v0_idx]);
        Point2<Float> const v1(&vertices[2 * v1_idx]);

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

            // Compute normal velocities for all vertices
            Vector2<Float> const d_xbn_d_v0 = normal * (1 - u);
            Vector2<Float> const d_xbn_d_v1 = normal * u;

            // Compute normal velocity for theta in forward mode
            [[maybe_unused]] Float d_xbn = 0; // Unused in backward
            if constexpr (Mode == DifferentiationMode::Forward)
            {
                // x_b = (1-u) * v0 + u * v1 = v0 + u * (v1 - v0)
                Vector2<Float> const d_v0(&d_vertices[2 * v0_idx]);
                Vector2<Float> const d_v1(&d_vertices[2 * v1_idx]);
                d_xbn += dot(d_v0, d_xbn_d_v0);
                d_xbn += dot(d_v1, d_xbn_d_v1);
            }

            Point2<Float> const sample            = v0 + u * v0v1;
            Point2<Float> const sample_grid_coord = point_to_grid_coord(sample, voxel_size);

            int32_t min_coord[2];
            int32_t max_coord[2];
            get_grid_support<Float, 2>(sample_grid_coord, filter.radius / voxel_size, min_coord, max_coord);

            for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t const>(height), max_coord[1] + 1); ++ny)
            {
                for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t const>(width), max_coord[0] + 1); ++nx)
                {
                    // Compute the weight using the distance from the sample to the voxel center
                    Point2<Float> const n_center{
                        (nx + Float(0.5)) * voxel_size[0] - Float(1),
                        (ny + Float(0.5)) * voxel_size[1] - Float(1)};

                    Float const    filter_weight = filter.eval(n_center - sample);
                    uint64_t const n_voxel_index = width * ny + nx;
                    if (filter_weight != Float(0))
                    {
                        if constexpr (Mode == DifferentiationMode::Forward)
                            d_occupancy[n_voxel_index] += d_xbn * filter_weight * area / Float(num_samples_per_simplex);
                        else // (Mode == DifferentiationMode::Backward)
                        {
                            Float const differential_weight = d_occupancy[n_voxel_index] * filter_weight * area / Float(num_samples_per_simplex);
                            for (int d = 0; d < 2; ++d)
                            {
                                d_vertices[2 * v0_idx + d] += d_xbn_d_v0[d] * differential_weight;
                                d_vertices[2 * v1_idx + d] += d_xbn_d_v1[d] * differential_weight;
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename Float>
void voxelize_mc_3d(Float const* vertices, uint32_t num_vertices,
                    uint32_t const* faces, uint32_t num_faces,
                    Float* occupancy, Extent<3> const& grid,
                    MonteCarloParameters const& mc_params,
                    Filter<Float> const& filter)
{
    // TODO: Inject into this function
    Allocator& allocator = DefaultAllocator::instance();

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector3<Float> const voxel_size = get_voxel_size<Float>(grid);

    // Tag near-surface voxels
    Bitset mask;
    if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Adaptive))
    {
        create_bitset(&mask, grid.num_elements(), allocator);
        mark_boundary_voxels<Float, 3>(vertices, faces, num_faces,
                                       grid, filter.radius, &mask);
    }

    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    Float const voxel_volume = volume(voxel_size);

    uint32_t num_samples_per_voxel = mc_params.num_samples;

    uint64_t num_samples_total = num_samples_per_voxel * grid.num_elements();
    for (uint64_t sample_index = 0; sample_index < num_samples_total; ++sample_index)
    {
        uint64_t const voxel_index = sample_index / num_samples_per_voxel;

        // Override the number of samples for this voxel to 1, if it is constant (not near the surface)
        uint32_t num_samples_this_voxel = num_samples_per_voxel;
        if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Adaptive))
        {
            bool const is_constant_voxel = !mask.is_set(voxel_index);
            if (is_constant_voxel)
            {
                num_samples_this_voxel = 1u;

                // Skip all but sample 0 for this voxel
                uint64_t const sample_lane = sample_index % num_samples_per_voxel;
                if (sample_lane > 0)
                    continue;
            }
        }

        Float const sample_weight = voxel_volume / num_samples_this_voxel;

        // Generate sample position in voxel-local space
        Point3<Float> const sample_local{
            distribution(engine),
            distribution(engine),
            distribution(engine)};

        Point3<uint32_t> const voxel = linear_index_to_coords(voxel_index, grid);

        // Transform sample to global space
        // TODO: Optimize these expressions:
        // Point3<Float> const voxel_origin = Point3<Float>(voxel) * voxel_size - Float(1);
        Point3<Float> const voxel_origin = {
            voxel.x() * voxel_size[0] - Float(1),
            voxel.y() * voxel_size[1] - Float(1),
            voxel.z() * voxel_size[2] - Float(1)};

        Point3<Float> const sample = voxel_origin + voxel_size * sample_local;

        Float sample_occupancy = generalized_winding_number<Float, 3>(vertices, num_vertices,
                                                                      faces, num_faces,
                                                                      sample);

        // Splat the sample into the grid
        Point3<Float> const sample_grid_coord = Point3<Float>(voxel) + sample_local;

        int32_t min_coord[3];
        int32_t max_coord[3];
        get_grid_support<Float, 3>(sample_grid_coord, filter.radius / voxel_size, min_coord, max_coord);

        for (int32_t nz = std::max(min_coord[2], 0); nz < std::min(static_cast<int32_t const>(grid.depth()), max_coord[2] + 1); ++nz)
        {
            for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t const>(grid.height()), max_coord[1] + 1); ++ny)
            {
                for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t const>(grid.width()), max_coord[0] + 1); ++nx)
                {
                    // Compute the weight using the distance from the sample to the voxel center
                    Point3<int32_t> n_voxel(nx, ny, nz);

                    // TODO: Optimize these expressions:
                    // Point3<Float> const n_center = (Point3<Float>(n_voxel) + Float(0.5)) * voxel_size - Float(1);
                    Point3<Float> const n_center{
                        (nx + Float(0.5)) * voxel_size[0] - Float(1),
                        (ny + Float(0.5)) * voxel_size[1] - Float(1),
                        (nz + Float(0.5)) * voxel_size[2] - Float(1)};

                    Float const    filter_weight = filter.eval(n_center - sample);
                    uint64_t const n_voxel_index = coords_to_linear_index(n_voxel, grid);
                    if (filter_weight > 0)
                        occupancy[n_voxel_index] += filter_weight * sample_occupancy * sample_weight;
                }
            }
        }
    }

    if (has_flag(mc_params.sampling_flags, SamplingFlagBits::Adaptive))
        free_bitset(&mask, allocator);
}

// Forward-mode derivatives of the smooth indicator function
template<typename Float, DifferentiationMode Mode>
void d_voxelize_mc_3d(Float const* vertices, uint32_t const num_vertices,
                      uint32_t const* faces, uint32_t const num_faces,
                      Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                      dIn<Float, Mode>*   d_vertices,
                      dOut<Float, Mode>*  d_occupancy,
                      uint32_t const      num_samples_per_simplex,
                      Filter<Float> const filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector3<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height,
        Float(2) / depth};

    // Compute the boundary term (integration over edges)
    for (uint32_t f = 0; f < num_faces; ++f)
    {
        uint32_t const v0_idx = faces[3 * f + 0];
        uint32_t const v1_idx = faces[3 * f + 1];
        uint32_t const v2_idx = faces[3 * f + 2];

        Point3<Float> const v0(&vertices[3 * v0_idx]);
        Point3<Float> const v1(&vertices[3 * v1_idx]);
        Point3<Float> const v2(&vertices[3 * v2_idx]);

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

            // Compute normal velocities for all vertices
            Vector3<Float> const d_xbn_d_v0 = normal * (1 - u - v);
            Vector3<Float> const d_xbn_d_v1 = normal * u;
            Vector3<Float> const d_xbn_d_v2 = normal * v;

            // Compute normal velocity for theta in forward mode
            [[maybe_unused]] Float d_xbn = 0; // Unused in backward
            if constexpr (Mode == DifferentiationMode::Forward)
            {
                // x_b = (1-u-v) * v0 + u * v1 + v * v2 = v0 + u * (v1 - v0) + v * (v2 - v0)
                Vector3<Float> const d_v0{d_vertices[3 * v0_idx + 0], d_vertices[3 * v0_idx + 1], d_vertices[3 * v0_idx + 2]};
                Vector3<Float> const d_v1{d_vertices[3 * v1_idx + 0], d_vertices[3 * v1_idx + 1], d_vertices[3 * v1_idx + 2]};
                Vector3<Float> const d_v2{d_vertices[3 * v2_idx + 0], d_vertices[3 * v2_idx + 1], d_vertices[3 * v2_idx + 2]};
                d_xbn += dot(d_v0, d_xbn_d_v0);
                d_xbn += dot(d_v1, d_xbn_d_v1);
                d_xbn += dot(d_v2, d_xbn_d_v2);
            }

            Point3<Float> const sample            = v0 + u * v0v1 + v * v0v2;
            Point3<Float> const sample_grid_coord = point_to_grid_coord(sample, voxel_size);

            int32_t min_coord[3];
            int32_t max_coord[3];
            get_grid_support<Float, 3>(sample_grid_coord, filter.radius / voxel_size, min_coord, max_coord);

            for (int32_t nz = std::max(min_coord[2], 0); nz < std::min(static_cast<int32_t const>(depth), max_coord[2] + 1); ++nz)
            {
                for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t const>(height), max_coord[1] + 1); ++ny)
                {
                    for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t const>(width), max_coord[0] + 1); ++nx)
                    {
                        // Compute the weight using the distance from the sample to the voxel center
                        Point3<Float> const n_center{
                            (nx + Float(0.5)) * voxel_size[0] - Float(1),
                            (ny + Float(0.5)) * voxel_size[1] - Float(1),
                            (nz + Float(0.5)) * voxel_size[2] - Float(1)};

                        Float const    filter_weight = filter.eval(n_center - sample);
                        uint64_t const n_voxel_index = width * (height * nz + ny) + nx;
                        if (filter_weight != Float(0))
                        {
                            if constexpr (Mode == DifferentiationMode::Forward)
                                d_occupancy[n_voxel_index] += d_xbn * filter_weight * area / Float(num_samples_per_simplex);
                            else // (Mode == DifferentiationMode::Backward)
                            {
                                Float const differential_weight = d_occupancy[n_voxel_index] * filter_weight * area / Float(num_samples_per_simplex);
                                for (int d = 0; d < 3; ++d)
                                {
                                    d_vertices[3 * v0_idx + d] += d_xbn_d_v0[d] * differential_weight;
                                    d_vertices[3 * v1_idx + d] += d_xbn_d_v1[d] * differential_weight;
                                    d_vertices[3 * v2_idx + d] += d_xbn_d_v2[d] * differential_weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace dvx