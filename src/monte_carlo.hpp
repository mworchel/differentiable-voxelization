#pragma once

#include <algorithm>
#include <cstdint>
#include <random>

#include "aabb.hpp"
#include "common.hpp"
#include "filter.hpp"
#include "grid.hpp"
#include "winding_number.hpp"

namespace dvx
{

template<typename Float>
void voxelize_mc_2d(Float const* vertices, uint32_t const num_vertices,
                    uint32_t const* edges, uint32_t const num_edges,
                    Float* occupancy, uint32_t const height, uint32_t const width,
                    uint32_t const      num_samples_per_voxel,
                    Filter<Float> const filter)
{
    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector2<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height};

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
            y * voxel_size[1] - Float(1)};

        Point2<Float> const sample = voxel_origin + voxel_size * sample_local;

        Float sample_occupancy = generalized_winding_number<Float, 2>(vertices, num_vertices,
                                                                      edges, num_edges,
                                                                      sample);

        // Splat the sample into the grid
        Point2<Float> const sample_grid_coord = Point2<Float>(Float(x), Float(y)) + sample_local;

        int32_t min_coord[2];
        int32_t max_coord[2];
        get_grid_support<Float, 2>(sample_grid_coord, filter.radius / voxel_size, min_coord, max_coord);

        for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t>(height), max_coord[1] + 1); ++ny)
        {
            for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t>(width), max_coord[0] + 1); ++nx)
            {
                // Compute the weight using the distance from the sample to the voxel center
                Point2<Float> const n_center{
                    (nx + Float(0.5)) * voxel_size[0] - Float(1),
                    (ny + Float(0.5)) * voxel_size[1] - Float(1)};

                Float const    filter_weight = filter.eval(n_center - sample);
                uint64_t const n_voxel_index = width * ny + nx;
                if (filter_weight > 0)
                    occupancy[n_voxel_index] += filter_weight * sample_occupancy * voxel_volume / num_samples_per_voxel;
            }
        }
    }
}

// Forward-mode derivatives of the smooth indicator function
template<typename Float>
void voxelize_forward_mc_2d(Float const* vertices, uint32_t const num_vertices,
                            uint32_t const* edges, uint32_t const num_edges,
                            Float* occupancy, uint32_t const height, uint32_t const width,
                            Float const*        d_vertices,
                            Float*              d_occupancy,
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
            // x_b = (1-u) * v0 + u * v1 = v0 + u * (v1 - v0)
            Vector2<Float> const d_v0(&d_vertices[2 * v0_idx]);
            Vector2<Float> const d_v1(&d_vertices[2 * v1_idx]);

            Float const xb0 = dot(d_v0, normal * (1 - u));
            Float const xb1 = dot(d_v1, normal * u);

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
                        d_occupancy[n_voxel_index] += filter_weight * (xb0 + xb1) * area / Float(num_samples_per_simplex); // TODO: the normalization is incorrect
                    }
                }
            }
        }
    }
}

#define DVX_MC_PRIMAL_ADAPTIVE_SAMPLING 1

template<typename Float>
void voxelize_mc_3d(Float const* vertices, uint32_t num_vertices,
                    uint32_t const* faces, uint32_t num_faces,
                    Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                    uint32_t const      num_samples_per_voxel,
                    Filter<Float> const filter)
{
    // TODO: Lift assumption of grid being in [-1,1]^3
    Vector3<Float> const voxel_size{
        Float(2) / width,
        Float(2) / height,
        Float(2) / depth};

#if DVX_MC_PRIMAL_ADAPTIVE_SAMPLING
    // Tag near-surface voxels using the axis-aligned bounding box
    Bitset mask(depth * height * width);
    for (uint32_t face_index = 0; face_index < num_faces; ++face_index)
    {
        Point3<uint32_t> const face(&faces[3 * face_index]);

        Point3<Float> const v0(&vertices[3 * face[0]]);
        Point3<Float> const v1(&vertices[3 * face[1]]);
        Point3<Float> const v2(&vertices[3 * face[2]]);

        AABB3<Float> aabb;
        aabb.extend(v0);
        aabb.extend(v1);
        aabb.extend(v2);

        //
        aabb.min -= Vector3<Float>(filter.radius);
        aabb.max += Vector3<Float>(filter.radius);

        Point3<Float> const min_grid_coord = point_to_grid_coord(aabb.min, voxel_size);
        Point3<Float> const max_grid_coord = point_to_grid_coord(aabb.max, voxel_size);

        int32_t const min_x = floor(min_grid_coord[0]);
        int32_t const max_x = floor(max_grid_coord[0]);
        int32_t const min_y = floor(min_grid_coord[1]);
        int32_t const max_y = floor(max_grid_coord[1]);
        int32_t const min_z = floor(min_grid_coord[2]);
        int32_t const max_z = floor(max_grid_coord[2]);

        for (int32_t nz = std::max(min_z, 0); nz < std::min(static_cast<int32_t const>(depth), max_z + 1); ++nz)
        {
            for (int32_t ny = std::max(min_y, 0); ny < std::min(static_cast<int32_t const>(height), max_y + 1); ++ny)
            {
                for (int32_t nx = std::max(min_x, 0); nx < std::min(static_cast<int32_t const>(width), max_x + 1); ++nx)
                {
                    uint64_t const n_voxel_index = width * (height * nz + ny) + nx;
                    mask.set(n_voxel_index);
                    // occupancy[n_voxel_index] = Float(1);
                }
            }
        }
    }
#endif

    std::default_random_engine            engine(std::random_device{}());
    std::uniform_real_distribution<Float> distribution;

    Float const voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];

    uint64_t num_samples = num_samples_per_voxel * depth * height * width;
    for (uint64_t sample_index = 0; sample_index < num_samples; ++sample_index)
    {
        // Transform sample to global space
        uint64_t const voxel_index = sample_index / num_samples_per_voxel;

#if DVX_MC_PRIMAL_ADAPTIVE_SAMPLING
        // Only use a single sample for voxels that are not near the surface
        bool const     is_constant_voxel      = !mask.is_set(voxel_index);
        uint32_t const num_samples_this_voxel = is_constant_voxel ? 1u : num_samples_per_voxel;
        if (is_constant_voxel && (sample_index % num_samples_per_voxel) > 0)
            continue;
#else
        uint32_t const num_samples_this_voxel = num_samples_per_voxel;
#endif

        // Generate sample position in voxel-local space
        Point3<Float> const sample_local{
            distribution(engine),
            distribution(engine),
            distribution(engine)};

        // Linear indexing: voxel_index = height*width*z + width*y + x
        uint32_t const z = voxel_index / (height * width);
        uint32_t const y = (voxel_index % (height * width)) / width;
        uint32_t const x = (voxel_index % (height * width)) % width;

        Point3<Float> const voxel_origin = {
            x * voxel_size[0] - Float(1),
            y * voxel_size[1] - Float(1),
            z * voxel_size[2] - Float(1)};

        Point3<Float> const sample = voxel_origin + voxel_size * sample_local;

        Float sample_occupancy = generalized_winding_number<Float, 3>(vertices, num_vertices,
                                                                      faces, num_faces,
                                                                      sample);

        // Splat the sample into the grid
        Point3<Float> const sample_grid_coord = Point3<Float>(Float(x), Float(y), Float(z)) + sample_local;

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
                    if (filter_weight > 0)
                        occupancy[n_voxel_index] += filter_weight * sample_occupancy * voxel_volume / num_samples_this_voxel;
                }
            }
        }
    }
}

// Forward-mode derivatives of the smooth indicator function
template<typename Float>
void voxelize_forward_mc_3d(Float const* vertices, uint32_t const num_vertices,
                            uint32_t const* faces, uint32_t const num_faces,
                            Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                            Float const*        d_vertices,
                            Float*              d_occupancy,
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
            // x_b = (1-u-v) * v0 + u * v1 + v * v2 = v0 + u * (v1 - v0) + v * (v2 - v0)
            Vector3<Float> const d_v0{d_vertices[3 * v0_idx + 0], d_vertices[3 * v0_idx + 1], d_vertices[3 * v0_idx + 2]};
            Vector3<Float> const d_v1{d_vertices[3 * v1_idx + 0], d_vertices[3 * v1_idx + 1], d_vertices[3 * v1_idx + 2]};
            Vector3<Float> const d_v2{d_vertices[3 * v2_idx + 0], d_vertices[3 * v2_idx + 1], d_vertices[3 * v2_idx + 2]};

            Float const xb0 = dot(d_v0, normal * (1 - u - v));
            Float const xb1 = dot(d_v1, normal * u);
            Float const xb2 = dot(d_v2, normal * v);

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
                            d_occupancy[n_voxel_index] += filter_weight * (xb0 + xb1 + xb2) * area / Float(num_samples_per_simplex);
                        }
                    }
                }
            }
        }
    }
}

} // namespace dvx