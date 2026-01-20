#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <nanobind/nanobind.h>
#include <vector>

#include <string>

#include "common.hpp"
#include "math.hpp"
#include "differentiation.hpp"

namespace nb = nanobind;
namespace dvx
{

///////////////////////////////////////////////////////////////////////////////
// 2d Case
///////////////////////////////////////////////////////////////////////////////

// Split a segment (in barycentric coords) by a Cartesian line
// Barycentric coords for segment: b = (b0, b1) where point = b0*v0 + b1*v1, b0+b1=1
// Returns (below_or_equal, above) segments as (t_lo, t_hi) ranges where t = b1
template<typename Float>
void split_segment_bary(Float t_lo, Float t_hi,
                        Vector<Float, 2> const& edge_coords,
                        Float                   value,
                        Float&                  below_lo, Float& below_hi,
                        Float&                  above_lo, Float& above_hi,
                        bool&                   has_below, bool& has_above)
{
    has_below = has_above = false;
    if (t_lo >= t_hi)
        return;

    // Cartesian coord at t: (1-t)*edge_coords[0] + t*edge_coords[1]
    Float v_lo = (Float(1) - t_lo) * edge_coords[0] + t_lo * edge_coords[1];
    Float v_hi = (Float(1) - t_hi) * edge_coords[0] + t_hi * edge_coords[1];
    bool  b_lo = v_lo <= value;
    bool  b_hi = v_hi <= value;

    if (b_lo && b_hi)
    {
        has_below = true;
        below_lo  = t_lo;
        below_hi  = t_hi;
    }
    else if (!b_lo && !b_hi)
    {
        has_above = true;
        above_lo  = t_lo;
        above_hi  = t_hi;
    }
    else
    {
        Float d = v_hi - v_lo;
        if (std::abs(d) > Float(1e-12))
        {
            Float t_mid = t_lo + (value - v_lo) / d * (t_hi - t_lo);
            if (b_lo)
            {
                has_below = true;
                below_lo  = t_lo;
                below_hi  = t_mid;
                has_above = true;
                above_lo  = t_mid;
                above_hi  = t_hi;
            }
            else
            {
                has_above = true;
                above_lo  = t_lo;
                above_hi  = t_mid;
                has_below = true;
                below_lo  = t_mid;
                below_hi  = t_hi;
            }
        }
    }
}

// 2D Explicit Voxelization (edges/segments)
template<typename Float, bool primal, DifferentiationMode Mode>
void voxelize_cf_2d_generalized(Float const* vertices, uint32_t const num_vertices,
                                      uint32_t const* edges, uint32_t const num_edges,
                                      Float* occupancy, uint32_t const height, uint32_t const width,
                                      dIn<Float, Mode>*  d_vertices,
                                      dOut<Float, Mode>* d_occupancy)
{
    const Float step_x    = Float(2) / height;
    const Float step_y    = Float(2) / width;
    const Float cell_area = step_x * step_y;

    // Center of the grid in y (for center-based optimization)
    const int center_j = static_cast<int>(width) / 2;

    for (uint32_t edge_idx = 0; edge_idx < num_edges; ++edge_idx)
    {
        const uint32_t i0 = edges[edge_idx * 2 + 0];
        const uint32_t i1 = edges[edge_idx * 2 + 1];

        const Vector<Float, 2> v0(&vertices[i0 * 2]);
        const Vector<Float, 2> v1(&vertices[i1 * 2]);

        const Vector<Float, 2> edge_vec = v1 - v0;
        const Float            edge_len = norm(edge_vec);

        if (edge_len < Float(1e-12))
            continue;

        // Normal perpendicular to edge: (dy, -dx)
        const Vector<Float, 2> normal(edge_vec.y(), -edge_vec.x());

        // for primal only
        Float sign_ny;
        // for differential only
        Vector<Float, 2> dvn_vec;
        Vector<Float, 2> unit_normal;

        if constexpr (primal)
        {
            // Only care about y-component of normal for primal
            if (std::abs(normal.y()) < Float(1e-12))
                continue;

            sign_ny = (normal.y() > 0) ? Float(1) : Float(-1);
        }
        else
        {
            unit_normal = normal / edge_len;

            if constexpr (Mode == DifferentiationMode::Backward)
            {
                dvn_vec[0] = unit_normal[0];
                dvn_vec[1] = unit_normal[1];
            }
            else
            {
                const Vector<Float, 2> dv0(&d_vertices[i0 * 2]);
                const Vector<Float, 2> dv1(&d_vertices[i1 * 2]);
                dvn_vec[0] = dot(dv0, unit_normal);
                dvn_vec[1] = dot(dv1, unit_normal);
            }
        }

        // Bounding box in grid coordinates
        const Float min_x = std::min(v0.x(), v1.x());
        const Float min_y = std::min(v0.y(), v1.y());
        const Float max_x = std::max(v0.x(), v1.x());
        const Float max_y = std::max(v0.y(), v1.y());

        const int bbox_min_i = std::max(0, static_cast<int>((min_x + 1) / step_x));
        const int bbox_min_j = std::max(0, static_cast<int>((min_y + 1) / step_y));
        const int bbox_max_i = std::min(static_cast<int>(height) - 1, static_cast<int>((max_x + 1) / step_x));
        const int bbox_max_j = std::min(static_cast<int>(width) - 1, static_cast<int>((max_y + 1) / step_y));

        // Edge coordinates per axis
        const Vector<Float, 2> edge_x(v0.x(), v1.x());
        const Vector<Float, 2> edge_y(v0.y(), v1.y());

        // Start with full segment: t in [0, 1]
        Float remaining_lo = Float(0), remaining_hi = Float(1);

        for (int i = bbox_min_i; i <= bbox_max_i; ++i)
        {
            if (remaining_lo >= remaining_hi)
                break;

            const Float x_hi = Float(-1) + (i + 1) * step_x;

            Float seg_lo, seg_hi, above_lo, above_hi;
            bool  has_seg, has_above;
            split_segment_bary(remaining_lo, remaining_hi, edge_x, x_hi,
                               seg_lo, seg_hi, above_lo, above_hi, has_seg, has_above);

            if (has_above)
            {
                remaining_lo = above_lo;
                remaining_hi = above_hi;
            }
            else
            {
                remaining_lo = remaining_hi;
            }

            if (!has_seg)
                continue;

            // Get y bounds for this x-slice
            Float y_lo_slice = (Float(1) - seg_lo) * v0.y() + seg_lo * v1.y();
            Float y_hi_slice = (Float(1) - seg_hi) * v0.y() + seg_hi * v1.y();
            if (y_lo_slice > y_hi_slice)
                std::swap(y_lo_slice, y_hi_slice);

            const int j_min = std::max(bbox_min_j, static_cast<int>((y_lo_slice + 1) / step_y));
            const int j_max = std::min(bbox_max_j, static_cast<int>((y_hi_slice + 1) / step_y));

            Float remaining_y_lo = seg_lo, remaining_y_hi = seg_hi;

            for (int j = j_min; j <= j_max; ++j)
            {
                if (remaining_y_lo >= remaining_y_hi)
                    break;

                const Float y_hi = Float(-1) + (j + 1) * step_y;

                Float cell_lo, cell_hi, ay_lo, ay_hi;
                bool  has_cell, has_ay;
                split_segment_bary(remaining_y_lo, remaining_y_hi, edge_y, y_hi,
                                   cell_lo, cell_hi, ay_lo, ay_hi, has_cell, has_ay);

                if (has_ay)
                {
                    remaining_y_lo = ay_lo;
                    remaining_y_hi = ay_hi;
                }
                else
                {
                    remaining_y_lo = remaining_y_hi;
                }

                if (!has_cell)
                    continue;

                // Compute clipped segment endpoints
                const Vector<Float, 2> p0 = v0 * (Float(1) - cell_lo) + v1 * cell_lo;
                const Vector<Float, 2> p1 = v0 * (Float(1) - cell_hi) + v1 * cell_hi;
                const Float            seg_len = norm(p1 - p0);

                if (seg_len < Float(1e-12))
                    continue;

                const size_t idx = static_cast<size_t>(i * width + j);

                if constexpr (primal)
                {
                    // Projected length in x direction
                    const Float len_x = std::abs(p1.x() - p0.x());
                    if (len_x < Float(1e-12))
                        continue;

                    const bool  below_center = y_hi <= step_y / 2;
                    const Float sign         = below_center ? -sign_ny : sign_ny;
                    const int   start        = below_center ? j + 1 : center_j;
                    const int   end          = below_center ? center_j : j;
                    const Float y_ref        = below_center ? y_hi : y_hi - step_y;

                    // Full contribution to cells toward center
                    const Float contribution = sign * len_x / step_x;
                    for (int jj = start; jj < end; ++jj)
                    {
                        const size_t full_idx = static_cast<size_t>(i * width + jj);
                        occupancy[full_idx] += contribution;
                    }

                    // Partial contribution for this cell
                    const Float avg_y   = (p0.y() + p1.y()) / Float(2);
                    const Float contrib = len_x * (avg_y - y_ref);
                    occupancy[idx] += sign_ny * contrib / cell_area;
                }
                else
                {
                    // Average barycentric coord (midpoint of segment)
                    const Float b0 = Float(1) - (cell_lo + cell_hi) / Float(2);
                    const Float b1 = (cell_lo + cell_hi) / Float(2);

                    if constexpr (Mode == DifferentiationMode::Forward)
                    {
                        const Float avg_dvn = b0 * dvn_vec[0] + b1 * dvn_vec[1];
                        d_occupancy[idx] += avg_dvn * seg_len / cell_area;
                    }
                    else
                    {
                        const Float w = d_occupancy[idx] * seg_len / cell_area;
                        d_vertices[i0 * 2 + 0] += b0 * dvn_vec[0] * w;
                        d_vertices[i0 * 2 + 1] += b0 * dvn_vec[1] * w;
                        d_vertices[i1 * 2 + 0] += b1 * dvn_vec[0] * w;
                        d_vertices[i1 * 2 + 1] += b1 * dvn_vec[1] * w;
                    }
                }
            }
        }
    }
}

// 2D wrappers
template<typename Float>
void voxelize_cf_2d(Float const* vertices, uint32_t const num_vertices,
                          uint32_t const* edges, uint32_t const num_edges,
                          Float* occupancy, uint32_t const height, uint32_t const width)
{
    return voxelize_cf_2d_generalized<Float, true, DifferentiationMode::Forward>(
        vertices, num_vertices, edges, num_edges, occupancy, height, width, nullptr, nullptr);
}

template<typename Float>
void voxelize_cf_2d_forward(Float const* vertices, uint32_t const num_vertices,
                                  uint32_t const* edges, uint32_t const num_edges,
                                  Float* occupancy, uint32_t const height, uint32_t const width,
                                  Float const* d_vertices, Float* d_occupancy)
{
    return voxelize_cf_2d_generalized<Float, false, DifferentiationMode::Forward>(
        vertices, num_vertices, edges, num_edges, occupancy, height, width, d_vertices, d_occupancy);
}

template<typename Float>
void voxelize_cf_2d_backward(Float const* vertices, uint32_t const num_vertices,
                                   uint32_t const* edges, uint32_t const num_edges,
                                   Float* occupancy, uint32_t const height, uint32_t const width,
                                   Float* d_vertices, Float const* d_occupancy)
{
    return voxelize_cf_2d_generalized<Float, false, DifferentiationMode::Backward>(
        vertices, num_vertices, edges, num_edges, occupancy, height, width, d_vertices, d_occupancy);
}


///////////////////////////////////////////////////////////////////////////////
// 3d Case
///////////////////////////////////////////////////////////////////////////////

// Compute area of a polygon using shoelace formula
template<typename Float>
Float polygon_area_2d(std::vector<Vector<Float, 3>> const& vertices)
{
    if (vertices.size() < 3)
        return Float(0);

    Float  area = Float(0);
    size_t n    = vertices.size();
    for (size_t i = 0; i < n; ++i)
    {
        size_t j = (i + 1) % n;
        area += vertices[i].x() * vertices[j].y();
        area -= vertices[j].x() * vertices[i].y();
    }
    return std::abs(area) / Float(2);
}

// Split a polygon (in barycentric coords) by a Cartesian plane
// Returns (below_or_equal, above) polygons in barycentric coordinates
template<typename Float>
void split_polygon_bary(std::vector<Vector<Float, 3>> const& polygon_bary,
                        Vector<Float, 3> const&              tri_coords,
                        Float                                value,
                        std::vector<Vector<Float, 3>>&       below,
                        std::vector<Vector<Float, 3>>&       above)
{
    below.clear();
    above.clear();

    if (polygon_bary.size() < 3)
        return;

    const size_t n = polygon_bary.size();

    for (size_t i = 0; i < n; ++i)
    {
        Vector<Float, 3> const& curr_bary = polygon_bary[i];
        Vector<Float, 3> const& next_bary = polygon_bary[(i + 1) % n];

        // Get Cartesian coord for this axis via dot product
        Float curr_val = dot(curr_bary, tri_coords);
        Float next_val = dot(next_bary, tri_coords);

        const bool curr_below = curr_val <= value;
        const bool next_below = next_val <= value;

        if (curr_below)
            below.push_back(curr_bary);
        else
            above.push_back(curr_bary);

        if (curr_below != next_below)
        {
            // Compute intersection in barycentric space
            const Float d = next_val - curr_val;
            if (std::abs(d) > Float(1e-12))
            {
                const Float      t                 = (value - curr_val) / d;
                Vector<Float, 3> intersection_bary = curr_bary + t * (next_bary - curr_bary);
                below.push_back(intersection_bary);
                above.push_back(intersection_bary);
            }
        }
    }
}

template<typename Float, bool primal, DifferentiationMode Mode>
void voxelize_cf_generalized(Float const* vertices, uint32_t const num_vertices,
                                   uint32_t const* faces, uint32_t const num_faces,
                                   Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                                   dIn<Float, Mode>*   d_vertices,
                                   dOut<Float, Mode>*  d_occupancy)
{
    const Float step_x      = Float(2) / depth;
    const Float step_y      = Float(2) / height;
    const Float step_z      = Float(2) / width;
    const Float cell_volume = step_x * step_y * step_z;

    // Center of the grid in z (for center-based optimization)
    const int center_k = static_cast<int>(width) / 2;

    // Temporary storage for polygon splitting
    std::vector<Vector<Float, 3>> remaining_x, poly_x, temp_below, temp_above;
    std::vector<Vector<Float, 3>> remaining_y, poly_xy;
    std::vector<Vector<Float, 3>> remaining_z, poly_bary;
    std::vector<Vector<Float, 3>> poly_xyz;

    // for non primal only
    std::vector<Float> poly_dvn;

    for (uint32_t face_idx = 0; face_idx < num_faces; ++face_idx)
    {
        const uint32_t i0 = faces[face_idx * 3 + 0];
        const uint32_t i1 = faces[face_idx * 3 + 1];
        const uint32_t i2 = faces[face_idx * 3 + 2];

        const Vector<Float, 3> v0(&vertices[i0 * 3]);
        const Vector<Float, 3> v1(&vertices[i1 * 3]);
        const Vector<Float, 3> v2(&vertices[i2 * 3]);

        const Vector<Float, 3> e1 = v1 - v0;
        const Vector<Float, 3> e2 = v2 - v0;

        // for primal only
        Float sign_nz;
        // for differential only
        Vector<Float, 3> dvn_vec;
        Vector<Float, 3> normal;

        if constexpr (primal)
        {
            const Float normal_z = e1.x() * e2.y() - e1.y() * e2.x();

            if (std::abs(normal_z) < Float(1e-12))
                continue;

            sign_nz = (normal_z > 0) ? Float(1) : Float(-1);
        }
        else
        {
            // Compute normal
            normal = cross(e1, e2);

            const Float normal_len = norm(normal);

            // Area of triangle is zero, skip
            if (normal_len < Float(1e-12))
                continue;

            const Vector<Float, 3> unit_normal = normal / normal_len;

            const Vector<Float, 3> dv0(&d_vertices[i0 * 3]);
            const Vector<Float, 3> dv1(&d_vertices[i1 * 3]);
            const Vector<Float, 3> dv2(&d_vertices[i2 * 3]);

            if constexpr (Mode == DifferentiationMode::Backward)
            {
                // Compute dvn (derivative projected onto normal direction)
                dvn_vec[0] = unit_normal[0];
                dvn_vec[1] = unit_normal[1];
                dvn_vec[2] = unit_normal[2];
            }
            else
            {
                // Compute dvn (derivative projected onto normal direction)
                dvn_vec[0] = dot(dv0, unit_normal);
                dvn_vec[1] = dot(dv1, unit_normal);
                dvn_vec[2] = dot(dv2, unit_normal);
            }
        }

        // Bounding box in grid coordinates
        const Float min_x = std::min({v0.x(), v1.x(), v2.x()});
        const Float min_y = std::min({v0.y(), v1.y(), v2.y()});
        const Float min_z = std::min({v0.z(), v1.z(), v2.z()});
        const Float max_x = std::max({v0.x(), v1.x(), v2.x()});
        const Float max_y = std::max({v0.y(), v1.y(), v2.y()});
        const Float max_z = std::max({v0.z(), v1.z(), v2.z()});

        const int bbox_min_i = std::max(0, static_cast<int>((min_x + 1) / step_x));
        const int bbox_min_j = std::max(0, static_cast<int>((min_y + 1) / step_y));
        const int bbox_min_k = std::max(0, static_cast<int>((min_z + 1) / step_z));
        const int bbox_max_i = std::min(static_cast<int>(depth) - 1, static_cast<int>((max_x + 1) / step_x));
        const int bbox_max_j = std::min(static_cast<int>(height) - 1, static_cast<int>((max_y + 1) / step_y));
        const int bbox_max_k = std::min(static_cast<int>(width) - 1, static_cast<int>((max_z + 1) / step_z));

        // Triangle coordinates per axis for efficient plane tests
        const Vector<Float, 3> tri_x(v0.x(), v1.x(), v2.x());
        const Vector<Float, 3> tri_y(v0.y(), v1.y(), v2.y());
        const Vector<Float, 3> tri_z(v0.z(), v1.z(), v2.z());

        // Start with full triangle in barycentric coordinates
        remaining_x.clear();
        remaining_x.push_back(Vector<Float, 3>(Float(1), Float(0), Float(0)));
        remaining_x.push_back(Vector<Float, 3>(Float(0), Float(1), Float(0)));
        remaining_x.push_back(Vector<Float, 3>(Float(0), Float(0), Float(1)));

        for (int i = bbox_min_i; i <= bbox_max_i; ++i)
        {
            if (remaining_x.size() < 3)
                break;

            const Float x_hi = Float(-1) + (i + 1) * step_x;
            split_polygon_bary(remaining_x, tri_x, x_hi, poly_x, temp_above);
            remaining_x = std::move(temp_above);

            if (poly_x.size() < 3)
                continue;

            // Get y bounds for this x-slice
            Float y_min_slice = std::numeric_limits<Float>::max();
            Float y_max_slice = std::numeric_limits<Float>::lowest();
            for (auto const& bary : poly_x)
            {
                const Float y_val = dot(bary, tri_y);
                y_min_slice       = std::min(y_min_slice, y_val);
                y_max_slice       = std::max(y_max_slice, y_val);
            }

            const int j_min = std::max(bbox_min_j, static_cast<int>((y_min_slice + 1) / step_y));
            const int j_max = std::min(bbox_max_j, static_cast<int>((y_max_slice + 1) / step_y));

            remaining_y = poly_x;

            for (int j = j_min; j <= j_max; ++j)
            {
                if (remaining_y.size() < 3)
                    break;

                const Float y_hi = Float(-1) + (j + 1) * step_y;
                split_polygon_bary(remaining_y, tri_y, y_hi, poly_xy, temp_above);
                remaining_y = std::move(temp_above);

                if (poly_xy.size() < 3)
                    continue;

                // Get z bounds for this xy-slice
                Float z_min_slice = std::numeric_limits<Float>::max();
                Float z_max_slice = std::numeric_limits<Float>::lowest();
                for (auto const& bary : poly_xy)
                {
                    const Float z_val = dot(bary, tri_z);
                    z_min_slice       = std::min(z_min_slice, z_val);
                    z_max_slice       = std::max(z_max_slice, z_val);
                }

                const int k_min = std::max(bbox_min_k, static_cast<int>((z_min_slice + 1) / step_z));
                const int k_max = std::min(bbox_max_k, static_cast<int>((z_max_slice + 1) / step_z));

                remaining_z = poly_xy;

                for (int k = k_min; k <= k_max; ++k)
                {
                    if (remaining_z.size() < 3)
                        break;

                    const Float z_hi = Float(-1) + (k + 1) * step_z;
                    split_polygon_bary(remaining_z, tri_z, z_hi, poly_bary, temp_above);
                    remaining_z = std::move(temp_above);

                    if (poly_bary.size() < 3)
                        continue;

                    // Convert to Cartesian for area/integral computation
                    poly_xyz.clear();
                    for (auto const& bary : poly_bary)
                        poly_xyz.push_back(v0 * bary[0] + v1 * bary[1] + v2 * bary[2]);

                    if constexpr (primal)
                    {
                        // Compute area of x-y projection
                        const Float area_xy = polygon_area_2d(poly_xyz);

                        if (area_xy < Float(1e-12))
                            continue;

                        const bool  below_center = z_hi <= (0 + (step_z / 2));
                        const Float sign         = below_center ? -sign_nz : sign_nz;
                        const int   start        = below_center ? k + 1 : center_k;
                        const int   end          = below_center ? center_k : k;
                        const Float z_ref        = below_center ? z_hi : z_hi - step_z;

                        // Full contribution: iterate toward center only
                        // Below: k+1 to center_k (inclusive)
                        // Above: center_k+1 to k-1 (exclusive of center to avoid double counting)
                        const Float contribution = sign * area_xy / (step_x * step_y);
                        for (int kk = start; kk < end; kk++)
                        {
                            const size_t idx = static_cast<size_t>(i * height * width + j * width + kk);
                            occupancy[idx] += contribution;
                        }

                        // This cell gets partial contribution
                        Vector<Float, 3> const& pv0     = poly_xyz[0];
                        Float                   contrib = Float(0);
                        // k == center_k: no full contribution, only partial
                        // Partial contribution for this cell
                        for (size_t vi = 1; vi < poly_xyz.size() - 1; ++vi)
                        {
                            Vector<Float, 3> const& pv1      = poly_xyz[vi];
                            Vector<Float, 3> const& pv2      = poly_xyz[vi + 1];
                            const Float             sub_area = Float(0.5) * std::abs(
                                                                                (pv1.x() - pv0.x()) * (pv2.y() - pv0.y()) -
                                                                                (pv2.x() - pv0.x()) * (pv1.y() - pv0.y()));
                            const Float sub_z_avg = (pv0.z() + pv1.z() + pv2.z()) / Float(3);
                            contrib += sub_area * (sub_z_avg - z_ref);
                        }

                        const size_t idx = static_cast<size_t>(i * height * width + j * width + k);
                        occupancy[idx] += sign_nz * contrib / cell_volume;
                    }
                    else
                    {
                        const size_t idx = static_cast<size_t>(i * height * width + j * width + k);
                        if constexpr (Mode == DifferentiationMode::Forward)
                        {
                            poly_dvn.clear();
                            for (auto const& bary : poly_bary)
                            {
                                poly_dvn.push_back(dot(bary, dvn_vec));
                            }
                            // This cell gets partial contribution
                            Vector<Float, 3> const& pv0     = poly_xyz[0];
                            Float                   contrib = Float(0);

                            const Float pivot = poly_dvn[0] / Float(3);
                            for (size_t vi = 1; vi < poly_xyz.size() - 1; ++vi)
                            {
                                Vector<Float, 3> const& pv1 = poly_xyz[vi];
                                Vector<Float, 3> const& pv2 = poly_xyz[vi + 1];

                                const Vector<Float, 3> edge1    = pv1 - pv0;
                                const Vector<Float, 3> edge2    = pv2 - pv0;
                                const Float            sub_area = Float(0.5) * norm(cross(edge1, edge2));

                                const Float b1 = poly_dvn[vi] / Float(3);
                                const Float b2 = poly_dvn[vi + 1] / Float(3);

                                contrib += sub_area * (pivot + b1 + b2);
                            }
                            d_occupancy[idx] += contrib / cell_volume;
                        }
                        else
                        {
                            // This cell gets partial contribution
                            Vector<Float, 3> const& pv0     = poly_xyz[0];
                            Vector<Float, 3>        contrib = Vector<Float, 3>(Float(0));

                            const Vector<Float, 3> pivot = poly_bary[0] / Float(3);
                            for (size_t vi = 1; vi < poly_xyz.size() - 1; ++vi)
                            {
                                Vector<Float, 3> const& pv1 = poly_xyz[vi];
                                Vector<Float, 3> const& pv2 = poly_xyz[vi + 1];

                                const Vector<Float, 3> edge1    = pv1 - pv0;
                                const Vector<Float, 3> edge2    = pv2 - pv0;
                                const Float            sub_area = Float(0.5) * norm(cross(edge1, edge2));

                                const Vector<Float, 3> b1 = poly_bary[vi] / Float(3);
                                const Vector<Float, 3> b2 = poly_bary[vi + 1] / Float(3);

                                contrib += sub_area * (pivot + b1 + b2);
                            }
                            Float const differential_weight = d_occupancy[idx] / cell_volume;
                            for (int d = 0; d < 3; ++d)
                            {
                                d_vertices[3 * i0 + d] += contrib[0] * dvn_vec[d] * differential_weight;
                                d_vertices[3 * i1 + d] += contrib[1] * dvn_vec[d] * differential_weight;
                                d_vertices[3 * i2 + d] += contrib[2] * dvn_vec[d] * differential_weight;
                            }
                        }
                    }
                }
            }
        }
    }
}


// 3d Wrappers
template<typename Float>
void voxelize_cf(Float const* vertices, uint32_t const num_vertices,
                       uint32_t const* faces, uint32_t const num_faces,
                       Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width)
{
    return voxelize_cf_generalized<Float, true, DifferentiationMode::Forward>(vertices, num_vertices,
                                                             faces, num_faces,
                                                             occupancy, depth, height, width,
                                                             NULL,
                                                             NULL);
}

template<typename Float>
void voxelize_cf_forward(Float const* vertices, uint32_t const num_vertices,
                               uint32_t const* faces, uint32_t const num_faces,
                               Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                               Float const* d_vertices,
                               Float*       d_occupancy)
{
    return voxelize_cf_generalized<Float, false, DifferentiationMode::Forward>(vertices, num_vertices,
                                                             faces, num_faces,
                                                             occupancy, depth, height, width,
                                                             d_vertices,
                                                             d_occupancy);
}

template<typename Float>
void voxelize_cf_backward(Float const* vertices, uint32_t const num_vertices,
                               uint32_t const* faces, uint32_t const num_faces,
                               Float* occupancy, uint32_t const depth, uint32_t const height, uint32_t const width,
                               Float* d_vertices,
                               Float const*       d_occupancy)
{
    return voxelize_cf_generalized<Float, false, DifferentiationMode::Backward>(vertices, num_vertices,
                                                             faces, num_faces,
                                                             occupancy, depth, height, width,
                                                             d_vertices,
                                                             d_occupancy);
}

} // namespace dvx
