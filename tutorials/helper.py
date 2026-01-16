import numpy as np
import polyscope as ps

def render_grid(name, griddata):
    grid_size = griddata.shape
    n_x = grid_size[0] + 1
    n_y = grid_size[1] + 1
    n_z = grid_size[2] + 1
    # Create hexahedral mesh for voxel visualization
    x_verts = np.linspace(-1, 1, n_x)
    y_verts = np.linspace(-1, 1, n_y)
    z_verts = np.linspace(-1, 1, n_z)

    X, Y, Z = np.meshgrid(x_verts, y_verts, z_verts, indexing='ij')
    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Create hexahedral cells
    cells = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                # Vertex indices for this hex cell
                # Bottom 4 vertices counter-clockwise, then top 4 vertices counter-clockwise
                v = []

                for dk in [0, 1]:  # bottom then top
                    for di, dj in [(0, 0), (0, 1), (1, 1), (1, 0)]:  # counter-clockwise
                        idx = (i + di) * n_y * n_z + (j + dj) * n_z + (k + dk)
                        v.append(idx)
                cells.append(v)
    cells = np.array(cells)

    # Filter out cells with winding number close to zero
    griddata_flat = griddata.ravel()
    eps = 1e-4
    mask = np.abs(griddata_flat) > eps
    cells_filtered = cells[mask]
    griddata_filtered = griddata_flat[mask]

    # Register volume mesh with winding numbers as scalar quantity
    vm = ps.register_volume_mesh(name, vertices, hexes=cells_filtered)
    vm.add_scalar_quantity("data", griddata_filtered, defined_on='cells', cmap='viridis', enabled=True)