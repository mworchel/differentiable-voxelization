import numpy as np
from pathlib import Path
from svgpathtools import svg2paths

def edge_indices(num_vertices: int, closed: bool):
    indices = np.arange(num_vertices)
    if closed:
        return np.stack([indices, (indices + 1) % num_vertices], axis=-1)
    else:
        return np.stack([indices[:-1], indices[1:]], axis=-1)

def load_svg_as_linear_path(filepath: Path, path_idx: int, resolution: int = 256):
    paths, _ = svg2paths(filepath)

    v = np.array([(p.real, p.imag) for p in [paths[path_idx].point(t) for t in np.linspace(0, 1, resolution)]])
    # Normalize to [-1, 1] range
    xmin, xmax, ymin, ymax = paths[path_idx].bbox()
    v[:, 0] = 2 * (v[:, 0] - xmin) / (xmax - xmin) - 1
    v[:, 1] = 2 * (v[:, 1] - ymin) / (ymax - ymin) - 1
    v *= 0.98
    f = edge_indices(v.shape[0], closed=True)

    return v, f