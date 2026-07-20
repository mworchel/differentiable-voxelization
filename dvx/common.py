def half_voxel_size(resolution: int) -> float:
    """Returns half the size of a voxel given the resolution of the voxel grid in one dimension."""
    return 1 / resolution # = 0.5 * (2 / resolution)
                          #          |
                          #        size of the volume 