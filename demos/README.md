# Differentiable Voxelization: Demos

This folder contains demo notebooks that reproduce the different applications shown in the paper:

- **Self-Intersection Resolving** (Section 4.2): [`self-intersection_experiment.ipynb`](./self-intersection_experiment.ipynb)
- **Shape Deformation for Bandsaw Cutting** (Section 4.3): [`bandsaw_experiment.ipynb`](./bandsaw_experiment.ipynb)
- **Space Filling Shapes in 3D** (Section 4.4): [`tiling_experiment.ipynb`](./tiling_experiment.ipynb)

## Setup

1. Create a virtual Python environment

    using **venv**:

    ```bash
    python -m venv .env
    source .env/bin/activate # On Windows use `.env/Scripts/activate`
    ```

    or using **conda**:
    ```bash
    conda create -n dvx-demos python=3.11
    conda activate dvx-demos
    ```
2. Install the required dependencies

    ```bash
    pip install -r requirements.txt
    ```
3. Run any notebook using the virtual environment.