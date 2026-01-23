<p align="center">

  <h1 align="center"><a href="https://link/to/conference">Differentiable Voxelization of Shape Representations</a></h1>

  <div  align="center">
    <a href="https://link/to/conference">
      <img src="." alt="Logo" width="100%">
    </a>
  </div>

  <p align="center">
    <i>Dummy Conference or Journal (2026)</i>
    <br />
    <a href="https://author.page"><strong>Author A</strong></a>
    ·
    <a href="https://author.page"><strong>Author B</strong></a>
    ·
    <a href="https://author.page"><strong>Author C</strong></a>
  </p>
</p>

## About

This repository provides an implementation of the paper "Differentiable Voxelization of Shape Representations". The algorithms are implement in C++, targeting the CPU and the GPU (using CUDA). The `dvx` package exposes these algorithms to Python, where they are readily usable with PyTorch, NumPy, and [Dr.Jit](https://github.com/mitsuba-renderer/drjit).

## Installation

The easiest way to install the Python package is via `pip`

```bash
pip install git+https://link/to/this/repository.git
```

### Optional: Test the Installation

To test the installation, run

```bash
pip install numpy pytest
python -m pytest .\tests -v
```

Some tests will be skipped, depending on the availability of packages (NumPy is required, PyTorch/Dr.Jit are optional).

## Usage

TODO

## License and Copyright

The code in this repository is provided under a TODO license. 

## Citation

If you find this code or our method useful for your research, please cite our paper

```bibtex
@article{authors:2026:dvx,
    ...
}
```
