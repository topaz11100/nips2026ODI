### Conda environment

Use Linux, Conda, and an NVIDIA GPU with a driver compatible with CUDA 13. [environment.yml](../../environment.yml) pins Python 3.14.6, JAX 0.11.0, and the runtime dependencies.

From the repository root:

```bash
conda env create --file environment.yml
conda activate jax
python -c "import jax; print(jax.devices())"
```

The last command should list a GPU device. The launchers load `src/` directly, so no package installation is needed.

If the environment exposes its interpreter as `python3`:

```bash
PYTHON_BIN=python3 ./bash/paper_data.sh
```

See the [README](../../README.md) for data placement, execution order, configuration, and GPU selection.
