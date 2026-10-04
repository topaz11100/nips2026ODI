### SMR lambda calibration

After data preparation, run:

```bash
./bash/calibrate_lambda.sh
```

The saved `train_probe_indices.npy` selects rows from the prepared training arrays. Each complete probe is treated as one batch at fixed initial parameters; the hardcoded `batch_size` divides the computation into memory-sized chunks.

$$ \lambda = \rho \frac{\lVert g_{\mathrm{task}} \rVert_2}{\lVert g_{\mathrm{SMR}} \rVert_2}. $$

Both gradients use shared hidden parameters, excluding the output head. The hardcoded `target_ratio` is `0.1`, the chosen intervention strength at initialization. The coefficient stays fixed during training; the gradient ratio can change.

The `lambda_calibration` settings are hardcoded in [paper_defaults.py](../src/workshop_jax/config/paper_defaults.py). `gpu_ids` selects GPUs, `memory_fraction` controls GPU memory allocation, and `compile_threads` controls compilation parallelism. `CALIBRATION_ARTIFACT_ROOT` selects the prepared run, and `CALIBRATION_OUTPUT_CSV` sets the output path. These paths are independent of `run.yaml`; update them if the prepared run's location changes. `--dry-run` displays the planned jobs.

Output: `paper_artifacts/lambda_calibration.csv` by default, with columns `dataset,lambda`. Copy its six values into `REGULARIZATION_LAMBDA` in [paper_defaults.py](../src/workshop_jax/config/paper_defaults.py) before running `paper_reg.sh`. Calibration does not modify the hardcoded coefficients automatically.

The gradient-ratio weighting principle follows [Esser et al., Taming Transformers for High-Resolution Image Synthesis, CVPR 2021, Eq. (7)](https://arxiv.org/abs/2012.09841), adapted here to a fixed coefficient calibrated on the saved training probe.
