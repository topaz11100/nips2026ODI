### SMR lambda calibration

After data preparation, run:

```bash
./bash/calibrate_lambda.sh
```

The saved `train_probe_indices.npy` selects rows from the prepared training arrays. Each complete probe is treated as one batch at fixed initial parameters; the YAML `batch_size` divides the computation into memory-sized chunks.

$$\lambda = \rho \frac{\lVert g_{\mathrm{task}} \rVert_2}{\lVert g_{\mathrm{SMR}} \rVert_2}.$$

Both gradients use shared hidden parameters, excluding the output head. The default `target_ratio: 0.1` is the chosen intervention strength at initialization. The coefficient stays fixed during training; the gradient ratio can change.

Configuration: [lambda_calibration.yaml](../config/paper/lambda_calibration.yaml). `gpu_ids` selects GPUs, `memory_fraction` controls GPU memory allocation, and `compile_threads` controls compilation parallelism. `--config PATH` selects another YAML; `--dry-run` displays the planned jobs.

Output: `paper_artifacts/lambda_calibration.csv`, with columns `dataset,lambda`. Copy its six values into [train_regularization_on.yaml](../config/paper/train_regularization_on.yaml) before running `paper_reg.sh`.

The gradient-ratio weighting principle follows [Esser et al., Taming Transformers for High-Resolution Image Synthesis, CVPR 2021, Eq. (7)](https://arxiv.org/abs/2012.09841), adapted here to a fixed coefficient calibrated on the saved training probe.
