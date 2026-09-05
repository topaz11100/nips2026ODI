### Workshop ODI

JAX implementation of the six SNN experiments. Every dataset uses its input as the spectral reference. SMR increases the mean spectral distance between the input and hidden spike representations through $ L = L_{\mathrm{task}} - \lambda D_{\mathrm{model}} $.

#### Installation

Run from the repository root:

```bash
conda env create --file environment.yml
conda activate jax
```

See [Conda setup](docs/setup/conda_environment.md) for GPU requirements.

#### Raw data setup

Raw datasets are not included in the release. Download them separately and place them under `data/raw_data/` in the repository root. The preprocessing launcher reads local files and does not download or unpack archives.

Create the dataset directories:

```bash
mkdir -p data/raw_data/{mnist,shd,cifar-10,dvs128-gesture,cifar-100,cifar10-dvs}
```

Download and extract the following files into the corresponding directories:

| Dataset ID | Model | Download and preparation |
| --- | --- | --- |
| `s-mnist` | MLP | [MNIST, CVDF mirror](https://github.com/cvdfoundation/mnist): put the four IDX files in `mnist/`; both original filenames and filenames ending in `.gz` are accepted |
| `shd` | MLP | [Spiking Heidelberg Digits](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/): extract the SHD training and test downloads into `shd/` as `shd_train.h5` and `shd_test.h5` |
| `cifar-10_16` | VGG11 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): extract `cifar-10-python.tar.gz` into `cifar-10/`, keeping the `cifar-10-batches-py/` directory |
| `dvs128-gesture` | VGG11 | [IBM DvsGesture dataset](https://ibm.biz/EventCameraData): extract the complete archive into `dvs128-gesture/`, keeping `DvsGesture/` and its trial lists, AEDAT recordings, and label CSV files |
| `cifar-100_16` | Seven-block ResNet | [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html): extract `cifar-100-python.tar.gz` into `cifar-100/`, keeping the `cifar-100-python/` directory |
| `cifar10-dvs` | Seven-block ResNet | [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671): extract all ten class archives into `cifar10-dvs/`, with one directory per class containing the original `.aedat` recordings |

The resulting layout must contain these files and directories. Wildcards below represent all matching recordings and their labels.

```text
data/raw_data/
|-- mnist/
|   |-- train-images-idx3-ubyte
|   |-- train-labels-idx1-ubyte
|   |-- t10k-images-idx3-ubyte
|   `-- t10k-labels-idx1-ubyte
|-- shd/
|   |-- shd_train.h5
|   `-- shd_test.h5
|-- cifar-10/
|   `-- cifar-10-batches-py/
|       |-- data_batch_1
|       |-- data_batch_2
|       |-- data_batch_3
|       |-- data_batch_4
|       |-- data_batch_5
|       `-- test_batch
|-- dvs128-gesture/
|   `-- DvsGesture/
|       |-- trials_to_train.txt
|       |-- trials_to_test.txt
|       |-- *.aedat
|       `-- *_labels.csv
|-- cifar-100/
|   `-- cifar-100-python/
|       |-- train
|       `-- test
`-- cifar10-dvs/
    |-- airplane/*.aedat
    |-- automobile/*.aedat
    |-- bird/*.aedat
    |-- cat/*.aedat
    |-- deer/*.aedat
    |-- dog/*.aedat
    |-- frog/*.aedat
    |-- horse/*.aedat
    |-- ship/*.aedat
    `-- truck/*.aedat
```

Use the Python archives for CIFAR-10 and CIFAR-100 and the original AEDAT recordings for the DVS datasets. Preserve the DvsGesture recording names referenced by the trial lists and their matching `*_labels.csv` files. Keep CIFAR10-DVS filenames unchanged; preprocessing creates its train/test split from sorted recordings within each class.

Run `./bash/paper_data.sh` after all six datasets are in place. It creates the prepared arrays and train/test probes automatically. Sequential MNIST is derived from the MNIST images, and static CIFAR images are repeated for 16 time steps.

#### Execution

With the supplied training coefficients, run:

```bash
./bash/paper_data.sh
./bash/paper_train_signal.sh
./bash/paper_reg.sh
```

| Command | What it does |
| --- | --- |
| `paper_data.sh` | Prepares all six datasets, saves train/test probe indices, and computes input reference spectra |
| `paper_train_signal.sh` | Trains without SMR and evaluates spectral distances at the saved checkpoints |
| `paper_reg.sh` | Trains with each dataset's configured SMR coefficient |

The output root in `config/paper/run.yaml` must be missing or empty before data preparation. Later commands reuse that run and its seed. Repeating a training command archives the previous output with a `.previous_<timestamp>` suffix and starts training again.

To recalculate coefficients after data preparation:

```bash
./bash/calibrate_lambda.sh
```

Copy the six values from `paper_artifacts/lambda_calibration.csv` into the `lambda` mapping in `config/paper/train_regularization_on.yaml` before regularized training. Calibration reads the saved training probe and writes one CSV. See [lambda calibration](docs/lambda_calibration.md).

Select another regularized-training configuration with `./bash/paper_reg.sh --config PATH`. The launcher saves and uses a snapshot of that configuration for the entire run.

#### Configuration

All configuration files are under `config/paper/`. Dataset IDs match the table above.

| File | Settings and meaning |
| --- | --- |
| `run.yaml` | `root`: output directory; `seed`: random seed for the run |
| `data_prep.yaml` | `train_probe_size`, `test_probe_size`: saved probe sample counts per dataset |
| `dataset_signal_analysis.yaml` | `batch_size`: samples processed together when computing input spectra |
| `train_regularization_off.yaml` | `epochs`, `learning_rate`, per-dataset training `batch_size`, and `checkpoint_epochs` for spectral analysis |
| `train_regularization_on.yaml` | `epochs`, `learning_rate`, per-dataset training `batch_size`, and one positive `lambda` per dataset |
| `signal_analysis_regularization_off.yaml` | `batch_size`: samples processed together during checkpoint analysis |
| `lambda_calibration.yaml` | `artifact_root`: prepared run; `output_csv`: result path; `target_ratio`: weighted SMR/task gradient-norm ratio; `batch_size`: processing chunks; `gpu_ids`, `memory_fraction`, `compile_threads`: execution resources |

The supplied training settings use seed 0, 50 epochs, learning rate 0.0025, and checkpoints at epochs 1, 5, 10, and 50. Calibration processing batches match the training batch sizes. If the output root changes, update calibration's `artifact_root` and `output_csv` as well.

#### GPU selection

The three paper launchers use GPUs 0 and 1 by default. For regularized training, GPU 0 runs CIFAR10-DVS and SHD; GPU 1 runs the other four datasets. Each GPU runs one dataset at a time.

To use one GPU:

```bash
WORKSHOP_ODI_GPU_IDS="0" ./bash/paper_reg.sh
```

For calibration, select GPUs with `gpu_ids` in its YAML.

| Environment variable | Meaning |
| --- | --- |
| `PYTHON_BIN` | Python executable; defaults to `python` |
| `WORKSHOP_ODI_GPU_IDS` | Physical GPU IDs for the paper launchers; defaults to `0 1` |
| `WORKSHOP_ODI_MEMORY_FRACTION` | GPU memory allocation fraction for the paper launchers; defaults to `0.75` |
| `WORKSHOP_ODI_CPU_AFFINITY` | CPU core list; defaults to `0-15`; set to an empty string to disable CPU pinning |

The paper launchers wait for each selected GPU to have at most 512 MiB in use before starting a job.

#### Results

Outputs are stored under the configured root. `run_manifest.yaml` identifies the stage directories; execution logs are under `logs/`.

| Output | Contents |
| --- | --- |
| `train_regularization_off_<timestamp>/training_metrics.csv` | Per-epoch loss and accuracy without SMR |
| `train_regularization_on_<timestamp>/training_metrics.csv` | Per-epoch loss and accuracy with the applied dataset-specific lambda |
| `signal_analysis_regularization_off_<timestamp>/d_model.csv` | Train/test probe spectral distances at each saved checkpoint |
| `lambda_calibration.csv` | Six rows with columns `dataset,lambda` |

Accuracy is stored as a fraction; multiply by 100 for percent. Training accuracy is accumulated during each epoch; test accuracy is evaluated on the full test split after the epoch. Only training without SMR saves model checkpoints.
