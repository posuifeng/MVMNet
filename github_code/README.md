# MVMNet Model Evaluation

This repository provides the code for evaluating the pretrained MVMNet model on 80 test samples, including point cloud registration, segmentation, and muck volume estimation.

## 1. Data Preparation

Download and extract the dataset and pretrained model weights. Organize the files as follows:

```text
project/
├── repository/
│   ├── run_test.py
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   └── scripts/
└── data_and_weights/
    ├── manifest.jsonl
    ├── samples/          # Samples 1–80
    ├── splits/
    │   └── test.txt
    └── weights/
        └── best.pt
```

By default, the program automatically locates `data_and_weights` either inside the repository or in its parent directory.

If the dataset is stored elsewhere, specify its location using `--data-dir`. The path must point to the folder containing `manifest.jsonl`, `samples/`, `splits/`, and `weights/`.

## 2. Environment Setup

The evaluation code has been tested with:

- Python 3.11.7
- PyTorch 2.3.0
- NumPy 1.24.0
- SciPy 1.10.1

Install the dependencies from the repository directory:

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The default configuration uses an NVIDIA GPU with CUDA 12.1-compatible PyTorch.

## 3. Run Evaluation

Run the following commands from the repository directory.

### Step 1. Check Dataset Integrity

Verify the dataset structure, sample IDs, file paths, and required metadata:

```bash
python run_test.py --check-data
```

Optionally, check the point cloud files for exact duplicate XYZ coordinates within the same state and view, regardless of point order:

```bash
python run_test.py --check-content
```

### Step 2. Test the Evaluation Pipeline

Run a smoke test using one sample:

```bash
python run_test.py --smoke --device cuda:0
```

### Step 3. Evaluate All 80 Samples

Run the complete evaluation:

```bash
python run_test.py --device cuda:0
```

The evaluation includes point cloud registration, segmentation, and muck volume estimation.

### Custom Data Directory

If `data_and_weights` is stored in another location, specify the path explicitly:

```bash
python run_test.py --data-dir "/path/to/data_and_weights" --device cuda:0
```

### CPU Evaluation

For systems without an NVIDIA GPU, install the CPU version of PyTorch and run:

```bash
python run_test.py --device cpu
```

CPU evaluation is supported but has not been benchmarked.

## 4. Evaluation Outputs

By default, evaluation results are saved in:

```text
repository/
└── results/
    └── full80/
        ├── segmentation/
        └── volume/
```

The segmentation directory contains registration and segmentation evaluation outputs, while the volume directory contains muck volume estimation results and error statistics.

To specify a custom output directory:

```bash
python run_test.py --device cuda:0 --output-dir results/custom_test
```

## 5. Notes

- The evaluation uses the pretrained checkpoint `weights/best.pt`.
- The test dataset contains 80 samples, each comprising empty and loaded point clouds from three LiDAR views.
- `manifest.jsonl` provides sample paths, reference labels, and geometric parameters required for evaluation.
- `--check-data` verifies dataset structure and metadata without running model inference.
- `--check-content` additionally checks for exact duplicate point clouds within the same state and view.
- `--smoke` evaluates one sample to verify the execution pipeline.
- The model evaluation requires the provided `src/` and `scripts/` directories.