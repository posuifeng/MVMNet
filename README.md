# model evaluation

## Download and layout


The data root is the folder containing **manifest.jsonl**, **samples/**,
**splits/** and **weights/** directly. Do not pass the archive file or the
`samples` subfolder. If extraction produces nested folders, select the inner
`data_and_weights` folder containing these files.

Option A — default layout (no path argument required):

```text
any_parent_folder/
  repository/       # this downloaded GitHub repository; its name may vary
    run_test.py
    README.md
  data_and_weights/
    manifest.jsonl
    samples/        # folders 1 through 80
    splits/test.txt
    weights/best.pt
    evaluation/
```

Option B — keep data anywhere and pass `--data-dir`. You do not need to edit
Python source files. The path may be absolute, or relative to the terminal's
current working directory. Quote paths containing spaces. No particular drive
letter, original project folder or NAS is required.

## Install and run
Tested with Python 3.11.7, PyTorch 2.3.0, NumPy 1.24.0, SciPy 1.10.1 on Windows/NVIDIA GPU.
In the code directory:

```shell
python -m venv .venv
```
Windows activation: `.venv\Scripts\activate`

```shell
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python run_test.py --check-data
python run_test.py --smoke --device cuda:0
python run_test.py --device cuda:0
```
If the data is elsewhere, replace `YOUR_EXTRACTED_DATA_FOLDER` below with the
actual data root on your computer. This is a placeholder, not a literal path:

```shell
python run_test.py --data-dir "YOUR_EXTRACTED_DATA_FOLDER" --check-data
python run_test.py --data-dir "YOUR_EXTRACTED_DATA_FOLDER" --smoke --device cuda:0
python run_test.py --data-dir "YOUR_EXTRACTED_DATA_FOLDER" --device cuda:0
```

For example, if it is in a `datasets` folder next to the repository:

```shell
python run_test.py --data-dir "../datasets/data_and_weights" --device cuda:0
```

`--check-data` validates all 80 entries and input paths without loading PyTorch
or running inference. A successful check does not replace installing the runtime.
CPU option: install torch==2.3.0 from https://download.pytorch.org/whl/cpu and use
`--device cpu` (slower; not benchmarked). Load only the bundled trusted checkpoint.

## Files and outputs
`src` implements the model/data processing; `scripts` implements evaluation.
`run_test.py` runs both stages. New outputs go to `results/full80` in the code folder;
`--smoke` evaluates one sample. Reference outputs are in the data package's `evaluation`.
The data package includes 80 empty/full three-view samples and epoch 98 `best.pt`.
Keep manifest.jsonl and splits/test.txt: these provide required paths/parameters.
Labels box/surface are in the target coordinate system.

