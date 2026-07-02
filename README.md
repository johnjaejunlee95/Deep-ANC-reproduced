# Deep ANC (Active Noise Control)

PyTorch reproduction code for experiments inspired by [**Deep ANC: A deep learning approach to active noise control**](https://www.sciencedirect.com/science/article/pii/S0893608021001258).

This repository is not an official implementation. The reproduced code may differ from the original paper, and it also references implementation patterns from other public Active Noise Control codebases.

## Repository layout

```text
.
├── main.py                 # readable top-level train/infer process
├── src/models/             # DeepANC network definition
├── src/utils/              # datasets, STFT, DSP, checkpoints, train/eval helpers
├── src/mat_files/          # primary/secondary path filter .mat files
├── results/noise_samples/  # sample WAV files for inference
├── tests/                  # regression tests
├── requirements.txt
└── pyproject.toml
```

## Install

Create a conda environment:

```bash
conda create -n deepanc python=3.10
conda activate deepanc
```

Install PyTorch with CUDA 11.8 using conda:

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Or install PyTorch with CUDA 11.8 using pip:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset format

Dataset files are available here: [Synthetic Dataset](https://drive.google.com/file/d/1hs7_eHITxL16HeugjQoqYFTs-Cm7J-Tq/view?usp=sharing)

The synthetic data and filters here are referenced from official implementation of GFANC paper. 

Paper: [Link](https://arxiv.org/abs/2303.05788) \\
Github: [Link](https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control)


The training dataset directory must contain these split files:

- `Hard_Index_train.csv`
- `Hard_Index_val.csv`
- `Hard_Index_test.csv`

Each CSV must include a `File_path` column containing paths relative to the dataset root.

## Train

```bash
python main.py --mode train \
  --data-path /path/to/dataset \
  --filter-dir src/mat_files \
  --checkpoint-dir checkpoints \
  --device auto
```

The best and latest checkpoints are written to:

- `checkpoints/deepanc_best.pt`
- `checkpoints/deepanc_latest.pt`

## Inference

```bash
python main.py --mode infer \
  --checkpoint checkpoints/deepanc_best.pt \
  --input-dir results/noise_samples \
  --output-dir wav_results \
  --filter-dir src/mat_files
```

## Test

```bash
python3 -m pytest
python3 -m py_compile main.py src/utils/*.py src/models/*.py tests/*.py
```

## Notes

- Device selection is centralized through `--device`; use `auto`, `cpu`, or `cuda`.
- Checkpoints use a shared schema and can also load legacy checkpoints containing `state_dict`.
- Validation and test dataloaders do not drop trailing samples.
