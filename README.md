# CondiNILM

<div align="center">

**CondiNILM: Feature-wise Modulated Multi-Task Learning for Non-Intrusive Load Monitoring**

Research codebase by **Siyi Li**  


![Python](https://img.shields.io/badge/Python-3.10%20|%203.11|%203.12%20-blue)

</div>

---

## Overview

**CondiNILM** is a **novel multi-task deep learning framework for Non-Intrusive Load Monitoring (NILM)**, developed as part of a Master’s thesis at **TU Braunschweig**.

The framework targets **device-level power disaggregation from aggregate household measurements**, with a particular focus on:

- **Multi-appliance joint learning**
- **Non-stationary power signals**
- **Time–frequency feature fusion**
- **Device-conditioned modeling via FiLM (Feature-wise Linear Modulation)**

Unlike classical NILM approaches that train **one model per appliance**, CondiNILM formulates NILM as a **single multi-output learning problem**, where **shared representations** are dynamically modulated by **device-specific conditions**.

---

## Key Contributions

CondiNILM introduces several original design choices:

### 1. Multi-Task NILM with Device Conditioning

- A **single unified model** predicts power consumption for multiple appliances simultaneously
- Each appliance is modeled via **device-conditioned output heads**
- Reduces parameter redundancy and improves cross-device generalization

### 2. FiLM-Modulated Feature Decoding

- **FiLM layers** are used to modulate intermediate representations:
  
  \[
  \text{FiLM}(x \mid d) = \gamma_d \cdot x + \beta_d
  \]

- Enables **explicit device-aware control** over shared temporal features
- Prevents power “leakage” and cross-device interference common in multi-head NILM

### 3. Time–Frequency Feature Fusion

- The model jointly exploits:
  - **Time-domain power sequences**
  - **Frequency-domain representations** (FFT / STFT / spectral statistics)
  - **Engineered auxiliary features** (e.g. activity priors, signal energy)

- These modalities are fused through attention-based encoders

### 4. Sequence-Level Supervision with Dense Outputs

- Supports **Seq2Seq**, **Seq2Subsequence**, and **Seq2Point** supervision
- Enables **high-resolution waveform reconstruction** at inference time
- Training and inference strategies can be decoupled for efficiency

---

## Project Scope

This repository serves as the **official thesis codebase of CondiNILM** and includes:

- The complete implementation of **CondiNILM**
- A unified training and evaluation pipeline
- Re-implementations of **10+ recent NILM baselines** in PyTorch
- Reproducible experiment scripts and configurations

The framework is designed for **research extensibility**, not only benchmark reproduction.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CondiNILM.git
cd CondiNILM
```

### 2.Create Conda Environment
macOS:
```bash
conda env create -f environment_mac.yaml
conda activate condinilm
```
win:
```bash
conda env create -f environment_win.yaml
conda activate condinilm
```

### Verify Installation(nvidia)
```bash
python -c "import torch; print(torch.version.cuda)"
```
---

## Environment Specification
environment.yml:

```yaml
name: condinilm
channels:
  - defaults
dependencies:
  - bzip2=1.0.8=h80987f9_6
  - ca-certificates=2025.12.2=hca03da5_0
  - expat=2.7.3=h50f4ffc_4
  - libcxx=20.1.8=hd7fd590_1
  - libexpat=2.7.3=h50f4ffc_4
  - libffi=3.4.4=hca03da5_1
  - libzlib=1.3.1=h5f15de7_0
  - ncurses=6.5=hee39554_0
  - openssl=3.0.18=h9b4081a_0
  - pip=25.3=pyhc872135_0
  - python=3.12.12=h2bfc596_1
  - readline=8.3=h0b18652_0
  - setuptools=80.9.0=py312hca03da5_0
  - sqlite=3.51.0=hab6afd1_0
  - tk=8.6.15=hcd8a7d5_0
  - wheel=0.45.1=py312hca03da5_0
  - xz=5.6.4=h80987f9_1
  - zlib=1.3.1=h5f15de7_0
  - pip:
      - absl-py==2.3.1
      - aiohappyeyeballs==2.6.1
      - aiohttp==3.13.2
      - aiosignal==1.4.0
      - alembic==1.17.2
      - altair==6.0.0
      - antlr4-python3-runtime==4.9.3
      - attrs==25.4.0
      - blinker==1.9.0
      - bottle==0.13.4
      - cachetools==6.2.4
      - certifi==2025.11.12
      - charset-normalizer==3.4.4
      - click==8.3.1
      - colorlog==6.10.1
      - contourpy==1.3.3
      - cycler==0.12.1
      - einops==0.8.1
      - einx==0.3.0
      - filelock==3.20.1
      - fonttools==4.61.1
      - frozendict==2.4.7
      - frozenlist==1.8.0
      - fsspec==2025.12.0
      - gitdb==4.0.12
      - gitpython==3.1.45
      - grpcio==1.76.0
      - idna==3.11
      - jinja2==3.1.6
      - joblib==1.5.3
      - jsonschema==4.25.1
      - jsonschema-specifications==2025.9.1
      - kiwisolver==1.4.9
      - lightning==2.6.0
      - lightning-utilities==0.15.2
      - mako==1.3.10
      - markdown==3.10
      - markupsafe==3.0.3
      - matplotlib==3.10.8
      - mpmath==1.3.0
      - multidict==6.7.0
      - narwhals==2.14.0
      - networkx==3.6.1
      - numpy==2.4.0
      - omegaconf==2.3.0
      - optuna==4.6.0
      - optuna-dashboard==0.20.0
      - packaging==25.0
      - pandas==2.3.3
      - pillow==12.0.0
      - propcache==0.4.1
      - protobuf==6.33.2
      - pyarrow==22.0.0
      - pydeck==0.9.1
      - pyparsing==3.3.1
      - python-dateutil==2.9.0.post0
      - pytorch-lightning==2.6.0
      - pytz==2025.2
      - pyyaml==6.0.3
      - referencing==0.37.0
      - requests==2.32.5
      - rpds-py==0.30.0
      - scikit-learn==1.8.0
      - scipy==1.16.3
      - six==1.17.0
      - smmap==5.0.2
      - sqlalchemy==2.0.45
      - streamlit==1.52.2
      - sympy==1.14.0
      - tenacity==9.1.2
      - tensorboard==2.20.0
      - tensorboard-data-server==0.7.2
      - threadpoolctl==3.6.0
      - toml==0.10.2
      - torch==2.9.1
      - torchmetrics==1.8.2
      - torchvision==0.24.1
      - tornado==6.5.4
      - tqdm==4.67.1
      - typing-extensions==4.15.0
      - tzdata==2025.3
      - urllib3==2.6.2
      - werkzeug==3.1.4
      - yarl==1.22.0
prefix: /Users/yu/miniconda3/envs/condinilm

```

---

## Code Structure
```bash
.
├── assets/                 # figures and visual assets
├── configs/                # experiment configuration files (YAML)
├── data/                   # dataset metadata and splits
├── results/                # experiment outputs and logs
├── scripts/                # experiment launch scripts
│   ├── run_one_expe.py
│   └── run_all_expe.sh
├── src/
│   ├── helpers/            # training, metrics, preprocessing
│   ├── baselines/          # NILM baseline models
│   └── film_multinilm/     # CondiNILM core implementation
├── environment.yml
└── README.md
```
---

## Running Experiments
Run a Single Experiment
```bash
python scripts/run_one_expe.py \
    --dataset "UKDALE" \
    --sampling_rate "1min" \
    --appliance "WashingMachine" \
    --window_size 128 \
    --name_model NILMFormer \
    --seed 42
```
Run Full Benchmark Suite
```bash
bash scripts/run_all_expe.sh
```

---
Research Context

CondiNILM was developed in the context of:
	•	Advanced NILM research
	•	Multi-task learning for energy disaggregation
	•	Transformer-based time-series modeling
	•	Device-aware representation learning

The codebase is structured to support ablation studies, loss-function research, and architecture extensions.

---
## Acknowledgement
This project builds upon established NILM research and re-implements several prior baselines for comparison.
All newly introduced architectures, FiLM conditioning mechanisms, multi-task heads, and training strategies are original contributions of this work.

## Contact
**Siyi Li**
M.Sc. Electrical Engineering · TU Braunschweig
For questions or collaborations, please reach out to:
- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [your-username](https://github.com/your-username)