## SIGNL
This is code implementation for Spatio-temporal Vision Graph Non-Contrastive Learning for Audio Deepfake Detection

## Installation
This code needs Python-3.9 or higher.
```bash
pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```
## Dataset

The details to download the datasets are available in the ``./data/`` folder. Once the datasets are downloaded, preprocess all datasets using the command:

```
python
```

## Quick Start

```
python main.py --dataset <dataset>
```