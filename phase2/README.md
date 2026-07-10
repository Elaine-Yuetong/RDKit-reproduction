# Phase 2: SchNet Reproduction on Colab

This phase runs on Colab GPU only. Do not install PyTorch Geometric into the
local Phase-1 venv.

## Colab Cells

```bash
!git clone <REPO_URL>
%cd RDKit
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
```

```python
import os, torch
os.environ["TORCH"] = torch.__version__.split("+")[0]
os.environ["CUDA"] = "cu" + torch.version.cuda.replace(".", "")
print(os.environ["TORCH"], os.environ["CUDA"])
```

```bash
!pip install -q rdkit
!pip install -q torch_geometric
!pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

SchNet uses `radius_graph`, which requires `torch_cluster`; the wheel index
must match Colab's installed torch and CUDA versions.

```python
import torch
import torch_geometric
import torch_cluster
from torch_geometric.nn.models import SchNet
print("ok", torch.__version__, torch_geometric.__version__)
```

First run a smoke test to verify install -> dataset download -> manifest
validation -> coverage print -> a few training steps -> checkpoint:

```bash
!python phase2/train_schnet.py \
  --splits_dir data/splits \
  --split random \
  --target gap \
  --batch_size 64 \
  --out_dir /content/drive/MyDrive/schnet_runs \
  --smoke
```

Then launch the real 50k run:

```bash
!python phase2/train_schnet.py \
  --splits_dir data/splits \
  --split random \
  --target gap \
  --train_subset 50000 \
  --epochs 100 \
  --batch_size 64 \
  --out_dir /content/drive/MyDrive/schnet_runs
```

On disconnect, rerun the same real-training cell with `--resume` to continue
from the Drive checkpoint.

Expected runtime on a Colab T4 is a few hours for 50k training molecules. The
target band for QM9 gap with 50k train is roughly 0.08-0.12 eV. The Phase-1
baseline to beat is 0.1364 eV on the random split and 0.2904 eV on the
scaffold split.
