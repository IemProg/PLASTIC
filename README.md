<div align="center">
  <div>
  <h1>Enhancing Plasticity for First Session Adaptation Continual Learning</h1> 

[![Static Badge](https://img.shields.io/badge/PLASTIC-arXiv-red)](https://arxiv.org/abs/2310.11482)
[![Conference](https://img.shields.io/badge/CoLLAs-2025-blue)](https://arxiv.org/abs/2310.11482)


  </div>

</div>

<div align="center">

<div>
    <a href='https://iemprog.github.io/' target='_blank'>Imad Eddine MAROUF</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.it/citations?user=YfzgrDYAAAAJ&hl=en' target='_blank'>Subhankar Roy</a><sup>2</sup>&emsp;
    <a href='https://stelat.eu/' target='_blank'>Stéphane Lathuilière</a><sup>1,3</sup>
    <a href='https://enzotarta.github.io/' target='_blank'>Enzo Tartaglione</a><sup>1</sup>&emsp;
</div>
<div>
<sup>1</sup>Télécom-Paris, Institut Polytechnique de Paris, France &emsp;
<sup>2</sup>University of Bergamo, Italy&emsp;
<sup>3</sup>Inria at University Grenoble Alpes, France&emsp;
</div>
</div>


The official code repository for "[Enhancing Plasticity for First Session Adaptation Continual Learning](https://arxiv.org/abs/2310.11482)" in PyTorch.

📣 **Accepted as a conference paper at CoLLAs 2025**

---

## Abstract

The integration of large pre-trained models (PTMs) into Class-Incremental Learning (CIL) has facilitated the development of compute-efficient strategies such as First-Session Adaptation (FSA), which fine-tunes the model solely on the first task while keeping it frozen for subsequent tasks. Although effective in homogeneous task sequences, these approaches struggle when faced with the heterogeneity of real-world task distributions. We introduce **PLASTIC** (Plasticity-Enhanced Test-Time Adaptation in Class-Incremental Learning), a method that reinstates plasticity in CIL while preserving model stability. PLASTIC leverages Test-Time Adaptation (TTA) by dynamically fine-tuning LayerNorm parameters on unlabeled test data, enabling adaptability to evolving tasks and improving robustness against data corruption. To prevent TTA-induced model divergence and maintain stable learning across tasks, we introduce a teacher-student distillation framework, ensuring that adaptation remains controlled and generalizable. Extensive experiments across multiple benchmarks demonstrate that PLASTIC consistently outperforms both conventional and state-of-the-art PTM-based CIL approaches, while also exhibiting inherent robustness to data corruptions.


<p align="center">
  <img src="public/pipeline_plot.png" alt="PLASTIC Design" />
</p>

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/PLASTIC.git
cd PLASTIC
```

2. **Create environment** (conda recommended)
```bash
conda env create -f environment.yml
conda activate cil
```

Alternatively, you can install the key dependencies manually:
```bash
# Python 3.10 with PyTorch 1.11.0 and CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install timm==0.6.12 wandb opencv-python scikit-learn tqdm
```

3. **Configure WandB** (optional but recommended for logging)
```bash
wandb login
# Or set environment variable: export WANDB_API_KEY=your_key
```

Edit `main.py` line 123 to set your WandB entity:
```python
entity="your-entity-name",  # Change from "imad-ma" to your entity
```

To disable WandB logging entirely, comment out lines 116-125 in `main.py`.

### Running PLASTIC on CIFAR-100

**Our Method (PLASTIC with Test-Time Adaptation):**
```bash
python main.py --config ./exps/adapt_tta.json \
    --dataset cifar224 \
    --experiment_name PLASTIC_CIFAR100_10Tasks \
    --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k_adapter \
    --layers norm \
    --increment 10 \
    --init_cls 10 \
    --selection_p 0.5 \
    --niter 1 \
    --N_augment 8 \
    --batch_tta \
    --random_aug \
    --start_TTA 0
```

**Baseline (SimpleCIL without TTA):**
```bash
python main.py --config ./exps/simplecil.json \
    --dataset cifar224 \
    --experiment_name SimpleCIL_CIFAR100_10Tasks \
    --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k \
    --increment 10 \
    --init_cls 10
```

See `launch.sh` for more examples.

---

## 📊 Supported Datasets

Datasets are automatically downloaded to `./data/` on first run. The following datasets are supported:

| Dataset | Name in Code | Classes | Download |
|---------|-------------|---------|----------|
| CIFAR-100 (32×32) | `cifar100` | 100 | Automatic |
| CIFAR-100 (224×224) | `cifar224` | 100 | Automatic |
| CIFAR-10 | `cifar10` | 10 | Automatic |
| ImageNet-100 | `imagenet100` | 100 | Manual* |
| ImageNet-1000 | `imagenet1000` | 1000 | Manual* |
| ImageNet-R | `imagenetr` | 200 | Manual* |
| ImageNet-A | `imageneta` | 200 | Manual* |
| CUB-200 | `cub` | 200 | Manual* |
| ObjectNet | `objectnet` | 113 | Manual* |
| VTAB | `vtab` | Mixed | Manual* |
| OmniBenchmark | `omnibenchmark` | Mixed | Manual* |

\* For datasets requiring manual download, place them in `./data/` with the following structure:
```
data/
├── cifar-100-python/          # Auto-downloaded
├── imagenet/                  # Manual: train/ and val/ subdirs
├── imagenet-r/                # Manual: images organized by class
├── cub/                       # Manual: CUB_200_2011/images/
└── ...
```

### Corrupted Datasets
For robustness evaluation, we support corrupted versions:
- `cifar100-c`, `cifar10-c` (CIFAR-100-C/10-C with various corruptions)
- Individual corruptions: `cifar100-c_shotnoise`, `cifar100-c_impulse`, `cifar100-c_fog`, etc.
- `vtab-c_gaussian`, `vtab-c_shot`, `vtab-c_impulse`

---

## 🧪 Available Methods

This codebase implements PLASTIC and several state-of-the-art CIL baselines:

| Method | Config File | Description | Paper |
|--------|------------|-------------|-------|
| **PLASTIC (Ours)** | `adapt_tta.json` | Test-Time Adaptation with LayerNorm tuning | [Link](https://arxiv.org/abs/2310.11482) |
| SimpleCIL | `simplecil.json` | Simple Continual Learning baseline | [Link](https://arxiv.org/abs/2303.07338) |
| FECAM | `fecam.json` | Forward Compatible Mean | [Link](https://arxiv.org/abs/2203.16562) |
| ADAM | `adam_adapter.json` | Adaptive Aggregation | [Link](https://arxiv.org/abs/2303.07426) |
| CodaPrompt | `coda_prompt.json` | Prompt-based learning | [Link](https://arxiv.org/abs/2211.13223) |
| RanPAC | `ranpac.json` | Random Path Aggregation | [Link](https://arxiv.org/abs/2307.02911) |
| Finetune | `finetune.json` | Standard fine-tuning baseline | - |
| EWC | `ewc.json` | Elastic Weight Consolidation | [Link](https://arxiv.org/abs/1612.00796) |

---

## ⚙️ Configuration Guide

### Key Hyperparameters

**General Settings:**
- `--dataset`: Dataset name (e.g., `cifar224`, `imagenetr`)
- `--init_cls`: Number of classes in the first task (default: 10)
- `--increment`: Number of classes per subsequent task (default: 10)
- `--seed`: Random seed for reproducibility
- `--batch_size`: Training/testing batch size

**Model Architecture:**
- `--convnet_type`: Backbone architecture
  - `pretrained_vit_b16_224_in21k_adapter`: ViT-B/16 with adapters (PLASTIC)
  - `pretrained_vit_b16_224_in21k`: Standard ViT-B/16 (SimpleCIL)
- `--layers`: Which layers to adapt during TTA
  - `norm`: LayerNorm only (recommended for PLASTIC)
  - `adapter`: Adapter modules only
  - `head`: Classification head only
  - `all`: All parameters

**PLASTIC-Specific (Test-Time Adaptation):**
- `--start_TTA`: Task index to start TTA (0 = from first task)
- `--niter`: Number of TTA iterations per batch (default: 1)
- `--selection_p`: Percentile threshold for entropy-based sample selection (default: 0.5)
- `--N_augment`: Number of augmented views per sample (default: 8)
- `--batch_tta`: Enable batch-level TTA
- `--random_aug`: Use random augmentations during TTA
- `--e_margin`: Margin for entropy threshold
- `--enable_ema`: Enable exponential moving average for model recovery
- `--kl_weight`: Weight for KL divergence in teacher-student distillation

**Training Settings:**
- `--epochs` / `tuned_epoch`: Number of training epochs for first task
- `--init_lr`: Initial learning rate
- `--weight_decay`: Weight decay for optimizer
- `--optimizer`: Optimizer choice (`sgd`, `adam`, `adamw`)

### Creating Custom Configurations

1. Copy an existing config from `exps/`:
```bash
cp exps/adapt_tta.json exps/my_custom_config.json
```

2. Edit the JSON file with your settings:
```json
{
    "dataset": "cifar224",
    "init_cls": 10,
    "increment": 10,
    "model_name": "adapt_tta",
    "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
    "tuned_epoch": 20,
    "batch_size": 256,
    ...
}
```

3. Run with your custom config:
```bash
python main.py --config ./exps/my_custom_config.json --dataset cifar224
```

---

## 📈 Reproducing Paper Results

### Main Experiments (Table 1 in paper)

**CIFAR-100 (10 tasks, 10 classes each):**
```bash
# PLASTIC (Ours)
python main.py --config ./exps/adapt_tta.json --dataset cifar224 \
    --experiment_name PLASTIC_CIFAR100 --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k_adapter --layers norm \
    --increment 10 --init_cls 10 --selection_p 0.5 --niter 1 --N_augment 8 \
    --batch_tta --random_aug --start_TTA 0

# SimpleCIL Baseline
python main.py --config ./exps/simplecil.json --dataset cifar224 \
    --experiment_name SimpleCIL_CIFAR100 --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k --increment 10 --init_cls 10
```

**ImageNet-R (20 tasks, 10 classes each):**
```bash
# PLASTIC (Ours)
python main.py --config ./exps/adapt_tta.json --dataset imagenetr \
    --experiment_name PLASTIC_ImgnetR_20Tasks --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k_adapter --layers norm \
    --increment 10 --init_cls 10 --selection_p 0.5 --niter 1 --N_augment 8 \
    --batch_tta --random_aug --start_TTA 0 --print_forget
```

**ImageNet-A:**
```bash
python main.py --config ./exps/adapt_tta.json --dataset imageneta \
    --experiment_name PLASTIC_ImgnetA --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k_adapter --layers norm \
    --increment 10 --selection_p 0.5 --niter 1 --N_augment 8 \
    --batch_tta --random_aug --start_TTA 0
```

### Robustness Experiments (Table 2 in paper)

**CIFAR-100-C (Corrupted):**
```bash
python main.py --config ./exps/adapt_tta.json --dataset cifar100-c \
    --experiment_name PLASTIC_CIFAR100C_Robustness --seed 0 \
    --convnet_type pretrained_vit_b16_224_in21k_adapter --layers norm \
    --increment 10 --selection_p 0.5 --niter 1 --N_augment 8 \
    --batch_tta --random_aug --start_TTA 0
```

### Ablation Studies

**Different adaptation layers:**
```bash
# LayerNorm only (best performance)
--layers norm

# Adapter modules only
--layers adapter

# All parameters
--layers all
```

**Different TTA start points:**
```bash
# From first task (best)
--start_TTA 0

# From second task
--start_TTA 1
```

### Expected Results

Final average accuracy on CIFAR-100 (after 10 tasks):
- **PLASTIC (Ours)**: ~73-75%
- SimpleCIL: ~68-70%
- FECAM: ~69-71%

*Note: Results may vary slightly based on random seed and hardware.*

---

## 🔧 Hardware Requirements

- **GPU**: NVIDIA GPU with at least 16GB VRAM (tested on RTX 3090, A100)
- **CUDA**: 11.3 or compatible
- **RAM**: At least 32GB system RAM recommended
- **Storage**: ~50GB for datasets and checkpoints

For GPUs with less VRAM, reduce batch size:
```bash
--batch_size 128  # or lower
```

---

## 📝 Project Structure

```
PLASTIC/
├── main.py                     # Main entry point
├── trainer.py                  # Training orchestration
├── launch.sh                   # Example launch commands
├── environment.yml             # Conda environment specification
├── models/                     # CIL method implementations
│   ├── adapt_tta.py           # PLASTIC (our method)
│   ├── simplecil.py           # SimpleCIL baseline
│   ├── fecam.py               # FECAM baseline
│   ├── base.py                # Base learner class
│   └── ...                    # Other baselines
├── convs/                      # Network architectures
│   ├── vision_transformer_adapter.py  # ViT with adapters
│   └── ...
├── utils/                      # Utilities
│   ├── data_manager.py        # Dataset management
│   ├── data.py                # Dataset implementations
│   ├── factory.py             # Model factory
│   └── toolkit.py             # Helper functions
├── exps/                       # Configuration files
│   ├── adapt_tta.json         # PLASTIC config
│   ├── simplecil.json         # SimpleCIL config
│   └── ...
└── public/                     # Images and visualizations
```

---

## 📖 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{marouf2025plastic,
  title={Enhancing Plasticity for First Session Adaptation Continual Learning},
  author={Marouf, Imad Eddine and Roy, Subhankar and Lathuili{\`e}re, St{\'e}phane and Tartaglione, Enzo},
  booktitle={Conference on Lifelong Learning Agents (CoLLAs)},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This work was supported by:
- Télécom-Paris, Institut Polytechnique de Paris
- University of Bergamo
- Inria at University Grenoble Alpes

We thank the authors of [PyCIL](https://github.com/G-U-N/PyCIL), [FeCAM](https://github.com/dipamgoswami/FECAM), and other baseline methods for their open-source implementations.

---

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

---

## 🐛 Troubleshooting

**Issue: WandB authentication errors**
- Solution: Run `wandb login` or set `WANDB_MODE=offline`

**Issue: CUDA out of memory**
- Solution: Reduce `--batch_size` or use gradient accumulation

**Issue: Dataset not found**
- Solution: Ensure dataset is in `./data/` directory. For manual datasets, check directory structure matches expected format.

**Issue: Pre-trained model download fails**
- Solution: Models are downloaded automatically from timm. Ensure internet connection and try again.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/yourusername/PLASTIC/issues)
- Contact: [Imad Eddine Marouf](https://iemprog.github.io/)

---

**Star ⭐ this repository if you find it helpful!**
