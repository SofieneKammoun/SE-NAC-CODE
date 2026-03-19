# Modeling strategies for speech enhancement in the latent space of a neural audio codec

This repository provides the official implementation of the paper *[Modeling strategies for speech enhancement in the latent space of a neural audio codec](https://arxiv.org/abs/2510.26299)* authored by Sofiene Kammoun, Xavier Alameda-Pineda and Simon Leglaive. We explore different modeling strategies (autoregressive vs. non-autoregressive) and representation spaces (discrete vs. continuous) for speech enhancement using neural audio codecs and Conformer-based architectures.

[arXiv preprint](https://arxiv.org/abs/2510.26299) | [Audio examples](https://sofienekammoun.github.io/SE-NAC-25/) | [Bibtex](#citation)


##  Overview

Our work introduces and compares a family of speech enhancement models that systematically vary along two main axes:

- **Representation Type**  
  - Discrete tokens 
  - Continuous latent vectors  

- **Modeling Strategy**  
  - Autoregressive (AR): Sequential prediction of clean speech representation  
  - Non-Autoregressive (NAR): Parallel prediction of clean speech representation  

The current release includes the following models:

| Model Name | Modeling Strategy |  Input Representation | Output Representation | Trainer Script |
|-------------|------|----------------|----------------|----------------|
| **D-AR** | Autoregressive | Discrete |Discrete | `D_AR_Trainer.py` |
| **D-NAR** | Non-Autoregressive | Discrete |Discrete | `D_NAR_Trainer.py` |
| **D-NAR*** | Non-Autoregressive | Continuous |Discrete | `D_NAR_star_Trainer.py` |
| **C-AR** | Autoregressive | Continuous | Continuous | `C_AR_Trainer.py` |
| **C-NAR** | Non-Autoregressive | Continuous | Continuous | `C_NAR_Trainer.py` |

Additional models:
- **C-FT** (`C-FT_Trainer.py`)  and **D-FT** (`D-FT_Trainer.py`), where we only finetune the NAC's encoder with an MSE loss and a cross-entropy loss, respectively. 
- **STFT-NAR** (`C_NAR_Trainer.py`), where instead of the embeddings of the NAC, we work with STFT representations, and we train the model to output an STFT mask.

##  How the Code Works

### 1. **Base Training Logic (`Trainer.py`)**

The `Trainer` class encapsulates the components shared by all model variants:
- Distributed training setup via `torch.distributed` and `mp.spawn`.
- Core training and validation code.
- Encoding, Decoding, and Tokenization methods using [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (DAC). 

This file also contains the code for: 
- Data loading through a custom dataset class (`labled_AudioDataset`).
- Other helper functions.


---

### 2. **Variant-Specific Trainers** 

Each `[Model Name]_Trainer.py` file provides the full training script for a given model variant by inheriting from the base `Trainer` and redefining the following methods:
- `process_batch_train_audio()`  
  Defines how the model processes noisy/clean audio pairs during training (e.g., encoding with DAC, quantization, and loss computation).
  
- `_denoise_validation()`  
  Implements model-specific inference pipeline to reconstruct clean speech during validation and compute SI-SDR/STOI scores.
  
- `_save_checkpoint()`  
  Saves model weights, optimizer states, and configuration parameters.
  
 while the `Models/` directory contains files that define the corresponding model architectures.
 
### 3. **Audio Codec Integration**
All variants use the same pretrained neural audio codec (DAC) to:
- Encode raw waveforms into either discrete or continuous latent representations.
- Quantize embeddings via a learned vector quantizer.
- Decode discrete or continuous latent representations back to waveforms for objective and perceptual evaluation.

This modular approach allows any compatible codec to be substituted, facilitating future experiments.

##  Example: Running a Training Script

To train a model, simply edit the configuration block at the top of `[Model_Name]_Trainer.py`:

```python
DAC_Model = "DAC_MODELS/weights_16khz.pth"
DATA_PATHS = [
    "Path/to/train/clean",
    "Path/to/train/mixture",
    "Path/to/val/clean",
    "Path/to/val/mixture",
]

...

```

Then launch distributed training:

```bash
python C_AR_Trainer.py
```
All checkpoints and reconstructed audio samples will be automatically saved under the corresponding directories.

## Coming Soon

- Pretrained checkpoints  
- Evaluation metrics and inference scripts

## Acknowledgments
This work was performed using computational resources from the Mésocentre computing center of Université Paris-Saclay, CentraleSupélec, and Ecole Normale Supérieure ParisSaclay, as part of the DEGREASE project (ANR-23-CE23-0009), funded by the French National Research Agency.

Some code in this repository is adapted from the following repositories:

- [RQ-Transformer](https://github.com/lucidrains/RQ-Transformer)
- [rq-vae-transformer](https://github.com/kakaobrain/rq-vae-transformer)
- [conformer](https://github.com/lucidrains/conformer)
- [DAC](https://github.com/descriptinc/descript-audio-codec)

## Citation

If you find this code useful, please star the project and consider citing:
```
@article{kammoun2025modeling,
  title={Modeling strategies for speech enhancement in the latent space of a neural audio codec},
  author={Kammoun, Sofiene and Alameda-Pineda, Xavier and Leglaive, Simon},
  journal={arXiv preprint arXiv:2510.26299},
  year={2025}
}
```
