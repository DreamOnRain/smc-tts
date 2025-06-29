# VITS-CS: ENGLISH-PINYIN CODE-SWITCHING TEXT-TO-SPEECH

To support bilingual speech synthesis, we employ a code-switching TTS model that enables seamless transitions between Chinese and English, while maintaining speaker identity and pronunciation accuracy. The model consists of two independent Single-Language Encoders, one for Chinese and one for English, each processing text input independently to extract language-specific phonetic and prosodic features. These features are then combined using a language-aware fusion mechanism to generate a coherent speech waveform that accommodates both languages.

## Project Structure

```
├── configs/             # Model configuration files
├── filelists/           # Training data file lists
├── text/                # Text processing modules
├── cs_chinese.py        # Chinese text processing
├── models.py            # Original VITS models
├── models_cs.py         # Code-Switching Text Encoder
├── train_cs.py          # Training procedure
├── cs.py                # Infernece procedure
├── data_utils.py        # Data loading and processing
├── evaluation.py        # Evaluation
├── resample.py          # Resamples audio files to a target sample rate
├── matplot.py           # Calculate the statistics of the dataset
├── pitch_controller.py  # Calculate the pitch of the dataset
└── ...
```

## Installation

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Model Initialization
```python
from cs import ocelot_load_models

model = ocelot_load_models()
```

### 2. Speech Synthesis
```python
from cs import ocelot_generation

# Generate speech
message, filepath, sep_text, phonemes, tones, inference_time = ocelot_generation(
    model=model,
    language="English",
    text="Hello",
    speaker_id=0,
    emotion="happy",
    speed=1.0,
    resynthesis=True
)

```

### 3. Command Line Usage
```bash
python cs.py English "Hello" 0 happy
```

### Text Processing

#### Chinese Text Processing
```python
from chinese import get_pinyin_from_text, get_tones_from_text

sep_text, phonemes = get_pinyin_from_text("你好世界")
tones = get_tones_from_text("你好世界")
```

#### English Text Processing
```python
from generation import get_phonemes

phonemes = get_phonemes("English", "Hello world")
```

## Training

### Training Command
```bash
CUDA_VISIBLE_DEVICES=3 python3 train_cs.py
```

## Model Configuration

### Cross-Speaker Emotion Model
- **Config**: `configs/csemotion.json`
- **Checkpoint**: `checkpoints/csemotion/G_1228000.pth`
- **Sampling Rate**: 16kHz

### Chinese Model
- **Config**: `configs/chinese.json`
- **Checkpoint**: `models/chinese.pth`
- **Sampling Rate**: 44.1kHz

### English Emotion Model
- **Config**: `configs/english_emotion.json`
- **Checkpoint**: `models/english_emotion.pth`
- **Sampling Rate**: 16kHz

## Evaluation

### Performance Metrics
- **RTF (Real-Time Factor)**: Inference speed measurement
- **CER (Character Error Rate)**: Text accuracy evaluation

### Evaluation Scripts
```bash
# Evaluate Chinese model
python evaluation.py

# Evaluate English model
python evaluation_english.py

# Evaluate cross-speaker model
python evaluation_cs.py
```

## Citation
```bibtex
@article{vits2021,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
  journal={arXiv preprint arXiv:2106.06103},
  year={2021}
}
```