# VibeVoice Project Context

## Project Overview

VibeVoice is a cutting-edge Text-to-Speech (TTS) framework designed for generating expressive, long-form, multi-speaker conversational audio, such as podcasts, from text. It addresses significant challenges in traditional TTS systems, particularly in scalability, speaker consistency, and natural turn-taking.

### Key Features
- Generates speech up to **90 minutes** long with up to **4 distinct speakers**
- Uses continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz
- Employs a next-token diffusion framework with a Large Language Model (LLM) for textual context understanding
- Can synthesize high-fidelity acoustic details through a diffusion head

### Architecture
The model consists of several key components:
1. **Acoustic Tokenizer**: Encodes/decodes audio to/from discrete tokens
2. **Semantic Tokenizer**: Handles semantic representation of speech
3. **Language Model**: Processes text and dialogue flow (based on Qwen2.5 1.5B)
4. **Diffusion Head**: Generates high-fidelity acoustic details
5. **Processor**: Handles input preprocessing and output postprocessing

## Installation

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended)
- Docker (optional but recommended for CUDA environment management)

### Setup Process

1. **Using Docker (Recommended)**
```bash
# Launch NVIDIA PyTorch Container (24.07/24.10/24.12 verified)
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3

# If flash attention is not included in your docker environment, install it manually
# pip install flash-attn --no-build-isolation
```

2. **Install from GitHub**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/
pip install -e .
```

### Dependencies
Key dependencies include:
- torch
- transformers==4.51.3
- diffusers
- librosa
- gradio
- accelerate==1.6.0

## Usage

### Gradio Demo
Launch an interactive web interface for generating podcasts:
```bash
# Install ffmpeg for demo
apt update && apt install ffmpeg -y

# For 1.5B model
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

# For Large model
python demo/gradio_demo.py --model_path aoi-ot/VibeVoice-Large --share
```

### File-based Inference
Generate audio from text files directly:
```bash
# 1 speaker
python demo/inference_from_file.py --model_path aoi-ot/VibeVoice-Large --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# Multiple speakers
python demo/inference_from_file.py --model_path aoi-ot/VibeVoice-Large --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank
```

### Input Format
Text inputs should be formatted as:
```
Speaker 1: Welcome to our podcast today!
Speaker 2: Thanks for having me. I'm excited to discuss...
```

Alternatively, plain text will be auto-assigned to speakers in rotation.

### Voice Samples
Voice samples are stored in `demo/voices/` and include:
- en-Alice_woman.wav
- en-Carter_man.wav
- en-Frank_man.wav
- en-Maya_woman.wav
- And several others for different languages and characteristics

## Project Structure

```
VibeVoice-Community/
├── demo/
│   ├── gradio_demo.py          # Gradio web interface
│   ├── inference_from_file.py  # Command-line inference script
│   ├── text_examples/          # Sample text inputs
│   └── voices/                 # Voice sample audio files
├── vibevoice/
│   ├── modular/                # Core model components
│   │   ├── configuration_vibevoice.py
│   │   ├── modeling_vibevoice.py
│   │   └── modeling_vibevoice_inference.py
│   ├── processor/              # Input preprocessing
│   │   └── vibevoice_processor.py
│   ├── schedule/               # Diffusion scheduling
│   └── scripts/                # Utility scripts
└── pyproject.toml              # Project dependencies and metadata
```

## Development Guidelines

### Code Organization
- Model architecture components are in `vibevoice/modular/`
- Input processing and tokenization are in `vibevoice/processor/`
- Diffusion scheduling is in `vibevoice/schedule/`
- Demo applications are in `demo/`

### Testing
The project includes example scripts for testing:
- `demo/text_examples/` contains various conversation scenarios
- `demo/voices/` contains voice samples for different speakers

### Best Practices
1. For Chinese text, use English punctuation (commas and periods only) for better stability
2. Use the Large model variant for more stable results
3. For long texts, chunk them with multiple speaker turns using the same speaker label if the voice speaks too fast

## Models

| Model | Context Length | Generation Length | Weight |
|-------|----------------|-------------------|--------|
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-Large | 32K | ~45 min | [HF link](https://huggingface.co/aoi-ot/VibeVoice-Large) |

## Important Notes

### Risks and Limitations
- Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused
- English and Chinese only: Other languages may result in unexpected outputs
- Non-Speech Audio: Model focuses solely on speech synthesis
- Overlapping Speech: Current model does not explicitly model overlapping speech segments

### Disclaimer
This model is intended for research and development purposes only. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. It is best practice to disclose the use of AI when sharing AI-generated content.

## Commands Summary

### Building/Installing
```bash
pip install -e .
```

### Running Demos
```bash
# Gradio interface
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

# File-based inference
python demo/inference_from_file.py --model_path aoi-ot/VibeVoice-Large --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank
```

### Testing
Example test scripts are located in `demo/text_examples/` and can be used with the inference scripts.