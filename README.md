# STT FastConformer Hybrid RNN-T ONNX Export & Inference

Export NeMo FastConformer Hybrid RNN-T models to ONNX format and run inference without NeMo dependencies.

## Overview

This project provides:
- **Export script**: Converts NeMo ASR models (encoder, decoder-joint, preprocessor, tokenizer) to ONNX
- **Inference script**: Standalone ONNX-based transcription without NeMo

> **Note**: Custom preprocessing modifications were required to make the featurizer ONNX-exportable (STFT and normalization adjustments).

## Installation

```bash
uv sync
```

## Usage

### 1. Export Model

```bash
uv run python export.py stt_de_fastconformer_hybrid_large_pc -o models
```

This exports:
- `encoder-stt_de_fastconformer_hybrid_large_pc.onnx`
- `decoder_joint-stt_de_fastconformer_hybrid_large_pc.onnx`
- `preprocessor-stt_de_fastconformer_hybrid_large_pc.onnx`
- `tokenizer-stt_de_fastconformer_hybrid_large_pc.model`

### 2. Run Inference

```bash
uv run python onnx_inference.py test_audio.wav \
  -e models/encoder-stt_de_fastconformer_hybrid_large_pc.onnx \
  -d models/decoder_joint-stt_de_fastconformer_hybrid_large_pc.onnx \
  -p models/preprocessor-stt_de_fastconformer_hybrid_large_pc.onnx \
  -t models/tokenizer-stt_de_fastconform_hybrid_large_pc.model
```

## License

See [LICENSE](LICENSE).
