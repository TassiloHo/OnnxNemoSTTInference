import json
import types
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import typer
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.preprocessing.features import (
    CONSTANT,
    normalize_batch,
    splice_frames,
)
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process
from torch import nn
from tqdm import tqdm

app = typer.Typer()


def stft(self, x):
    return torch.stft(
        x,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.win_length,
        center=False if self.exact_pad else True,
        window=self.window.to(dtype=torch.float, device=x.device),
        return_complex=False,
        pad_mode="constant",
    )


def exportable_forward(self, x, seq_len, linear_spec=False):
    seq_len_time = seq_len
    seq_len_unfixed = self.get_seq_len(seq_len)
    seq_len = torch.where(
        seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed
    )

    if self.stft_pad_amount is not None:
        x = torch.nn.functional.pad(
            x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
        ).squeeze(1)

    if self.training and self.dither > 0:
        x += self.dither * torch.randn_like(x)

    if self.preemph is not None:
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(
            0
        ) < seq_len_time.unsqueeze(1)
        x = torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )
        x = x.masked_fill(~timemask, 0.0)

    with torch.amp.autocast(x.device.type, enabled=False):
        x = self.stft(x)

    guard = 0 if not self.use_grads else CONSTANT
    x = torch.sqrt(x.pow(2).sum(-1) + guard)

    if self.training and self.nb_augmentation_prob > 0.0:
        for idx in range(x.shape[0]):
            if self._rng.random() < self.nb_augmentation_prob:
                x[idx, self._nb_max_fft_bin :, :] = 0.0

    if self.mag_power != 1.0:
        x = x.pow(self.mag_power)

    if linear_spec:
        return x, seq_len

    with torch.amp.autocast(x.device.type, enabled=False):
        x = torch.matmul(self.fb.to(x.dtype), x)

    if self.log:
        if self.log_zero_guard_type == "add":
            x = torch.log(x + self.log_zero_guard_value_fn(x))
        elif self.log_zero_guard_type == "clamp":
            x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
        else:
            raise ValueError("log_zero_guard_type was not understood")

    if self.frame_splicing > 1:
        x = splice_frames(x, self.frame_splicing)

    if self.normalize:
        x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

    max_len = x.size(-1)
    mask = torch.arange(max_len, device=x.device)
    mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
    x = x.masked_fill(
        mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value
    )
    del mask
    pad_to = self.pad_to
    if pad_to == "max":
        x = nn.functional.pad(
            x, (0, self.max_length - x.size(-1)), value=self.pad_value
        )
    elif pad_to > 0:
        pad_amt = x.size(-1) % pad_to
        if pad_amt != 0:
            x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
    return x, seq_len


# --- CALIBRATION DATA READER ---


def pad_numpy_batch(
    tensor: np.ndarray, target_len: int, axis: int = 2, pad_val: float = 0.0
):
    """Pad a numpy array to target_len along the specified axis."""
    current_len = tensor.shape[axis]
    if current_len >= target_len:
        return tensor

    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, target_len - current_len)
    return np.pad(tensor, pad_width, mode="constant", constant_values=pad_val)


class ASREncoderDataReader(CalibrationDataReader):
    def __init__(self, asr_model, manifest_path, max_rows=200, device="cpu"):
        self.data = []
        print(f"Preparing Encoder Calibration Data from {manifest_path}...")

        # We use the model's preprocessor to convert audio -> spectrograms
        preprocessor = asr_model.preprocessor
        preprocessor.eval().to(device)

        raw_samples = []
        min_time = None

        with open(manifest_path, "r") as f:
            lines = f.readlines()[:max_rows]

        # 1. Process all samples to find Min Time
        for line in tqdm(lines, desc="Loading Audio"):
            try:
                entry = json.loads(line)
                audio_path = entry["audio_filepath"]

                # Load Audio
                audio, sr = librosa.load(audio_path, sr=16000)
                audio_tensor = (
                    torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
                )
                length_tensor = torch.tensor([audio.shape[0]], dtype=torch.int64).to(
                    device
                )

                # Run Preprocessor (Spectrogram generation)
                # Returns: processed_signal, processed_length
                with torch.no_grad():
                    # Check preprocessor signature (some take 2 args, some 1)
                    processed_signal, processed_length = preprocessor(
                        input_signal=audio_tensor, length=length_tensor
                    )

                # processed_signal shape: [1, Features(80), Time]
                curr_time = processed_signal.shape[2]
                if min_time is None or curr_time < min_time:
                    min_time = curr_time

                raw_samples.append(
                    {
                        "audio_signal": processed_signal.cpu().numpy(),
                        "length": processed_length.cpu().numpy(),
                    }
                )

            except Exception as e:
                # print(f"Skipping {audio_path}: {e}")
                continue

        print(f"Min Spectrogram Time found: {min_time}. Center-cropping samples...")

        def center_crop_numpy_batch(tensor: np.ndarray, target_len: int, axis: int = 2):
            """Center-crop a numpy array to target_len along the specified axis."""
            current_len = tensor.shape[axis]
            if current_len <= target_len:
                # If already shorter or equal, return as is (should not happen)
                return tensor
            start = (current_len - target_len) // 2
            end = start + target_len
            slicer = [slice(None)] * tensor.ndim
            slicer[axis] = slice(start, end)
            return tensor[tuple(slicer)]

        # 2. Center-crop and Store
        for sample in raw_samples:
            spec = sample["audio_signal"]
            length = sample["length"]

            # Center-crop Spectrogram to min_time
            spec_cropped = center_crop_numpy_batch(spec, min_time, axis=2)

            inputs = {"audio_signal": spec_cropped, "length": length}
            self.data.append(inputs)

        self.iterator = iter(self.data)
        self._progress_bar = None
        self._total = len(self.data)
        self._count = 0

    def get_next(self) -> dict:
        if self._progress_bar is None:
            self._progress_bar = tqdm(
                total=self._total, desc="Quantization Progress", unit="sample"
            )
        item = next(self.iterator, None)
        if item is not None:
            self._progress_bar.update(1)
            self._count += 1
            if self._count == self._total:
                self._progress_bar.close()
        else:
            if self._progress_bar is not None:
                self._progress_bar.close()
        return item


# --- QUANTIZATION FUNCTIONS ---


def stt_dynamic_quantization(onnx_model_path: str, quantized_model_path: str):
    """Quantize an ONNX model dynamically (Decoder)."""
    print(f"Applying Dynamic Quantization to {onnx_model_path}...")
    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        op_types_to_quantize=[
            "MatMul",
            "Attention",
            "LSTM",
            "Gather",
            "Transpose",
            "EmbedLayerNormalization",
        ],
    )


def stt_static_quantization(
    onnx_model_path: str, quantized_model_path: str, data_reader
):
    """Quantize an ONNX model statically (Encoder)."""
    print(f"Applying Static Quantization to {onnx_model_path}...")

    # Pre-process (skip symbolic shape to avoid inference errors)
    quant_pre_process(onnx_model_path, onnx_model_path, skip_symbolic_shape=True)

    extra_options = {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
    }

    quantize_static(
        onnx_model_path,
        quantized_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        # IMPORTANT: Include 'Conv' and use per_channel for ASR Encoders
        op_types_to_quantize=[
            "MatMul",
            "Attention",
            "LSTM",
            "Gather",
            "Transpose",
            "EmbedLayerNormalization",
            "Conv",
        ],
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options=extra_options,
    )


def extract_vocab_from_tokenizer(tokenizer) -> dict:
    """Extract vocabulary from tokenizer to a dictionary."""
    vocab = {}
    for i in range(tokenizer.vocab_size):
        piece = tokenizer.ids_to_tokens([i])[0]
        vocab[piece] = i
    return vocab


@app.command()
def export(
    model_name: str = typer.Argument(
        "stt_de_fastconformer_hybrid_large_pc", help="Name of the ASR model to export"
    ),
    target_folder: str = typer.Option(
        "models",
        "--output",
        "-o",
        help="Target folder for exported models",
    ),
    manifest_path: Optional[str] = typer.Option(
        None,
        "--manifest",
        "-m",
        help=(
            "Path to manifest.json for static quantization calibration (optional). "
            "If omitted or not found, the encoder will fall back to dynamic quantization "
            "without convolution ops."
        ),
    ),
):
    """Export a NeMo ASR model to ONNX format with mixed quantization."""
    model_path = Path(model_name)
    model_basename = model_path.name
    model_subdir = model_path.parent
    export_target = f"{target_folder}/{model_name}.onnx"
    Path(export_target).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    model = ASRModel.from_pretrained(model_name, map_location="cpu")

    # 2. Patch Preprocessor for ONNX Export
    # We save a reference to the original methods if we needed them,
    # but for calibration we can use the patched ones provided we handle tensors correctly.
    model.preprocessor.featurizer.stft = types.MethodType(
        stft, model.preprocessor.featurizer
    )
    model.preprocessor.featurizer.forward = types.MethodType(
        exportable_forward, model.preprocessor.featurizer
    )

    # 3. Export Base Model (Splits into Encoder/Decoder automatically)
    print(f"Exporting base model to {export_target}...")
    model.export(export_target)

    # 4. Export Preprocessor
    model.preprocessor.export(
        f"{target_folder}/{model_subdir}/preprocessor-{model_basename}.onnx"
    )

    # 5. Export Tokenizer
    with open(
        f"{target_folder}/{model_subdir}/tokenizer-{model_basename}.model", "wb"
    ) as f:
        f.write(model.decoding.tokenizer.tokenizer.serialized_model_proto())

    vocab = extract_vocab_from_tokenizer(model.decoding.tokenizer)
    vocab_path = f"{target_folder}/{model_subdir}/tokenizer-{model_basename}.vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 6. Quantization
    print("Starting Quantization...")

    # Define prefixes generated by NeMo export
    prefixes = ["encoder-", "decoder_joint-"]

    for prefix in prefixes:
        part_export_target = f"{target_folder}/{prefix}{model_name}.onnx"
        quantized_target = f"{target_folder}/{prefix}{model_name}_qInt8.onnx"

        if not Path(part_export_target).exists():
            print(f"Warning: Could not find {part_export_target}, skipping.")
            continue

        if "encoder" in prefix:
            # --- STATIC QUANTIZATION FOR ENCODER (or fallback to dynamic) ---
            print(f"Detected Encoder: {part_export_target}")

            # If no manifest provided or file missing, fall back to dynamic quantization
            if not manifest_path or not Path(manifest_path).exists():
                print(
                    f"No calibration manifest provided or file not found ({manifest_path}). "
                    "Falling back to dynamic quantization for encoder (no Conv quantization)."
                )
                stt_dynamic_quantization(part_export_target, quantized_target)
            else:
                try:
                    # Initialize Data Reader; if it ends up with no samples, also fall back
                    dr = ASREncoderDataReader(model, manifest_path)
                    if len(dr.data) == 0:
                        print(
                            "Calibration manifest contains no usable samples. "
                            "Falling back to dynamic quantization for encoder."
                        )
                        stt_dynamic_quantization(part_export_target, quantized_target)
                    else:
                        stt_static_quantization(
                            part_export_target, quantized_target, dr
                        )
                except Exception as e:
                    print(
                        f"Encoder static quantization failed ({e}). "
                        "Falling back to dynamic quantization for encoder."
                    )
                    stt_dynamic_quantization(part_export_target, quantized_target)
        else:
            # --- DYNAMIC QUANTIZATION FOR DECODER ---
            print(f"Detected Decoder: {part_export_target}")
            stt_dynamic_quantization(part_export_target, quantized_target)

    typer.echo(f"Extracted {len(vocab)} tokens to {vocab_path}")
    typer.echo(f"Successfully exported {model_name} to {target_folder}")


if __name__ == "__main__":
    app()
