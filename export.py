from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.preprocessing.features import (
    splice_frames,
    normalize_batch,
    CONSTANT,
)
import torch
from torch import nn
import types
import typer


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
    # fix for seq_len = 0 for streaming; if size was 0, it is always padded to 1, and normalizer fails
    seq_len = torch.where(
        seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed
    )

    if self.stft_pad_amount is not None:
        x = torch.nn.functional.pad(
            x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
        ).squeeze(1)

    # dither (only in training mode for eval determinism)
    if self.training and self.dither > 0:
        x += self.dither * torch.randn_like(x)

    # do preemphasis
    if self.preemph is not None:
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(
            0
        ) < seq_len_time.unsqueeze(1)
        x = torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )
        x = x.masked_fill(~timemask, 0.0)

    # disable autocast to get full range of stft values
    with torch.amp.autocast(x.device.type, enabled=False):
        x = self.stft(x)

    # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
    # guard is needed for sqrt if grads are passed through
    guard = 0 if not self.use_grads else CONSTANT
    # x = torch.view_as_real(x)
    x = torch.sqrt(x.pow(2).sum(-1) + guard)

    if self.training and self.nb_augmentation_prob > 0.0:
        for idx in range(x.shape[0]):
            if self._rng.random() < self.nb_augmentation_prob:
                x[idx, self._nb_max_fft_bin :, :] = 0.0

    # get power spectrum
    if self.mag_power != 1.0:
        x = x.pow(self.mag_power)

    # return plain spectrogram if required
    if linear_spec:
        return x, seq_len

    # disable autocast, otherwise it might be automatically casted to fp16
    # on fp16 compatible GPUs and get NaN values for input value of 65520
    with torch.amp.autocast(x.device.type, enabled=False):
        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
    # log features if required
    if self.log:
        if self.log_zero_guard_type == "add":
            x = torch.log(x + self.log_zero_guard_value_fn(x))
        elif self.log_zero_guard_type == "clamp":
            x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
        else:
            raise ValueError("log_zero_guard_type was not understood")

    # frame splicing if required
    if self.frame_splicing > 1:
        x = splice_frames(x, self.frame_splicing)

    # normalize if required
    if self.normalize:
        x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

    # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
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


app = typer.Typer()


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
):
    """Export a NeMo ASR model to ONNX format."""
    model = ASRModel.from_pretrained(model_name, map_location="cpu")
    model.export(f"{target_folder}/{model_name}.onnx")
    model.preprocessor.featurizer.stft = types.MethodType(
        stft, model.preprocessor.featurizer
    )
    model.preprocessor.featurizer.forward = types.MethodType(
        exportable_forward, model.preprocessor.featurizer
    )
    model.preprocessor.export(f"{target_folder}/preprocessor-{model_name}.onnx")
    with open(f"{target_folder}/tokenizer-{model_name}.model", "wb") as f:
        f.write(model.decoding.tokenizer.tokenizer.serialized_model_proto())
    typer.echo(f"Successfully exported {model_name} to {target_folder}")


if __name__ == "__main__":
    app()
