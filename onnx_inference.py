from dataclasses import dataclass
from typing import List, Optional

import librosa
import numpy as np
import onnx
import onnxruntime
import typer


def _is_cuda_available() -> bool:
    """Check if CUDA is available via onnxruntime."""
    available_providers = onnxruntime.get_available_providers()
    return (
        "CUDAExecutionProvider" in available_providers
        or "TensorrtExecutionProvider" in available_providers
    )


@dataclass
class Hypothesis:
    """Hypothesis class for storing decoding results."""

    score: float
    y_sequence: np.ndarray
    length: Optional[List[int]] = None


class ONNXGreedyBatchedRNNTInfer:
    """
    ONNX Greedy Batched RNN-T Inference class.

    This class provides greedy decoding for RNN-T models exported to ONNX format.
    It loads encoder and decoder-joint ONNX models and performs auto-regressive
    decoding to generate transcriptions.

    Args:
        encoder_model: Path to the encoder ONNX model file.
        decoder_joint_model: Path to the decoder-joint ONNX model file.
        preprocessor_model: Path to the preprocessor ONNX model file.
        tokenizer_model: Path to the SentencePiece tokenizer model file.
        max_symbols_per_step: Maximum number of symbols that can be added
            to a sequence in a single time step. Default is 10.
    """

    def __init__(
        self,
        encoder_model: str,
        decoder_joint_model: str,
        preprocessor_model: Optional[str] = None,
        tokenizer_model: Optional[str] = None,
        max_symbols_per_step: Optional[int] = 10,
    ):
        self.encoder_model_path = encoder_model
        self.decoder_joint_model_path = decoder_joint_model
        self.preprocessor_model_path = preprocessor_model
        self.tokenizer_model_path = tokenizer_model
        self.max_symbols_per_step = max_symbols_per_step

        # Will be populated at runtime
        self._blank_index = None
        self.tokenizer = None

        # Setup execution providers
        if _is_cuda_available():
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        onnx_session_opt = onnxruntime.SessionOptions()
        onnx_session_opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Load preprocessor model if provided
        if self.preprocessor_model_path is not None:
            onnx_model = onnx.load(self.preprocessor_model_path)
            onnx.checker.check_model(onnx_model, full_check=True)
            self.preprocessor_model = onnx_model
            self.preprocessor = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(),
                providers=providers,
                sess_options=onnx_session_opt,
            )
            print("Successfully loaded preprocessor onnx model!")
        else:
            self.preprocessor = None
            self.preprocessor_model = None

        # Load encoder model
        onnx_model = onnx.load(self.encoder_model_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.encoder_model = onnx_model
        self.encoder = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(),
            providers=providers,
            sess_options=onnx_session_opt,
        )

        # Load decoder-joint model
        onnx_model = onnx.load(self.decoder_joint_model_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.decoder_joint_model = onnx_model
        self.decoder_joint = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(),
            providers=providers,
            sess_options=onnx_session_opt,
        )

        print("Successfully loaded encoder, decoder and joint onnx models!")

        # Load tokenizer if provided
        if self.tokenizer_model_path is not None:
            import sentencepiece as spm

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.tokenizer_model_path)
            print("Successfully loaded SentencePiece tokenizer!")

        self._setup_encoder_input_output_keys()
        self._setup_decoder_joint_input_output_keys()
        self._setup_blank_index()

    def _setup_encoder_input_output_keys(self):
        """Setup input/output key names from encoder model."""
        self.encoder_inputs = list(self.encoder_model.graph.input)
        self.encoder_outputs = list(self.encoder_model.graph.output)

    def _setup_decoder_joint_input_output_keys(self):
        """Setup input/output key names from decoder-joint model."""
        self.decoder_joint_inputs = list(self.decoder_joint_model.graph.input)
        self.decoder_joint_outputs = list(self.decoder_joint_model.graph.output)

    def _setup_blank_index(self):
        """Determine the blank token index by running a test inference."""
        # ASSUME: Single input with no time length information
        dynamic_dim = 257
        shapes = self.encoder_inputs[0].type.tensor_type.shape.dim
        ip_shape = []
        for shape in shapes:
            if hasattr(shape, "dim_param") and "dynamic" in shape.dim_param:
                ip_shape.append(dynamic_dim)  # replace dynamic axes with constant
            else:
                ip_shape.append(int(shape.dim_value))

        enc_logits, encoded_length = self.run_encoder(
            audio_signal=np.random.randn(*ip_shape).astype(np.float32),
            length=np.random.randint(0, 1, size=(dynamic_dim,)).astype(np.int64),
        )

        # prepare states
        states = self._get_initial_states(batchsize=dynamic_dim)

        # run decoder 1 step
        joint_out, states = self.run_decoder_joint(enc_logits, None, None, *states)
        log_probs, lengths = joint_out

        self._blank_index = (
            log_probs.shape[-1] - 1
        )  # last token of vocab size is blank token
        print(
            f"Enc-Dec-Joint step was evaluated, "
            f"blank token id = {self._blank_index}; vocab size = {log_probs.shape[-1]}"
        )

    def run_encoder(self, audio_signal, length):
        """
        Run encoder network.

        Args:
            audio_signal: Audio signal tensor/array of shape (batch, features, time).
            length: Audio length tensor/array of shape (batch,).

        Returns:
            Tuple of (encoder_output, encoded_length).
        """
        if hasattr(audio_signal, "cpu"):
            audio_signal = audio_signal.cpu().numpy()

        if hasattr(length, "cpu"):
            length = length.cpu().numpy()

        ip = {
            self.encoder_inputs[0].name: audio_signal,
            self.encoder_inputs[1].name: length,
        }
        enc_out = self.encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
        return enc_out, encoded_length

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        """
        Run decoder-joint networks.

        Args:
            enc_logits: Encoder logits.
            targets: Target tokens.
            target_length: Target length.
            states: LSTM states.

        Returns:
            Tuple of (decoder_output, new_states).
        """
        # ASSUME: Decoder is RNN Transducer
        if targets is None:
            targets = np.zeros((enc_logits.shape[0], 1), dtype=np.int32)
            target_length = np.ones(enc_logits.shape[0], dtype=np.int32)

        if hasattr(targets, "cpu"):
            targets = targets.cpu().numpy()

        if hasattr(target_length, "cpu"):
            target_length = target_length.cpu().numpy()

        ip = {
            self.decoder_joint_inputs[0].name: enc_logits,
            self.decoder_joint_inputs[1].name: targets,
            self.decoder_joint_inputs[2].name: target_length,
        }

        num_states = 0
        if states is not None and len(states) > 0:
            num_states = len(states)
            for idx, state in enumerate(states):
                if hasattr(state, "cpu"):
                    state = state.cpu().numpy()

                ip[self.decoder_joint_inputs[len(ip)].name] = state

        dec_out = self.decoder_joint.run(None, ip)

        # unpack dec output
        if num_states > 0:
            new_states = dec_out[-num_states:]
            dec_out = dec_out[:-num_states]
        else:
            new_states = None

        return dec_out, new_states

    def _get_initial_states(self, batchsize):
        """
        Get initial LSTM states.

        Args:
            batchsize: Batch size.

        Returns:
            List of initial state tensors, or None if no states.
        """
        # ASSUME: LSTM STATES of shape (layers, batchsize, dim)
        input_state_nodes = [
            ip for ip in self.decoder_joint_inputs if "state" in ip.name
        ]
        num_states = len(input_state_nodes)
        if num_states == 0:
            return None

        input_states = []
        for state_id in range(num_states):
            node = input_state_nodes[state_id]
            ip_shape = []
            for shape_idx, shape in enumerate(node.type.tensor_type.shape.dim):
                if hasattr(shape, "dim_param") and "dynamic" in shape.dim_param:
                    ip_shape.append(batchsize)  # replace dynamic axes with constant
                else:
                    ip_shape.append(int(shape.dim_value))

            input_states.append(np.zeros(ip_shape, dtype=np.float32))

        return input_states

    def __call__(self, audio_signal, length):
        """
        Perform greedy decoding on audio signal.

        Args:
            audio_signal: Audio features of shape (batch, features, time).
            length: Audio length of shape (batch,).

        Returns:
            List of Hypothesis objects containing decoded sequences.
        """
        # Apply encoder
        encoder_output, encoded_lengths = self.run_encoder(
            audio_signal=audio_signal, length=length
        )

        # Transpose to (B, T, D)
        if isinstance(encoder_output, np.ndarray):
            encoder_output = encoder_output.transpose((0, 2, 1))
        else:
            encoder_output = encoder_output.transpose(1, 2)

        logitlen = encoded_lengths

        inseq = encoder_output  # [B, T, D]
        hypotheses, timestamps = self._greedy_decode(inseq, logitlen)

        # Pack the hypotheses results
        packed_result = [
            Hypothesis(score=-1.0, y_sequence=np.array([], dtype=np.int64))
            for _ in range(len(hypotheses))
        ]
        for i in range(len(packed_result)):
            packed_result[i].y_sequence = np.array(hypotheses[i], dtype=np.int64)
            packed_result[i].length = timestamps[i]

        return packed_result

    def _greedy_decode(self, x, out_len):
        """
        Perform greedy decoding.

        Args:
            x: Encoder output of shape (B, T, D).
            out_len: Output lengths of shape (B,).

        Returns:
            Tuple of (labels, timestamps) where labels is a list of decoded token sequences
            and timestamps is a list of frame indices for each token.
        """
        # x: [B, T, D]
        # out_len: [B]

        # Initialize state
        batchsize = x.shape[0]
        hidden = self._get_initial_states(batchsize)
        target_lengths = np.ones(batchsize, dtype=np.int32)

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = np.full(
            [batchsize, 1], fill_value=self._blank_index, dtype=np.int32
        )

        # Mask buffers
        blank_mask = np.full([batchsize], fill_value=False, dtype=bool)

        # Get max sequence length
        max_out_len = int(out_len.max())
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]

            # Transpose to (B, D, 1)
            if isinstance(f, np.ndarray):
                f = f.transpose((0, 2, 1))
            else:
                f = f.transpose(1, 2)

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask[:] = False

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len

            # Start inner loop
            while not_blank and (
                self.max_symbols_per_step is None
                or symbols_added < self.max_symbols_per_step
            ):
                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g = np.full((batchsize, 1), self._blank_index, dtype=np.int32)
                else:
                    g = last_label.astype(np.int32)

                # Batched joint step - Output = [B, V + 1]
                joint_out, hidden_prime = self.run_decoder_joint(
                    f, g, target_lengths, *hidden
                )
                logp, pred_lengths = joint_out
                logp = logp[:, 0, 0, :]

                # Get index k, of max prob for batch
                k = np.argmax(logp, axis=1).astype(np.int32)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask = blank_mask | k_is_blank

                del k_is_blank
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = np.where(blank_mask)[0]

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[
                                state_id
                            ][:, blank_indices, :]

                    elif len(blank_indices) > 0 and hidden is None:
                        # Reset state if there were some blank and other non-blank predictions in batch
                        # Original state is filled with zeros so we just multiply
                        # LSTM has 2 states
                        for state_id in range(len(hidden_prime)):
                            hidden_prime[state_id][:, blank_indices, :] *= 0.0

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    last_label = k.copy().reshape(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propagate across all timesteps
                    # once they have occurred (normally stopping condition of sample level loop).
                    for kidx, ki in enumerate(k):
                        if not blank_mask[kidx]:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                    symbols_added += 1

        return label, timesteps

    def run_preprocessor(self, audio_signal: np.ndarray, length: np.ndarray):
        """
        Run preprocessor network.

        Args:
            audio_signal: Raw audio signal of shape (batch, samples).
            length: Audio length in samples of shape (batch,).

        Returns:
            Tuple of (processed_features, feature_length).
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor model not loaded")

        if hasattr(audio_signal, "cpu"):
            audio_signal = audio_signal.cpu().numpy()

        if hasattr(length, "cpu"):
            length = length.cpu().numpy()

        audio_signal = audio_signal.astype(np.float32)
        length = length.astype(np.int64)

        ip = {
            "input_signal": audio_signal,
            "length": length,
        }
        features, feature_length = self.preprocessor.run(None, ip)
        return features, feature_length

    def transcribe(self, audio_signal: np.ndarray, length: np.ndarray) -> List[str]:
        """
        Transcribe audio signal to text.

        Args:
            audio_signal: Raw audio signal of shape (batch, samples).
            length: Audio length in samples of shape (batch,).

        Returns:
            List of transcribed strings.
        """
        # Run preprocessor if available
        if self.preprocessor is not None:
            features, feature_length = self.run_preprocessor(audio_signal, length)
        else:
            features, feature_length = audio_signal, length

        # Run decoding
        hypotheses = self(features, feature_length)

        # Decode to text if tokenizer available
        if self.tokenizer is not None:
            results = []
            for hyp in hypotheses:
                predictions = [int(p) for p in hyp.y_sequence if p != self._blank_index]
                text = self.tokenizer.decode_ids(predictions)
                results.append(text)
            return results
        else:
            return [str(hyp.y_sequence.tolist()) for hyp in hypotheses]


app = typer.Typer()


@app.command()
def transcribe(
    audio_file: str = typer.Argument("test_audio.wav", help="Path to audio file to transcribe"),
    encoder_model: str = typer.Option(
        ..., "--encoder", "-e", help="Path to encoder ONNX model"
    ),
    decoder_joint_model: str = typer.Option(
        ..., "--decoder-joint", "-d", help="Path to decoder-joint ONNX model"
    ),
    preprocessor_model: str = typer.Option(
        None, "--preprocessor", "-p", help="Path to preprocessor ONNX model"
    ),
    tokenizer_model: str = typer.Option(
        None, "--tokenizer", "-t", help="Path to SentencePiece tokenizer model"
    ),
    max_symbols_per_step: int = typer.Option(
        5, "--max-symbols", "-m", help="Maximum symbols per step"
    ),
    sample_rate: int = typer.Option(
        16000, "--sample-rate", "-sr", help="Audio sample rate"
    ),
):
    """Transcribe an audio file using ONNX RNN-T model."""

    # Load audio
    audio, _ = librosa.load(audio_file, sr=sample_rate)
    input_signal = audio.reshape(1, -1).astype(np.float32)
    length = np.array([audio.shape[0]], dtype=np.int64)

    # Initialize inference session
    infer = ONNXGreedyBatchedRNNTInfer(
        encoder_model=encoder_model,
        decoder_joint_model=decoder_joint_model,
        preprocessor_model=preprocessor_model,
        tokenizer_model=tokenizer_model,
        max_symbols_per_step=max_symbols_per_step,
    )

    # Transcribe
    results = infer.transcribe(input_signal, length)

    for i, text in enumerate(results):
        typer.echo(f"Transcription {i + 1}: {text}")


if __name__ == "__main__":
    app()
