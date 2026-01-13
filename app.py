import os
import re
import time
import uuid
import librosa
import numpy as np
import onnxruntime as ort
import subprocess
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_sessions
    print("Loading ONNX sessions...")
    providers = ["CPUExecutionProvider"]
    onnx_sessions["preprocessor"] = ort.InferenceSession(PREPROCESSOR_PATH, providers=providers)
    onnx_sessions["encoder"] = ort.InferenceSession(ENCODER_PATH, providers=providers)
    onnx_sessions["proj"] = ort.InferenceSession(PROJ_PATH, providers=providers)
    onnx_sessions["decoder"] = ort.InferenceSession(DECODER_PATH, providers=providers)
    onnx_sessions["tokenizer"] = CanaryOnnxTokenizer(os.path.join(MODEL_DIR, "tokenizer", "vocab.json"))
    yield
    onnx_sessions.clear()

app = FastAPI(title="Canary ONNX Streaming", lifespan=lifespan)

# Paths and constants
MODEL_DIR = "canary"
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.onnx")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.onnx")
PROJ_PATH = os.path.join(MODEL_DIR, "encoder_decoder_proj.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.onnx")

# fixed decoder params (match the exported model)
NUM_STATES = 6
HIDDEN_DIM = 1024
PAD_ID = 2
EOS_ID = 3

sessions: Dict[str, Dict[str, Any]] = {}
onnx_sessions: Dict[str, ort.InferenceSession] = {}

class CanaryOnnxTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

    def ids_to_text(self, ids):
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        text = "".join([self.vocab[i] for i in ids if i < len(self.vocab)])
        text = text.replace('\u2581', ' ')
        text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
        return text.strip()

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Canary Streaming ASR</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
            #transcript { border: 1px solid #ccc; padding: 1rem; min-height: 100px; background: #f9f9f9; margin-top: 1rem; white-space: pre-wrap; }
            .controls { display: flex; gap: 10px; align-items: center; margin-bottom: 1rem; }
            .status { font-size: 0.9rem; color: #666; margin-top: 0.5rem; }
        </style>
    </head>
    <body>
        <h1>Canary ONNX Streaming</h1>
        <p style="color: red; font-size: 0.8rem;">Note: Access via <b>http://localhost:8000</b> to enable microphone access.</p>
        <div class="controls">
            <select id="source_lang"><option value="de">German</option><option value="en">English</option></select>
            <span>&rarr;</span>
            <select id="target_lang"><option value="de">German</option><option value="en">English</option></select>
            <button id="startBtn">Start Streaming</button>
            <button id="stopBtn" disabled>Stop</button>
        </div>
        <div id="status" class="status">Idle</div>
        <div id="transcript"></div>
        <div id="metrics" class="status"></div>

        <script>
            let sessionId = null;
            let mediaRecorder = null;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const transcriptDiv = document.getElementById('transcript');
            const statusDiv = document.getElementById('status');
            const metricsDiv = document.getElementById('metrics');

            startBtn.onclick = async () => {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    statusDiv.innerText = "Error: MediaDevices API not available. Ensure you are using HTTPS or localhost.";
                    console.error("navigator.mediaDevices is undefined. Secure context (HTTPS/localhost) is required.");
                    return;
                }

                try {
                    const formData = new FormData();
                    formData.append('source_lang', document.getElementById('source_lang').value);
                    formData.append('target_lang', document.getElementById('target_lang').value);

                    console.log("Starting stream...");
                    const res = await fetch('/stream/start', { method: 'POST', body: formData });
                    const data = await res.json();
                    sessionId = data.session_id;
                    console.log("Session created:", sessionId);

                    let stream;
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    } catch (mediaErr) {
                        throw new Error("Local microphone access denied or not found: " + mediaErr.message);
                    }

                    mediaRecorder = new MediaRecorder(stream);

                    mediaRecorder.ondataavailable = async (e) => {
                        if (e.data.size > 0 && sessionId) {
                            console.log("Sending chunk of size:", e.data.size);
                            const chunkData = new FormData();
                            chunkData.append('session_id', sessionId);
                            chunkData.append('file', e.data);
                            try {
                                const cRes = await fetch('/stream/chunk', { method: 'POST', body: chunkData });
                                const cData = await cRes.json();
                                transcriptDiv.innerText = cData.partial_transcript || "(listening...)";
                                metricsDiv.innerText = `Likelihood: ${(cData.likelihood * 100).toFixed(2)}%`;
                            } catch (err) {
                                console.error("Chunk upload failed:", err);
                            }
                        }
                    };

                    // Request data every 500ms
                    mediaRecorder.start(500);

                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    statusDiv.innerText = `Streaming (ID: ${sessionId})`;
                    transcriptDiv.innerText = "";
                } catch (err) {
                    console.error("Failed to start stream:", err);
                    statusDiv.innerText = "Error: " + err.message;
                }
            };

            stopBtn.onclick = async () => {
                if (mediaRecorder) mediaRecorder.stop();
                console.log("Stopping stream...");
                const formData = new FormData();
                formData.append('session_id', sessionId);
                const res = await fetch('/stream/stop', { method: 'POST', body: formData });
                const data = await res.json();

                transcriptDiv.innerText = data.final_transcript;
                statusDiv.innerText = "Stopped";
                sessionId = null;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };
        </script>
    </body>
    </html>
    """

# Helper: softmax along last dim (numpy)
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def build_fixed_prompt_ids(source_lang: str, target_lang: str, task: str, pnc: str) -> np.ndarray:
    """
    Return the fixed prompt ids array:
      input_ids = [[ 7, 4, 16, 76, 62, 5, 9, 11, 13 ]]
    Index 3 = source_lang token, index 4 = target_lang token.
    For English<->German swap tokens 76 <-> 62.
    """
    base = [7, 4, 16, 76, 62, 5, 9, 11, 13]
    # swap if source is english and target german (or vice versa)
    if source_lang in ("en", "eng") and target_lang in ("de", "ger"):
        base[3], base[4] = 62, 76
    elif source_lang in ("de", "ger") and target_lang in ("en", "eng"):
        base[3], base[4] = 76, 62
    # If needed, extend with more language mapping rules here.
    return np.array(base, dtype=np.int64).reshape(1, -1)

def run_encoder_on_buffer(audio_buffer: np.ndarray):
    # preprocessor -> encoder -> proj
    sess_pre = onnx_sessions["preprocessor"]
    sess_enc = onnx_sessions["encoder"]
    sess_proj = onnx_sessions["proj"]
    preprocessed_np, length_np = sess_pre.run(None, {"input_signal": audio_buffer.astype(np.float32).reshape(1, -1), "length": np.array([len(audio_buffer)], dtype=np.int64)})
    encoded_np, encoded_length_np = sess_enc.run(None, {"audio_signal": preprocessed_np, "length": length_np})
    projected, = sess_proj.run(None, {"encoder_output": encoded_np.transpose(0, 2, 1)})
    # encoded_mask with shape (1, seq_len)
    encoded_mask_np = (np.arange(encoded_np.shape[2])[None, :] < encoded_length_np[:, None]).astype(np.float32)
    return projected, encoded_mask_np, encoded_np, encoded_length_np

def evaluate_sequence_likelihood(prompt_ids: np.ndarray, prev_tokens: list, projected: np.ndarray, encoded_mask_np: np.ndarray):
    """
    Replay the previous decoded tokens and compute geometric mean probability of that token sequence.
    Returns (likelihood, memories, last_input_token).
    """
    if len(prev_tokens) == 0:
        return 0.0, np.zeros((NUM_STATES, 1, 0, HIDDEN_DIM), dtype=np.float32), None

    memories = np.zeros((NUM_STATES, 1, 0, HIDDEN_DIM), dtype=np.float32)
    sess_dec = onnx_sessions["decoder"]

    sum_logp = 0.0
    i = 0
    input_ids = prompt_ids.astype(np.int64)
    for token in prev_tokens:
        decoder_mask = (input_ids != PAD_ID).astype(np.float32)
        outputs = sess_dec.run(None, {
            "input_ids": input_ids,
            "decoder_mask": decoder_mask,
            "encoder_embeddings": projected,
            "encoder_mask": encoded_mask_np.astype(np.float32),
            "decoder_mems": memories,
            "start_pos": np.array(i, dtype=np.int64)
        })
        logits, memories = outputs[0], outputs[1]
        probs = softmax(logits[:, -1, :])
        p = float(probs[0, int(token)])
        sum_logp += np.log(p + 1e-9)
        i += 1
        input_ids = np.array([[int(token)]], dtype=np.int64)

    geo_mean = float(np.exp(sum_logp / len(prev_tokens)))
    return geo_mean, memories, int(prev_tokens[-1])

def greedy_decode_from_state(prompt_ids: np.ndarray, projected: np.ndarray, encoded_mask_np: np.ndarray, memories: np.ndarray, start_input_id, start_pos: int, max_steps: int):
    """
    Continue greedy decoding from provided memory state and starting input token.
    Returns updated decoded_tokens list and new memories.
    """
    sess_dec = onnx_sessions["decoder"]
    decoded = []
    i = start_pos
    input_ids = start_input_id.reshape(1, -1).astype(np.int64)  # shape (1, 1) or full prompt
    for _ in range(max_steps):
        decoder_mask = (input_ids != PAD_ID).astype(np.float32)
        outputs = sess_dec.run(None, {
            "input_ids": input_ids,
            "decoder_mask": decoder_mask,
            "encoder_embeddings": projected,
            "encoder_mask": encoded_mask_np.astype(np.float32),
            "decoder_mems": memories,
            "start_pos": np.array(i, dtype=np.int64)
        })
        logits, memories = outputs[0], outputs[1]
        next_tokens = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tok = int(next_tokens[0, 0])
        if tok == EOS_ID:
            break
        decoded.append(tok)
        input_ids = next_tokens.astype(np.int64)
        i += 1
    return decoded, memories

@app.post("/stream/start")
async def stream_start(
    source_lang: str = Form("de"),
    target_lang: str = Form("de"),
    task: str = Form("asr"),
    pnc: str = Form("yes"),
    encode_interval_ms: int = Form(250),
    threshold: float = Form(0.7),
    max_decode_steps: int = Form(50)
):
    if "decoder" not in onnx_sessions:
        raise HTTPException(status_code=503, detail="ONNX models not initialized")

    session_id = uuid.uuid4().hex
    prompt_ids = build_fixed_prompt_ids(source_lang, target_lang, task, pnc)
    sessions[session_id] = {
        "raw_audio_bytes": b"",
        "audio_buffer": np.zeros(0, dtype=np.float32),
        "last_encode_time": 0.0,
        "encode_interval_ms": encode_interval_ms,
        "threshold": float(threshold),
        "max_decode_steps": int(max_decode_steps),
        "prompt_ids": prompt_ids,
        "decoded_tokens": [],  # list[int]
        "memories": np.zeros((NUM_STATES, 1, 0, HIDDEN_DIM), dtype=np.float32),
        "last_likelihood": 0.0,
        "projected": None,
        "encoded_mask": None,
        "finished": False
    }
    return {"session_id": session_id}

@app.post("/stream/chunk")
async def stream_chunk(session_id: str = Form(...), file: UploadFile = File(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = sessions[session_id]
    if sess["finished"]:
        raise HTTPException(status_code=400, detail="Session already finished")

    # Load audio chunk robustly using cumulative bytes
    try:
        chunk_data = await file.read()
        sess["raw_audio_bytes"] += chunk_data

        # Decode the cumulative buffer to ensure headers are present
        process = subprocess.Popen(
            ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '16000', '-ac', '1', 'pipe:1'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = process.communicate(input=sess["raw_audio_bytes"])

        if process.returncode != 0:
            # If ffmpeg fails, we might just need more chunks
            return {"partial_transcript": onnx_sessions["tokenizer"].ids_to_text(sess["decoded_tokens"]), "likelihood": sess["last_likelihood"]}

        audio, _ = librosa.load(io.BytesIO(out), sr=16000)
    except Exception as e:
        print(f"Audio processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")

    # Replace processed buffer with the newly decoded full stream
    sess["audio_buffer"] = audio.astype(np.float32)

    # Determine whether to re-run encoder
    now = time.time()
    elapsed_ms = (now - sess["last_encode_time"]) * 1000.0
    if sess["last_encode_time"] == 0.0 or elapsed_ms >= sess["encode_interval_ms"]:
        # run encoder on full buffer
        projected, encoded_mask_np, _, _ = run_encoder_on_buffer(sess["audio_buffer"])
        sess["projected"] = projected
        sess["encoded_mask"] = encoded_mask_np
        sess["last_encode_time"] = now

        # Evaluate previous decoded sequence
        prev_tokens = sess["decoded_tokens"]
        likelihood, eval_memories, last_input_token = evaluate_sequence_likelihood(sess["prompt_ids"], prev_tokens, projected, encoded_mask_np)
        sess["last_likelihood"] = likelihood

        # Decide whether to resume or restart
        resume = False
        if len(prev_tokens) > 0 and likelihood >= sess["threshold"]:
            resume = True
            start_pos = len(prev_tokens)
            memories = eval_memories
            if last_input_token is None:
                # fallback: use last token from prev list
                start_input_id = np.array([[prev_tokens[-1]]], dtype=np.int64)
            else:
                start_input_id = np.array([[last_input_token]], dtype=np.int64)
        else:
            # restart: decode from prompt
            resume = False
            memories = np.zeros((NUM_STATES, 1, 0, HIDDEN_DIM), dtype=np.float32)
            start_pos = 0
            start_input_id = sess["prompt_ids"].astype(np.int64)

        # Greedy decode some steps
        new_decoded, new_memories = greedy_decode_from_state(
            sess["prompt_ids"],
            projected,
            encoded_mask_np,
            memories,
            start_input_id,
            start_pos,
            sess["max_decode_steps"]
        )

        if resume:
            # append only new tokens
            sess["decoded_tokens"].extend(new_decoded)
        else:
            # replace decoded tokens
            sess["decoded_tokens"] = new_decoded

        sess["memories"] = new_memories

    # Return quick status
    text = onnx_sessions["tokenizer"].ids_to_text(sessions[session_id]["decoded_tokens"])
    return {"partial_transcript": text, "likelihood": sess["last_likelihood"]}

@app.get("/stream/poll")
async def stream_poll(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = sessions[session_id]
    text = onnx_sessions["tokenizer"].ids_to_text(sess["decoded_tokens"])
    return {"transcript": text, "likelihood": sess["last_likelihood"], "finished": sess["finished"]}

@app.post("/stream/stop")
async def stream_stop(session_id: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = sessions[session_id]
    sess["finished"] = True
    text = onnx_sessions["tokenizer"].ids_to_text(sess["decoded_tokens"])
    return {"final_transcript": text, "likelihood": sess["last_likelihood"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
