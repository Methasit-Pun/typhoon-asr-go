#!/usr/bin/env python3
"""
Typhoon ASR Python bridge server.

Reads newline-delimited JSON requests from stdin, performs ASR inference,
and writes newline-delimited JSON responses to stdout.

The Go gRPC wrapper spawns this process once; the NeMo model is loaded on
first request and kept in memory for the lifetime of the process.

Protocol
--------
Request  (stdin, one JSON object per line):
  {
    "request_id":      "<str>",
    "action":          "ping" | "transcribe"  (default: "transcribe"),
    "audio_path":      "<absolute path>",     // mutually exclusive
    "audio_b64":       "<base64 bytes>",      // with audio_path
    "model_name":      "<str>",               // optional override
    "with_timestamps": <bool>,
    "device":          "auto" | "cpu" | "cuda",
    "language_hint":   "th" | "en" | "auto"
  }

Response (stdout, one JSON object per line):
  {
    "request_id":      "<str>",
    "text":            "<str>",
    "timestamps":      [{"word":"<w>","start":<f>,"end":<f>,"confidence":<f>}],
    "confidence":      <float 0-1>,
    "audio_duration":  <float seconds>,
    "processing_time": <float seconds>,
    "model_loaded":    <bool>,
    "device":          "<str>",
    "error":           "<str>"   // omitted on success
  }
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy model holder
# ---------------------------------------------------------------------------

_model = None
_model_name: str = "scb10x/typhoon-asr-realtime"
_device: str = "auto"


def _load_model():
    global _model, _device
    import torch
    import nemo.collections.asr as nemo_asr

    if _device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"

    _log(f"loading model '{_model_name}' on {_device.upper()} …")
    _model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=_model_name,
        map_location=_device,
    )
    _log(f"model loaded on {_device.upper()}")


def _ensure_model(model_name: str = None):
    global _model, _model_name
    if model_name and model_name != _model_name:
        # Model override: reload (rare in production).
        _model = None
        _model_name = model_name
    if _model is None:
        _load_model()
    return _model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _prepare_audio(source_path: str, target_sr: int = 16000) -> tuple[str, float]:
    """Resample and normalise audio; return (tmp_wav_path, duration_seconds)."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(source_path, sr=None)
    duration = len(y) / sr

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    y = y / (max(abs(y)) + 1e-8)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, target_sr)
    return tmp.name, duration


def _infer(audio_path: str, with_timestamps: bool, model_name: str = None) -> dict:
    import soundfile as sf

    model = _ensure_model(model_name)
    tmp_path, audio_duration = _prepare_audio(audio_path)

    try:
        start = time.time()

        if with_timestamps:
            hypotheses = model.transcribe(audio=[tmp_path], return_hypotheses=True)
            text = ""
            if hypotheses and hasattr(hypotheses[0], "text"):
                text = hypotheses[0].text
            elif hypotheses and isinstance(hypotheses[0], list) and hypotheses[0]:
                text = hypotheses[0][0].text if hasattr(hypotheses[0][0], "text") else str(hypotheses[0][0])
        else:
            results = model.transcribe(audio=[tmp_path])
            text = results[0] if results else ""
            if hasattr(text, "text"):
                text = text.text

        processing_time = time.time() - start

        # Estimated word timestamps (linear distribution — same as src/typhoon_asr_inference.py).
        timestamps = []
        if with_timestamps and text and audio_duration > 0:
            words = text.split()
            if words:
                avg_dur = audio_duration / len(words)
                for i, word in enumerate(words):
                    timestamps.append({
                        "word":       word,
                        "start":      round(i * avg_dur, 3),
                        "end":        round((i + 1) * avg_dur, 3),
                        "confidence": 0.0,  # NeMo basic does not expose per-word confidence
                    })

        # Heuristic confidence: penalise empty transcription.
        confidence = 0.9 if text.strip() else 0.0

        return {
            "text":            text,
            "timestamps":      timestamps,
            "confidence":      confidence,
            "audio_duration":  round(audio_duration, 3),
            "processing_time": round(processing_time, 3),
            "model_loaded":    True,
            "device":          _device,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _log(msg: str):
    """Write to stderr so it doesn't interfere with the stdout JSON protocol."""
    print(f"[bridge] {msg}", file=sys.stderr, flush=True)


def _respond(request_id: str, payload: dict):
    payload["request_id"] = request_id
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(req: dict):
    request_id = req.get("request_id", "")
    action     = req.get("action", "transcribe")

    if action == "ping":
        _respond(request_id, {
            "model_loaded": _model is not None,
            "device":       _device,
        })
        return

    # Resolve audio source.
    audio_path = req.get("audio_path", "")
    audio_b64  = req.get("audio_b64",  "")
    tmp_from_b64 = None

    if audio_b64:
        raw = base64.b64decode(audio_b64)
        suffix = ".wav"
        # Sniff format from magic bytes.
        if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb":
            suffix = ".mp3"
        elif raw[:4] == b"fLaC":
            suffix = ".flac"
        elif raw[:4] == b"OggS":
            suffix = ".ogg"
        tmp_from_b64 = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_from_b64.write(raw)
        tmp_from_b64.close()
        audio_path = tmp_from_b64.name
    elif not audio_path:
        _respond(request_id, {"error": "must supply audio_path or audio_b64", "model_loaded": _model is not None, "device": _device})
        return

    try:
        result = _infer(
            audio_path      = audio_path,
            with_timestamps = bool(req.get("with_timestamps", False)),
            model_name      = req.get("model_name") or None,
        )
        _respond(request_id, result)
    except Exception as exc:
        _log(f"inference error for {request_id}: {traceback.format_exc()}")
        _respond(request_id, {
            "error":        str(exc),
            "model_loaded": _model is not None,
            "device":       _device,
        })
    finally:
        if tmp_from_b64:
            try:
                os.unlink(tmp_from_b64.name)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Typhoon ASR Python bridge server")
    parser.add_argument("--model",  default="scb10x/typhoon-asr-realtime", help="NeMo model name")
    parser.add_argument("--device", default="auto",                         help="auto|cpu|cuda")
    args = parser.parse_args()

    global _model_name, _device
    _model_name = args.model
    _device     = args.device

    _log(f"bridge started (model={_model_name}, device={_device})")
    _log("ready — waiting for JSON requests on stdin")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"JSON parse error: {exc} — line: {line[:120]}")
            continue
        _handle(req)

    _log("stdin closed, bridge exiting")


if __name__ == "__main__":
    main()
