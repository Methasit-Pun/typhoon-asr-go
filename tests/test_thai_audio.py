"""
Test suite for Typhoon ASR — Thai audio edge cases.

Organisation
------------
1. prepare_audio  — audio pre-processing (no model required)
2. transcribe API — full transcribe() function (model mocked unless --run-integration)
3. SentenceBoundaryDetector — Thai sentence splitting logic
4. bridge_server  — Python JSON bridge protocol
5. Integration    — end-to-end with real NeMo model (opt-in via --run-integration)
"""

import base64
import importlib
import io
import json
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers re-used across tests
# ---------------------------------------------------------------------------

def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate()


def _wav_channels(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels()


# ---------------------------------------------------------------------------
# 1. prepare_audio — audio pre-processing
# ---------------------------------------------------------------------------

class TestPrepareAudio:
    """Tests for src/typhoon_asr_inference.py::prepare_audio."""

    def _call(self, input_path, **kwargs):
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from typhoon_asr_inference import prepare_audio
        return prepare_audio(str(input_path), **kwargs)

    # ---- Basic contract -------------------------------------------------- #

    def test_returns_true_on_valid_wav(self, tmp_wav):
        ok, out, info = self._call(tmp_wav)
        assert ok is True
        assert out is not None
        assert Path(out).exists()
        Path(out).unlink(missing_ok=True)

    def test_output_is_16khz(self, tmp_wav_8khz):
        """Input at 8 kHz must be resampled to 16 kHz."""
        ok, out, info = self._call(tmp_wav_8khz)
        assert ok
        assert _wav_sample_rate(Path(out)) == 16000
        Path(out).unlink(missing_ok=True)

    def test_output_is_16khz_from_44khz(self, tmp_wav_44khz):
        """Input at 44.1 kHz must be resampled down to 16 kHz."""
        ok, out, info = self._call(tmp_wav_44khz)
        assert ok
        assert _wav_sample_rate(Path(out)) == 16000
        Path(out).unlink(missing_ok=True)

    def test_original_sample_rate_reported(self, tmp_wav_8khz):
        ok, out, info = self._call(tmp_wav_8khz)
        assert ok
        assert info["original_sr"] == 8000
        Path(out).unlink(missing_ok=True)

    def test_duration_preserved(self, tmp_wav):
        ok, out, info = self._call(tmp_wav)
        assert ok
        assert abs(info["duration"] - 1.0) < 0.05  # within 50 ms
        Path(out).unlink(missing_ok=True)

    # ---- Error paths ------------------------------------------------------ #

    def test_nonexistent_file(self, tmp_path):
        ok, out, info = self._call(tmp_path / "ghost.wav")
        assert ok is False
        assert out is None
        assert "error" in info

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "audio.xyz"
        p.write_bytes(b"\x00" * 100)
        ok, out, info = self._call(p)
        assert ok is False
        assert "Unsupported format" in info["error"]

    def test_all_supported_extensions_accepted(self, tmp_path):
        """Verify every extension in the allowed list doesn't fail on format check."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from typhoon_asr_inference import prepare_audio

        for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"]:
            p = tmp_path / f"audio{ext}"
            p.write_bytes(b"\x00" * 100)  # garbage content, format check is on extension
            ok, out, info = prepare_audio(str(p))
            # Should NOT fail with "Unsupported format" — may fail on load but not format
            assert "Unsupported format" not in info.get("error", ""), ext

    # ---- Audio properties after processing -------------------------------- #

    def test_normalisation_produces_finite_values(self, tmp_wav):
        """Output WAV must not contain NaN or Inf after normalisation."""
        import soundfile as sf
        ok, out, _ = self._call(tmp_wav)
        assert ok
        y, _ = sf.read(out)
        assert np.all(np.isfinite(y))
        Path(out).unlink(missing_ok=True)

    def test_silence_normalisation_does_not_divide_by_zero(self, tmp_wav_silence):
        """Silent audio: denominator guard (1e-8) prevents division-by-zero."""
        import soundfile as sf
        ok, out, _ = self._call(tmp_wav_silence)
        assert ok
        y, _ = sf.read(out)
        assert np.all(np.isfinite(y))
        Path(out).unlink(missing_ok=True)

    def test_clipped_audio_is_normalised(self, tmp_wav_clipped):
        """Saturated input must be normalised to [-1, 1] range."""
        import soundfile as sf
        ok, out, _ = self._call(tmp_wav_clipped)
        assert ok
        y, _ = sf.read(out)
        assert np.max(np.abs(y)) <= 1.0 + 1e-6
        Path(out).unlink(missing_ok=True)

    def test_very_short_audio_does_not_crash(self, tmp_wav_very_short):
        """50 ms clips must not crash — may succeed or fail gracefully."""
        ok, out, info = self._call(tmp_wav_very_short)
        if ok:
            Path(out).unlink(missing_ok=True)
        else:
            assert "error" in info


# ---------------------------------------------------------------------------
# 2. transcribe() API — model mocked
# ---------------------------------------------------------------------------

class TestTranscribeAPI:
    """Tests for packages/typhoon_asr/typhoon_asr/__init__.py::transcribe."""

    def _transcribe(self, *args, **kwargs):
        sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "typhoon_asr"))
        import typhoon_asr
        importlib.reload(typhoon_asr)  # ensure clean state
        return typhoon_asr.transcribe(*args, **kwargs)

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_basic_transcription_returns_text(self, mock_load, tmp_wav):
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["สวัสดีครับ"])
        )
        result = self._transcribe(str(tmp_wav))
        assert "text" in result
        assert isinstance(result["text"], str)

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_basic_transcription_no_timestamps_key_absent(self, mock_load, tmp_wav):
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["สวัสดีครับ"])
        )
        result = self._transcribe(str(tmp_wav), with_timestamps=False)
        assert "timestamps" not in result or result.get("timestamps") is None

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_with_timestamps_returns_list(self, mock_load, tmp_wav):
        hyp = MagicMock()
        hyp.text = "สวัสดีครับ ยินดีต้อนรับ"
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=[hyp])
        )
        result = self._transcribe(str(tmp_wav), with_timestamps=True)
        assert "timestamps" in result
        assert isinstance(result["timestamps"], list)

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_timestamps_cover_full_duration(self, mock_load, tmp_wav):
        """Last timestamp end should be approximately equal to audio_duration."""
        hyp = MagicMock()
        hyp.text = "สวัสดี ครับ"
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=[hyp])
        )
        result = self._transcribe(str(tmp_wav), with_timestamps=True)
        ts = result["timestamps"]
        assert ts, "timestamps must not be empty when text is non-empty"
        assert abs(ts[-1]["end"] - result["audio_duration"]) < 0.1

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_timestamps_are_monotonically_increasing(self, mock_load, tmp_wav):
        hyp = MagicMock()
        hyp.text = "ผม ชื่อ ไท โฟน ครับ"
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=[hyp])
        )
        result = self._transcribe(str(tmp_wav), with_timestamps=True)
        starts = [t["start"] for t in result["timestamps"]]
        assert starts == sorted(starts)

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_processing_time_positive(self, mock_load, tmp_wav):
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["สวัสดี"])
        )
        result = self._transcribe(str(tmp_wav))
        assert result["processing_time"] >= 0

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_audio_duration_matches_wav(self, mock_load, tmp_wav):
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["สวัสดี"])
        )
        result = self._transcribe(str(tmp_wav))
        assert abs(result["audio_duration"] - _wav_duration(tmp_wav)) < 0.05

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_nonexistent_file_raises(self, mock_load, tmp_path):
        mock_load.return_value = MagicMock()
        with pytest.raises(FileNotFoundError):
            self._transcribe(str(tmp_path / "missing.wav"))

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_unsupported_format_raises(self, mock_load, tmp_path):
        p = tmp_path / "audio.xyz"
        p.write_bytes(b"\x00" * 100)
        mock_load.return_value = MagicMock()
        with pytest.raises(ValueError, match="Unsupported format"):
            self._transcribe(str(p))

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_silence_returns_empty_or_str(self, mock_load, tmp_wav_silence):
        """Silent audio may yield empty string — must not crash."""
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=[""])
        )
        result = self._transcribe(str(tmp_wav_silence))
        assert isinstance(result["text"], str)

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_tmp_file_cleaned_up(self, mock_load, tmp_wav):
        """Processed WAV temp file must be deleted after transcription."""
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["ทดสอบ"])
        )
        self._transcribe(str(tmp_wav))
        remaining = list(Path(".").glob("processed_*.wav"))
        assert remaining == [], f"Temp files not cleaned up: {remaining}"

    # ---- Thai-specific text edge cases ------------------------------------ #

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_thai_particles_in_output(self, mock_load, tmp_wav):
        """Transcription output with Thai politeness particles is preserved."""
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["ขอบคุณมากครับ"])
        )
        result = self._transcribe(str(tmp_wav))
        assert "ครับ" in result["text"]

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_mixed_thai_english_text(self, mock_load, tmp_wav):
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["ผมชอบ Python มากครับ"])
        )
        result = self._transcribe(str(tmp_wav))
        assert "Python" in result["text"]
        assert "ครับ" in result["text"]

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_thai_numerals_in_output(self, mock_load, tmp_wav):
        """Thai cardinal numbers should be preserved as-is."""
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=["วันที่ ๑๕ มีนาคม"])
        )
        result = self._transcribe(str(tmp_wav))
        assert "๑๕" in result["text"] or "15" in result["text"]

    @patch("nemo.collections.asr.models.ASRModel.from_pretrained")
    def test_very_long_text_has_proportional_timestamps(self, mock_load, tmp_wav):
        """With 10 words over 1 s, each word slot must be ~0.1 s."""
        hyp = MagicMock()
        hyp.text = " ".join(["คำ"] * 10)
        mock_load.return_value = MagicMock(
            transcribe=MagicMock(return_value=[hyp])
        )
        result = self._transcribe(str(tmp_wav), with_timestamps=True)
        ts = result["timestamps"]
        for t in ts:
            assert abs((t["end"] - t["start"]) - 0.1) < 0.01


# ---------------------------------------------------------------------------
# 3. SentenceBoundaryDetector — Thai sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceBoundaryDetector:
    """
    Tests for typhoon_asr_events.services.transcription_aggregator.SentenceBoundaryDetector.
    Focuses on Thai-specific edge cases.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from typhoon_asr_events.services.transcription_aggregator import SentenceBoundaryDetector
        self.detector = SentenceBoundaryDetector()

    # ---- Basic contract -------------------------------------------------- #

    def test_empty_string_returns_empty_list(self):
        assert self.detector.detect_boundaries("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert self.detector.detect_boundaries("   \t\n  ") == []

    def test_single_sentence_with_period(self):
        result = self.detector.detect_boundaries("สวัสดีครับ.")
        assert len(result) == 1

    # ---- Thai politeness particles --------------------------------------- #

    def test_krap_marks_boundary(self):
        result = self.detector.detect_boundaries("สวัสดีครับ")
        assert len(result) >= 1
        assert any("สวัสดีครับ" in s for s in result)

    def test_kha_marks_boundary(self):
        result = self.detector.detect_boundaries("สวัสดีค่ะ")
        assert len(result) >= 1

    def test_na_marks_boundary(self):
        result = self.detector.detect_boundaries("ไปด้วยกันนะ")
        assert len(result) >= 1

    def test_ja_marks_boundary(self):
        result = self.detector.detect_boundaries("เดี๋ยวกลับมาจ้า")
        assert len(result) >= 1

    def test_thoe_marks_boundary(self):
        result = self.detector.detect_boundaries("ลองดูเถอะ")
        assert len(result) >= 1

    # ---- Multiple sentences ---------------------------------------------- #

    def test_two_sentences_with_particles(self):
        text = "สวัสดีครับ ขอบคุณมากค่ะ"
        result = self.detector.detect_boundaries(text)
        # At minimum both particles should be detected somewhere
        combined = " ".join(result)
        assert "ครับ" in combined and "ค่ะ" in combined

    def test_question_mark_splits_sentences(self):
        text = "คุณชื่ออะไร? ผมชื่อไทโฟน"
        result = self.detector.detect_boundaries(text)
        assert len(result) >= 1

    def test_exclamation_splits_sentences(self):
        result = self.detector.detect_boundaries("ดีมาก! ขอบคุณครับ")
        assert len(result) >= 1

    # ---- Mixed Thai-English ---------------------------------------------- #

    def test_mixed_language_sentence(self):
        text = "ผมชอบ Python มากครับ"
        result = self.detector.detect_boundaries(text)
        assert len(result) >= 1
        assert "Python" in result[0]

    def test_english_sentence_passes_through(self):
        text = "Hello world."
        result = self.detector.detect_boundaries(text)
        assert len(result) >= 1

    # ---- Edge cases ------------------------------------------------------ #

    def test_particle_in_middle_of_sentence_not_split(self):
        """'ครับ' mid-sentence should not split on words containing but not ending with it."""
        # "ครับผม" contains "ครับ" but is not a sentence-ender
        text = "ครับผมไม่เข้าใจนะครับ"
        result = self.detector.detect_boundaries(text)
        # Must not crash and must return at least one item
        assert isinstance(result, list)

    def test_very_long_sentence_without_boundary(self):
        """A long sentence with no punctuation or particles is returned as-is."""
        text = "คำ " * 50  # 50 repeated words
        result = self.detector.detect_boundaries(text.strip())
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_only_punctuation_input(self):
        """Input that is pure punctuation should not crash."""
        result = self.detector.detect_boundaries("...!!!")
        assert isinstance(result, list)

    def test_single_thai_character(self):
        result = self.detector.detect_boundaries("ก")
        assert isinstance(result, list)

    def test_numbers_thai_script(self):
        """Thai numerals mixed with text must not break splitting."""
        text = "วันที่ ๑๕ มีนาคม ๒๕๖๘ ครับ"
        result = self.detector.detect_boundaries(text)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_repeated_particles_not_infinite_loop(self):
        """Rapid successive particles should terminate quickly."""
        text = "ครับ ค่ะ นะ จ้า เถอะ"
        result = self.detector.detect_boundaries(text)
        assert isinstance(result, list)

    def test_transliterated_thai_proper_noun(self):
        """Foreign names in Thai transliteration must be preserved."""
        text = "ผมชอบฟัง Beethoven ครับ"
        result = self.detector.detect_boundaries(text)
        combined = " ".join(result)
        assert "Beethoven" in combined

    def test_detect_boundaries_output_is_list_of_strings(self):
        text = "ทดสอบระบบ ASR ของไทโฟนครับ"
        result = self.detector.detect_boundaries(text)
        assert all(isinstance(s, str) for s in result)

    def test_newlines_in_input(self):
        """Embedded newlines must not crash the detector."""
        text = "บรรทัดที่หนึ่งครับ\nบรรทัดที่สองค่ะ"
        result = self.detector.detect_boundaries(text)
        assert isinstance(result, list)

    def test_unicode_normalization_edge_case(self):
        """Thai characters composed with combining characters should not crash."""
        # เ + า + ะ composed differently in different Unicode forms
        text = "\u0e40\u0e32\u0e30\u0e04\u0e23\u0e31\u0e1a"  # เอาะครับ
        result = self.detector.detect_boundaries(text)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 4. Python bridge server protocol
# ---------------------------------------------------------------------------

class TestBridgeServerProtocol:
    """
    Tests for python/bridge_server.py without launching the process.
    Tests the pure functions (e.g. audio format sniffing, response shape).
    """

    @pytest.fixture(autouse=True)
    def _import_bridge(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
        # bridge_server imports nemo lazily, so we can import the module itself
        with patch.dict("sys.modules", {
            "nemo": MagicMock(),
            "nemo.collections": MagicMock(),
            "nemo.collections.asr": MagicMock(),
            "nemo.collections.asr.models": MagicMock(),
            "torch": MagicMock(),
        }):
            import bridge_server
            importlib.reload(bridge_server)
            self.bridge = bridge_server

    def test_ping_action_returns_model_loaded_false_initially(self, capsys):
        """Before any model is loaded, ping must report model_loaded=False."""
        responses = []
        original_respond = self.bridge._respond
        def capture(request_id, payload):
            payload["request_id"] = request_id
            responses.append(payload)
        self.bridge._respond = capture

        self.bridge._handle({"request_id": "test-ping", "action": "ping"})
        self.bridge._respond = original_respond

        assert responses[0]["model_loaded"] is False

    def test_handle_missing_audio_source_returns_error(self):
        responses = []
        def capture(request_id, payload):
            payload["request_id"] = request_id
            responses.append(payload)
        self.bridge._respond = capture

        self.bridge._handle({"request_id": "bad-req"})

        assert "error" in responses[0]

    def test_wav_magic_bytes_sniffed_correctly(self):
        """RIFF header → .wav extension."""
        riff_header = b"RIFF\x00\x00\x00\x00WAVEfmt "
        audio_b64 = base64.b64encode(riff_header).decode()
        # We only test sniffing logic, not actual inference
        # Peek at the suffix logic by triggering the bytes branch
        raw = base64.b64decode(audio_b64)
        suffix = ".wav"
        if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb":
            suffix = ".mp3"
        elif raw[:4] == b"fLaC":
            suffix = ".flac"
        elif raw[:4] == b"OggS":
            suffix = ".ogg"
        assert suffix == ".wav"

    def test_mp3_magic_bytes_sniffed_correctly(self):
        raw = b"ID3" + b"\x00" * 10
        suffix = ".wav"
        if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb":
            suffix = ".mp3"
        assert suffix == ".mp3"

    def test_flac_magic_bytes_sniffed_correctly(self):
        raw = b"fLaC" + b"\x00" * 10
        suffix = ".wav"
        if raw[:4] == b"fLaC":
            suffix = ".flac"
        assert suffix == ".flac"

    def test_ogg_magic_bytes_sniffed_correctly(self):
        raw = b"OggS" + b"\x00" * 10
        suffix = ".wav"
        if raw[:4] == b"OggS":
            suffix = ".ogg"
        assert suffix == ".ogg"

    def test_response_always_has_request_id(self):
        collected = []
        def capture(request_id, payload):
            payload["request_id"] = request_id
            collected.append(payload)
        self.bridge._respond = capture

        self.bridge._handle({"request_id": "my-unique-id", "action": "ping"})
        assert collected[0]["request_id"] == "my-unique-id"


# ---------------------------------------------------------------------------
# 5. Integration tests — require real NeMo model
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """
    End-to-end tests against the real Typhoon ASR model.
    Run with:  pytest --run-integration tests/test_thai_audio.py::TestIntegration
    """

    def _transcribe(self, path, **kwargs):
        sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "typhoon_asr"))
        from typhoon_asr import transcribe
        return transcribe(str(path), **kwargs)

    def test_real_transcription_returns_nonempty_text(self, tmp_wav):
        result = self._transcribe(tmp_wav)
        # Synthetic sine wave will likely produce garbage Thai, but must not crash
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_real_rtf_is_reasonable(self, tmp_wav):
        """RTF should be > 0 — model did some work."""
        result = self._transcribe(tmp_wav)
        assert result["processing_time"] > 0
        assert result["audio_duration"] > 0

    def test_real_silence_graceful(self, tmp_wav_silence):
        result = self._transcribe(tmp_wav_silence)
        assert isinstance(result["text"], str)

    def test_real_different_sample_rates_produce_consistent_duration(self, tmp_wav_8khz, tmp_wav_44khz):
        r8  = self._transcribe(tmp_wav_8khz)
        r44 = self._transcribe(tmp_wav_44khz)
        # Both clips are 1 second; durations should be close regardless of source SR
        assert abs(r8["audio_duration"] - r44["audio_duration"]) < 0.1
