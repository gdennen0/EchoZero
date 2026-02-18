"""
Tests for canonical inference preprocessing metadata.
"""
from src.application.blocks.training.checkpointing import build_inference_preprocessing


def test_build_inference_preprocessing_includes_audio_input_standard():
    cfg = {
        "sample_rate": 44100,
        "max_length": 44100,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "fmax": 16000,
    }
    prep = build_inference_preprocessing(cfg, normalization=None)
    assert "audio_input_standard" in prep
    assert prep["audio_input_standard"]["encoding"] == "wav_pcm16"
    assert prep["audio_input_standard"]["channels"] == 1
    assert prep["audio_input_standard"]["sample_rate"] == 44100
