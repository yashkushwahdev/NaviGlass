"""Unit tests for device selection (local CPU and optional NVIDIA GPU)."""

import pytest
from unittest.mock import MagicMock, patch
from config.settings import ModelConfig
from detection.detector import ObjectDetector


def test_default_device_is_cpu():
    cfg = ModelConfig(device="cpu")
    det = ObjectDetector(cfg)
    assert det.device == "cpu"


def test_cuda_fallback_to_cpu_when_unavailable():
    cfg = ModelConfig(device="cuda")
    with patch("torch.cuda.is_available", return_value=False):
        det = ObjectDetector(cfg)
        assert det.device == "cpu"


def test_cuda_selected_when_available():
    cfg = ModelConfig(device="cuda")
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="GeForce RTX 4060 Laptop"):
            det = ObjectDetector(cfg)
            assert det.device == "cuda:0"


def test_auto_device_selection():
    # When CUDA is available, auto picks cuda:0
    cfg_auto = ModelConfig(device="auto")
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="RTX GPU"):
            det = ObjectDetector(cfg_auto)
            assert det.device == "cuda:0"

    # When CUDA is unavailable, auto falls back to cpu
    with patch("torch.cuda.is_available", return_value=False):
        det2 = ObjectDetector(cfg_auto)
        assert det2.device == "cpu"
