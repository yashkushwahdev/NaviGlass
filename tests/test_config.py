"""Unit tests for configuration loading and validation."""

import pytest
from naviglass.config import AppConfig, CameraConfig, ConfigurationError, ModelConfig, AlertConfig


def test_default_config_valid():
    cfg = AppConfig()
    cfg.validate()
    assert cfg.camera.index == 0
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.model.inference_size == 320
    assert cfg.alert.cooldown_seconds == 4.0


def test_invalid_camera_config():
    with pytest.raises(ConfigurationError):
        CameraConfig(width=0).validate()
    with pytest.raises(ConfigurationError):
        CameraConfig(fps=-5).validate()
    with pytest.raises(ConfigurationError):
        CameraConfig(index=-1).validate()


def test_invalid_model_config():
    with pytest.raises(ConfigurationError):
        ModelConfig(confidence_threshold=1.5).validate()
    with pytest.raises(ConfigurationError):
        ModelConfig(inference_size=300).validate()


def test_invalid_alert_config():
    with pytest.raises(ConfigurationError):
        AlertConfig(cooldown_seconds=-1.0).validate()


def test_config_save_and_load(tmp_path):
    p = tmp_path / "test_cfg.yaml"
    cfg = AppConfig()
    cfg.camera.index = 2
    cfg.alert.cooldown_seconds = 5.5
    cfg.save(p)

    loaded = AppConfig.load(p)
    assert loaded.camera.index == 2
    assert loaded.alert.cooldown_seconds == 5.5
