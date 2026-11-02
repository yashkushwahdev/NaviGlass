"""Unit tests for simplified GUI and SettingsDialog."""

import os
import tkinter as tk
import pytest
from config.settings import AppConfig
from app.gui import NaviGlassGUI, SettingsDialog


@pytest.fixture
def tk_app():
    try:
        root = tk.Tk()
    except Exception as e:
        pytest.skip(f"Tkinter display not available in current test environment: {e}")

    root.withdraw()
    cfg = AppConfig()
    cfg.audio.enabled = False

    app = NaviGlassGUI(root, cfg)
    root.update()

    yield app

    try:
        app.on_close()
    except Exception:
        pass


def test_gui_initialization_and_close(tk_app):
    assert tk_app.canvas is not None
    assert tk_app.btn_start is not None
    assert tk_app.btn_stop is not None
    assert tk_app.tree_detections is not None
    assert tk_app.var_cam_status.get() == "Camera: Stopped"
    assert "Camera ready" in tk_app.var_status_msg.get()


def test_settings_dialog_save_and_apply(tk_app):
    # Open Settings Dialog
    dlg = SettingsDialog(tk_app.root, tk_app)
    dlg.withdraw()
    tk_app.root.update()

    # Modify values in dialog
    dlg.var_conf.set(0.65)
    dlg.var_cool.set(5.5)
    dlg.var_device.set("cpu")

    # Save
    dlg._save_and_close()
    tk_app.root.update()

    # Verify changes reflected on app
    assert tk_app.var_confidence.get() == 0.65
    assert tk_app.filter_mgr.min_confidence == 0.65
    assert tk_app.var_cooldown.get() == 5.5
    assert tk_app.filter_mgr.cooldown_seconds == 5.5
    assert "Settings saved" in tk_app.var_status_msg.get()
