"""Simplified, beginner-friendly Desktop GUI for NaviGlass.

Architecture (non-blocking, thread-safe pipeline):
- Camera grabber thread (in Webcam): continuously captures latest frame into single-slot buffer.
- Detection worker thread: runs YOLO on latest frame, never waits for depth, reads configuration from thread-safe variables (never touches Tkinter objects).
- Depth worker thread: runs depth inference on latest frame independently without blocking GUI or camera.
- Tkinter main thread: renders frames and detection table via root.after(30).
- TTS thread: independent speech queue (in SpeechEngine).

Thread Safety Rules:
- Worker threads NEVER call `.get()` or `.set()` on Tkinter variables (tk.Variable, tk.DoubleVar, etc.)
  to prevent Tcl "RuntimeError: main thread is not in main loop".
- All GUI variables are accessed only from the main thread.
- Thread-safe getters/setters with Lock protect shared configuration values.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

from audio.speech import SpeechEngine
from camera.webcam import CameraError, Webcam
from config.settings import AppConfig
from depth.depth_estimator import DepthEstimator, DepthEstimatorError, MetricDepthMap
from detection.detector import DetectionResult, DetectorError, ObjectDetector
from detection.filtering import DetectionFilter
from detection.nearest_object import NearestObjectManager, WarningDistanceLevel
from utils.logging_config import get_logger

logger = get_logger("gui")


def _get_ram_mb() -> float:
    """Returns current process RSS in MB, or -1 if psutil unavailable."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


class SettingsDialog(tk.Toplevel):
    """Simplified modal settings dialog for confidence, cooldown, and hardware preferences."""

    def __init__(self, parent: tk.Tk, gui: NaviGlassGUI) -> None:
        super().__init__(parent)
        self.gui = gui
        self.title("NaviGlass Settings")
        self.geometry("440x480")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Light theme colors
        self.c_bg = "#FFFFFF"
        self.c_text = "#0F172A"
        self.c_muted = "#64748B"
        self.c_accent = "#2563EB"
        self.c_border = "#E2E8F0"

        self.configure(bg=self.c_bg)

        # Temporary variable copies (main thread only)
        self.var_conf = tk.DoubleVar(value=self.gui.get_confidence_threshold())
        self.var_cool = tk.DoubleVar(value=self.gui.get_cooldown_seconds())
        self.var_interval = tk.DoubleVar(value=self.gui.get_interval_seconds())
        self.var_depth = tk.BooleanVar(value=self.gui.is_depth_enabled())
        self.var_device = tk.StringVar(value=self.gui.config.model.device)

        self._build_ui()
        self.bind("<Escape>", lambda e: self.destroy())

    def _build_ui(self) -> None:
        pad_x = 24

        # Title
        tk.Label(
            self,
            text="Application Settings",
            font=("Segoe UI", 14, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        ).pack(anchor=tk.W, padx=pad_x, pady=(20, 4))

        tk.Label(
            self,
            text="Configure object detection and voice alert preferences.",
            font=("Segoe UI", 9),
            fg=self.c_muted,
            bg=self.c_bg,
        ).pack(anchor=tk.W, padx=pad_x, pady=(0, 16))

        # 1. Confidence Slider
        conf_frame = tk.Frame(self, bg=self.c_bg)
        conf_frame.pack(fill=tk.X, padx=pad_x, pady=(0, 14))

        conf_hdr = tk.Frame(conf_frame, bg=self.c_bg)
        conf_hdr.pack(fill=tk.X)
        tk.Label(
            conf_hdr,
            text="Detection Sensitivity",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        ).pack(side=tk.LEFT)

        self.lbl_conf_val = tk.Label(
            conf_hdr,
            text=f"{int(self.var_conf.get() * 100)}%",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_accent,
            bg=self.c_bg,
        )
        self.lbl_conf_val.pack(side=tk.RIGHT)

        tk.Label(
            conf_frame,
            text="Lower values detect more objects; higher values reduce false alerts.",
            font=("Segoe UI", 8),
            fg=self.c_muted,
            bg=self.c_bg,
        ).pack(anchor=tk.W, pady=(2, 4))

        scale_conf = ttk.Scale(
            conf_frame,
            from_=0.15,
            to=0.90,
            variable=self.var_conf,
            command=lambda val: self.lbl_conf_val.configure(text=f"{int(float(val) * 100)}%"),
        )
        scale_conf.pack(fill=tk.X)

        # 2. Cooldown Slider
        cool_frame = tk.Frame(self, bg=self.c_bg)
        cool_frame.pack(fill=tk.X, padx=pad_x, pady=(0, 14))

        cool_hdr = tk.Frame(cool_frame, bg=self.c_bg)
        cool_hdr.pack(fill=tk.X)
        tk.Label(
            cool_hdr,
            text="Alert Frequency (Cooldown)",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        ).pack(side=tk.LEFT)

        self.lbl_cool_val = tk.Label(
            cool_hdr,
            text=f"{self.var_cool.get():.1f}s",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_accent,
            bg=self.c_bg,
        )
        self.lbl_cool_val.pack(side=tk.RIGHT)

        tk.Label(
            cool_frame,
            text="Minimum wait time before announcing the same object class again.",
            font=("Segoe UI", 8),
            fg=self.c_muted,
            bg=self.c_bg,
        ).pack(anchor=tk.W, pady=(2, 4))

        scale_cool = ttk.Scale(
            cool_frame,
            from_=1.0,
            to=10.0,
            variable=self.var_cool,
            command=lambda val: self.lbl_cool_val.configure(text=f"{float(val):.1f}s"),
        )
        scale_cool.pack(fill=tk.X)

        # 3. Repeat Interval Slider
        interval_frame = tk.Frame(self, bg=self.c_bg)
        interval_frame.pack(fill=tk.X, padx=pad_x, pady=(0, 14))

        interval_hdr = tk.Frame(interval_frame, bg=self.c_bg)
        interval_hdr.pack(fill=tk.X)
        tk.Label(
            interval_hdr,
            text="Repeat Interval (Visible Objects)",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        ).pack(side=tk.LEFT)

        self.lbl_interval_val = tk.Label(
            interval_hdr,
            text=f"{self.var_interval.get():.1f}s",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_accent,
            bg=self.c_bg,
        )
        self.lbl_interval_val.pack(side=tk.RIGHT)

        tk.Label(
            interval_frame,
            text="Time between spoken reminders for continuously visible objects.",
            font=("Segoe UI", 8),
            fg=self.c_muted,
            bg=self.c_bg,
        ).pack(anchor=tk.W, pady=(2, 4))

        scale_interval = ttk.Scale(
            interval_frame,
            from_=1.0,
            to=10.0,
            variable=self.var_interval,
            command=lambda val: self.lbl_interval_val.configure(text=f"{float(val):.1f}s"),
        )
        scale_interval.pack(fill=tk.X)

        # 4. Hardware Preference
        dev_frame = tk.Frame(self, bg=self.c_bg)
        dev_frame.pack(fill=tk.X, padx=pad_x, pady=(0, 20))

        tk.Label(
            dev_frame,
            text="Hardware Processing",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        ).pack(anchor=tk.W)

        tk.Label(
            dev_frame,
            text="Choose between laptop CPU and optional NVIDIA GPU acceleration.",
            font=("Segoe UI", 8),
            fg=self.c_muted,
            bg=self.c_bg,
        ).pack(anchor=tk.W, pady=(2, 6))

        dev_row = tk.Frame(dev_frame, bg=self.c_bg)
        dev_row.pack(anchor=tk.W)

        for val, label in [("cpu", "Laptop CPU"), ("auto", "Auto (Detect GPU)"), ("cuda", "NVIDIA GPU")]:
            rb = tk.Radiobutton(
                dev_row,
                text=label,
                value=val,
                variable=self.var_device,
                font=("Segoe UI", 9),
                fg=self.c_text,
                bg=self.c_bg,
                activebackground=self.c_bg,
                selectcolor=self.c_bg,
            )
            rb.pack(side=tk.LEFT, padx=(0, 12))

        # Bottom Action Buttons
        btn_box = tk.Frame(self, bg=self.c_bg)
        btn_box.pack(fill=tk.X, side=tk.BOTTOM, padx=pad_x, pady=16)

        btn_save = tk.Button(
            btn_box,
            text="Save Settings",
            font=("Segoe UI", 9, "bold"),
            bg=self.c_accent,
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._save_and_close,
        )
        btn_save.pack(side=tk.RIGHT, padx=(8, 0))

        btn_cancel = tk.Button(
            btn_box,
            text="Cancel",
            font=("Segoe UI", 9),
            bg="#F1F5F9",
            fg="#334155",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.destroy,
        )
        btn_cancel.pack(side=tk.RIGHT)

    def _save_and_close(self) -> None:
        new_conf = self.var_conf.get()
        new_cool = self.var_cool.get()
        new_interval = self.var_interval.get()
        new_depth = self.var_depth.get()

        # Update GUI vars (main thread)
        self.gui.var_confidence.set(new_conf)
        self.gui.var_cooldown.set(new_cool)
        self.gui.var_interval.set(new_interval)
        self.gui.var_depth_enabled.set(new_depth)

        # Update thread-safe variables for background workers
        self.gui.update_settings(
            confidence=new_conf,
            cooldown=new_cool,
            interval=new_interval,
            depth_enabled=new_depth,
        )

        dev = self.var_device.get()
        self.gui.config.model.device = dev
        depth_str = "Enabled" if new_depth else "Disabled"
        self.gui.set_status(f"Settings saved (Depth: {depth_str}, Sensitivity: {int(new_conf*100)}%, Repeat: {new_interval:.1f}s)")
        self.destroy()


class NaviGlassGUI:
    """Simple, clean, beginner-friendly desktop interface for NaviGlass.

    Non-blocking, thread-safe pipeline architecture:
    - Camera grabber thread (inside Webcam) drains the OS camera buffer continuously.
    - Detection worker thread runs YOLO + enrichment, reads thread-safe variables.
    - Depth worker thread runs depth inference independently, stores result.
    - GUI main thread polls a bounded frame queue every 30 ms.
    """

    def __init__(self, root: tk.Tk, config: AppConfig) -> None:
        self.root = root
        self.config = config

        self.root.title("NaviGlass — AI Vision Assistant")
        self.root.geometry("980x640")
        self.root.minsize(820, 540)

        # Clean light theme palette
        self.c_bg = "#F8FAFC"          # Slate 50 background
        self.c_card = "#FFFFFF"        # Clean white card surface
        self.c_border = "#E2E8F0"      # Subtle border
        self.c_text = "#0F172A"        # Deep slate primary text
        self.c_muted = "#64748B"       # Slate 500 secondary text
        self.c_accent = "#2563EB"      # Tailored Blue 600
        self.c_start = "#16A34A"       # Clean Emerald Green
        self.c_stop = "#DC2626"        # Clean Crimson Red

        self.root.configure(bg=self.c_bg)

        # Subsystems
        self.speech_engine = SpeechEngine(self.config.audio)
        self.detector = ObjectDetector(self.config.model)
        self.depth_estimator = DepthEstimator(self.config.depth)
        self.filter_mgr = DetectionFilter(self.config.alert, min_confidence=self.config.model.confidence_threshold)
        self.nearest_mgr = NearestObjectManager(self.config.alert, self.config.depth)
        self.webcam = Webcam(self.config.camera)

        # Threading and State
        self._worker_thread: Optional[threading.Thread] = None
        self._depth_thread: Optional[threading.Thread] = None
        self._stop_worker_event = threading.Event()
        self._frame_queue: queue.Queue[tuple] = queue.Queue(maxsize=2)
        self._is_camera_active = False

        # Thread-safe configuration state for worker threads (NEVER touch tk.Variable in background)
        self._settings_lock = threading.Lock()
        self._conf_val: float = float(self.config.model.confidence_threshold)
        self._cool_val: float = float(self.config.alert.cooldown_seconds)
        self._interval_val: float = float(self.config.alert.announcement_interval_seconds)
        self._depth_enabled_val: bool = bool(self.config.depth.enabled)
        self._voice_enabled_val: bool = bool(self.config.audio.enabled)

        # Shared depth state (written by depth thread, read by worker thread)
        self._depth_lock = threading.Lock()
        self._latest_depth_map: Optional[MetricDepthMap] = None
        self._depth_ms: float = 0.0
        self._depth_busy = threading.Lock()  # Prevents overlapping depth calls

        # Frame timestamps for stall detection
        self._latest_displayed_timestamp: float = 0.0
        self._latest_displayed_fid: int = 0
        self._last_ui_callback_duration_ms: float = 0.0

        # GUI State Variables (accessed exclusively on Tkinter main thread)
        self.var_confidence = tk.DoubleVar(value=self._conf_val)
        self.var_cooldown = tk.DoubleVar(value=self._cool_val)
        self.var_interval = tk.DoubleVar(value=self._interval_val)
        self.var_voice_enabled = tk.BooleanVar(value=self._voice_enabled_val)
        self.var_depth_enabled = tk.BooleanVar(value=self._depth_enabled_val)
        self.var_muted = tk.BooleanVar(value=False)

        # Status text variables
        self.var_cam_status = tk.StringVar(value="Camera: Stopped")
        self.var_pill_status = tk.StringVar(value="● Ready")
        self.var_status_msg = tk.StringVar(value="Camera ready. Click 'Start Camera' to begin.")

        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._bind_shortcuts()
        self._init_subsystems()

        # Start periodic GUI updates
        self.root.after(30, self._ui_update_loop)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------------------------
    # Thread-Safe Settings Accessors (Called by worker threads)
    # -------------------------------------------------------------------------
    def get_confidence_threshold(self) -> float:
        with self._settings_lock:
            return self._conf_val

    def get_cooldown_seconds(self) -> float:
        with self._settings_lock:
            return self._cool_val

    def get_interval_seconds(self) -> float:
        with self._settings_lock:
            return self._interval_val

    def is_depth_enabled(self) -> bool:
        with self._settings_lock:
            return self._depth_enabled_val

    def is_voice_enabled(self) -> bool:
        with self._settings_lock:
            return self._voice_enabled_val

    def update_settings(
        self,
        confidence: Optional[float] = None,
        cooldown: Optional[float] = None,
        interval: Optional[float] = None,
        depth_enabled: Optional[bool] = None,
        voice_enabled: Optional[bool] = None,
    ) -> None:
        """Thread-safe update of settings consumed by background threads."""
        with self._settings_lock:
            if confidence is not None:
                self._conf_val = confidence
                self.filter_mgr.min_confidence = confidence
            if cooldown is not None:
                self._cool_val = cooldown
                self.filter_mgr.cooldown_seconds = cooldown
            if interval is not None:
                self._interval_val = interval
                self.filter_mgr.announcement_interval_seconds = interval
            if depth_enabled is not None:
                self._depth_enabled_val = depth_enabled
                self.config.depth.enabled = depth_enabled
            if voice_enabled is not None:
                self._voice_enabled_val = voice_enabled
                self.config.audio.enabled = voice_enabled

    def _build_ui(self) -> None:
        """Constructs a clean, minimal, accessible light-themed interface."""
        # 1. Header Bar
        header = tk.Frame(self.root, bg=self.c_card, padx=20, pady=12, highlightthickness=1, highlightbackground=self.c_border)
        header.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header, bg=self.c_card)
        title_box.pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="NaviGlass",
            font=("Segoe UI", 16, "bold"),
            fg=self.c_text,
            bg=self.c_card,
        ).pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="  AI Vision Assistant",
            font=("Segoe UI", 11),
            fg=self.c_muted,
            bg=self.c_card,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Right Header: Status Pill and Settings Button
        header_right = tk.Frame(header, bg=self.c_card)
        header_right.pack(side=tk.RIGHT)

        self.lbl_status_pill = tk.Label(
            header_right,
            textvariable=self.var_pill_status,
            font=("Segoe UI", 9, "bold"),
            fg="#1D4ED8",
            bg="#EFF6FF",
            padx=12,
            pady=4,
        )
        self.lbl_status_pill.pack(side=tk.LEFT, padx=(0, 10))

        btn_settings = tk.Button(
            header_right,
            text="⚙ Settings",
            font=("Segoe UI", 9),
            bg="#F1F5F9",
            fg="#334155",
            activebackground="#E2E8F0",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self.open_settings,
        )
        btn_settings.pack(side=tk.LEFT)

        # 2. Main Content Split
        main_body = tk.Frame(self.root, bg=self.c_bg, padx=16, pady=14)
        main_body.pack(fill=tk.BOTH, expand=True)

        # Left Panel: Webcam Preview
        video_card = tk.Frame(
            main_body,
            bg=self.c_card,
            padx=8,
            pady=8,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        self.canvas = tk.Canvas(video_card, bg="#0F172A", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._render_placeholder_video("Camera is off.\n\nPress Start Camera to begin.")

        # Right Panel: Controls & Telemetry
        right_panel = tk.Frame(main_body, bg=self.c_bg, width=320)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.pack_propagate(False)

        # Card A: Action Controls
        ctrl_card = tk.Frame(
            right_panel,
            bg=self.c_card,
            padx=14,
            pady=14,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        ctrl_card.pack(fill=tk.X, pady=(0, 12))

        self.btn_start = tk.Button(
            ctrl_card,
            text="▶  Start Camera",
            font=("Segoe UI", 11, "bold"),
            bg=self.c_start,
            fg="#FFFFFF",
            activebackground="#15803D",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=14,
            pady=10,
            cursor="hand2",
            command=self.start_camera,
        )
        self.btn_start.pack(fill=tk.X, pady=(0, 6))

        self.btn_stop = tk.Button(
            ctrl_card,
            text="⏹  Stop Camera",
            font=("Segoe UI", 10, "bold"),
            bg="#F1F5F9",
            fg="#94A3B8",
            activebackground="#E2E8F0",
            relief=tk.FLAT,
            padx=12,
            pady=7,
            state=tk.DISABLED,
            cursor="hand2",
            command=self.stop_camera,
        )
        self.btn_stop.pack(fill=tk.X, pady=(0, 10))

        audio_row = tk.Frame(ctrl_card, bg=self.c_card)
        audio_row.pack(fill=tk.X)

        chk_voice = tk.Checkbutton(
            audio_row,
            text="Voice Alerts",
            variable=self.var_voice_enabled,
            command=self._on_toggle_voice,
            font=("Segoe UI", 9, "bold"),
            fg=self.c_text,
            bg=self.c_card,
            activebackground=self.c_card,
            activeforeground=self.c_text,
            selectcolor="#FFFFFF",
        )
        chk_voice.pack(side=tk.LEFT)

        self.btn_mute = tk.Button(
            audio_row,
            text="Mute",
            font=("Segoe UI", 9),
            bg="#F1F5F9",
            fg="#475569",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._on_toggle_mute,
        )
        self.btn_mute.pack(side=tk.RIGHT)

        # Card B: Detected Objects Panel
        det_card = tk.Frame(
            right_panel,
            bg=self.c_card,
            padx=14,
            pady=12,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        det_card.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        tk.Label(
            det_card,
            text="Detected Objects",
            font=("Segoe UI", 10, "bold"),
            fg=self.c_text,
            bg=self.c_card,
        ).pack(anchor=tk.W, pady=(0, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Clean.Treeview",
            background="#FFFFFF",
            foreground=self.c_text,
            fieldbackground="#FFFFFF",
            font=("Segoe UI", 9),
            rowheight=24,
            borderwidth=0,
        )
        style.configure(
            "Clean.Treeview.Heading",
            font=("Segoe UI", 9, "bold"),
            background="#F8FAFC",
            foreground=self.c_muted,
            relief=tk.FLAT,
        )
        style.map("Clean.Treeview", background=[("selected", "#EFF6FF")], foreground=[("selected", self.c_accent)])

        self.tree_detections = ttk.Treeview(
            det_card,
            columns=("class", "position", "distance", "closest", "confidence"),
            show="headings",
            style="Clean.Treeview",
            selectmode="none",
        )
        self.tree_detections.heading("class", text="Object Name")
        self.tree_detections.heading("position", text="Position")
        self.tree_detections.heading("distance", text="Est. Distance")
        self.tree_detections.heading("closest", text="Closest")
        self.tree_detections.heading("confidence", text="Confidence")
        self.tree_detections.column("class", width=110, anchor=tk.W)
        self.tree_detections.column("position", width=80, anchor=tk.CENTER)
        self.tree_detections.column("distance", width=105, anchor=tk.CENTER)
        self.tree_detections.column("closest", width=65, anchor=tk.CENTER)
        self.tree_detections.column("confidence", width=70, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(det_card, orient=tk.VERTICAL, command=self.tree_detections.yview)
        self.tree_detections.configure(yscrollcommand=tree_scroll.set)
        self.tree_detections.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._show_empty_detection_state()

        # Card C: Simple Status Area
        status_card = tk.Frame(
            right_panel,
            bg=self.c_card,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.c_border,
        )
        status_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            status_card,
            text="Status",
            font=("Segoe UI", 8, "bold"),
            fg=self.c_muted,
            bg=self.c_card,
        ).pack(anchor=tk.W)

        self.lbl_status_msg = tk.Label(
            status_card,
            textvariable=self.var_status_msg,
            font=("Segoe UI", 9),
            fg=self.c_text,
            bg=self.c_card,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=280,
        )
        self.lbl_status_msg.pack(fill=tk.X, pady=(2, 0))

        # Bottom Exit Button
        btn_exit = tk.Button(
            right_panel,
            text="Exit NaviGlass",
            font=("Segoe UI", 9),
            bg="#F1F5F9",
            fg="#475569",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.on_close,
        )
        btn_exit.pack(fill=tk.X)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<space>", lambda e: self._toggle_camera_shortcut())
        self.root.bind("<m>", lambda e: self._on_toggle_mute())
        self.root.bind("<M>", lambda e: self._on_toggle_mute())
        self.root.bind("<v>", lambda e: self._toggle_voice_shortcut())
        self.root.bind("<V>", lambda e: self._toggle_voice_shortcut())
        self.root.bind("<Escape>", lambda e: self.on_close())

    def _init_subsystems(self) -> None:
        """Loads detector and TTS asynchronously to keep UI fast and responsive."""
        def bg_loader() -> None:
            try:
                self.speech_engine.start()
            except Exception as e:
                logger.error("Audio init error: %s", e)

            try:
                self.detector.load()
                dev = "GPU" if "cuda" in self.detector.device else "CPU"
                depth_tag = ""
                if self.config.depth.enabled:
                    try:
                        self.depth_estimator.load()
                        depth_tag = " + Depth VITS"
                    except DepthEstimatorError as de:
                        logger.warning("Depth estimator load failed: %s", de)
                        depth_tag = " (Depth Offline)"

                self.set_status(f"Camera ready. AI active on {dev}{depth_tag}.")
                self.speech_engine.speak_async("NaviGlass vision assistant ready.")

                # If camera is already active and depth is ready, start depth thread immediately
                if self._is_camera_active and self.is_depth_enabled():
                    self._start_depth_thread_if_needed()
            except DetectorError as e:
                self.set_status("AI model failed to load. Check model weights.")
                logger.error("Detector load failed: %s", e)
            except Exception as outer_e:
                logger.error("Unhandled exception in bg_loader: %s", outer_e, exc_info=True)

        t = threading.Thread(target=bg_loader, name="NaviGlassSubsystemLoader", daemon=True)
        t.start()

    def set_status(self, msg: str) -> None:
        """Updates status message in a thread-safe manner."""
        if threading.current_thread() is threading.main_thread():
            self.var_status_msg.set(msg)
        else:
            try:
                self.root.after(0, lambda: self.var_status_msg.set(msg))
            except Exception:
                pass

    def open_settings(self) -> None:
        """Opens the simplified settings dialog."""
        SettingsDialog(self.root, self)

    def _start_depth_thread_if_needed(self) -> None:
        """Thread-safe launcher for background depth worker."""
        with self._depth_lock:
            if (
                self._depth_thread is None or not self._depth_thread.is_alive()
            ) and not self._stop_worker_event.is_set():
                self._depth_thread = threading.Thread(
                    target=self._depth_worker_loop,
                    name="NaviGlassDepthWorker",
                    daemon=True,
                )
                self._depth_thread.start()
                logger.info("Depth worker thread started.")

    def start_camera(self) -> None:
        """Starts webcam feed, detection worker, and depth worker threads."""
        if self._is_camera_active:
            return

        try:
            self.webcam.start()
        except CameraError as e:
            self.var_cam_status.set("Camera: Error")
            self.var_pill_status.set("● Unavailable")
            self.lbl_status_pill.configure(bg="#FEE2E2", fg="#B91C1C")
            self.set_status("Camera unavailable. Check permissions or close other camera apps.")
            self._render_placeholder_video("Camera Unavailable\n\nPlease check your webcam connection or permissions.")
            logger.error("Camera startup error: %s", e)
            return

        self._is_camera_active = True
        self.var_cam_status.set("Camera: Running")
        self.var_pill_status.set("● Running")
        self.lbl_status_pill.configure(bg="#DCFCE7", fg="#15803D")
        self.set_status("Detecting objects...")

        # Update button states
        self.btn_start.configure(state=tk.DISABLED, bg="#E2E8F0", fg="#94A3B8")
        self.btn_stop.configure(state=tk.NORMAL, bg=self.c_stop, fg="#FFFFFF")

        self.filter_mgr.reset()
        self._stop_worker_event.clear()

        # Reset shared depth state
        with self._depth_lock:
            self._latest_depth_map = None
            self._depth_ms = 0.0

        # Launch detection worker thread
        self._worker_thread = threading.Thread(
            target=self._camera_worker_loop,
            name="NaviGlassDetectionWorker",
            daemon=True,
        )
        self._worker_thread.start()

        # Launch depth worker thread (if depth is enabled and loaded)
        if self.is_depth_enabled() and self.depth_estimator.is_loaded:
            self._start_depth_thread_if_needed()

        logger.info("Camera processing started (non-blocking, thread-safe pipeline).")

    def stop_camera(self) -> None:
        """Stops camera feed and releases hardware cleanly."""
        if not self._is_camera_active:
            return

        self._stop_worker_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join(timeout=2.0)
        self._depth_thread = None

        self.webcam.stop()
        self._is_camera_active = False

        self.var_cam_status.set("Camera: Stopped")
        self.var_pill_status.set("● Stopped")
        self.lbl_status_pill.configure(bg="#F1F5F9", fg="#475569")
        self.set_status("Camera is stopped. Click Start to resume.")

        self.btn_start.configure(state=tk.NORMAL, bg=self.c_start, fg="#FFFFFF")
        self.btn_stop.configure(state=tk.DISABLED, bg="#F1F5F9", fg="#94A3B8")

        self._show_empty_detection_state()
        self._render_placeholder_video("Camera is off.\n\nPress Start Camera to begin.")
        logger.info("Camera processing stopped.")

    # =========================================================================
    # Depth Worker Thread — completely decoupled from Tkinter & Camera capture
    # =========================================================================
    def _depth_worker_loop(self) -> None:
        """Dedicated thread for depth inference. Runs at its own cadence.

        Never touches Tkinter variables or blocks the GUI.
        """
        logger.info("Depth worker loop started.")
        last_processed_frame_id = -1

        while not self._stop_worker_event.is_set():
            try:
                # Only run if depth is still enabled at runtime (thread-safe boolean)
                if not self.is_depth_enabled() or not self.depth_estimator.is_loaded:
                    time.sleep(0.05)
                    continue

                # Get latest frame from camera grabber
                ok, frame, fid = self.webcam.read_latest_frame()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                # Skip if we already processed this frame
                if fid == last_processed_frame_id:
                    time.sleep(0.005)
                    continue

                # Try to acquire depth-busy lock (non-blocking)
                acquired = self._depth_busy.acquire(blocking=False)
                if not acquired:
                    time.sleep(0.005)
                    continue

                try:
                    t_dep = time.perf_counter()
                    dmap = self.depth_estimator.estimate_depth_map(frame)
                    dep_ms = (time.perf_counter() - t_dep) * 1000.0

                    with self._depth_lock:
                        self._latest_depth_map = dmap
                        self._depth_ms = dep_ms

                    last_processed_frame_id = fid
                except Exception as de_err:
                    logger.error("Depth worker inference error: %s", de_err)
                finally:
                    self._depth_busy.release()

                time.sleep(0.005)
            except Exception as outer_err:
                logger.error("Unexpected exception in _depth_worker_loop: %s", outer_err, exc_info=True)
                time.sleep(0.05)

        logger.info("Depth worker loop exited.")

    # =========================================================================
    # Detection Worker Thread — runs YOLO, enriches, enqueues for GUI
    # =========================================================================
    def _camera_worker_loop(self) -> None:
        """Detection worker: grabs latest frame, runs YOLO, enriches, enqueues for GUI.

        Never touches Tkinter variables or blocks on depth inference.
        """
        logger.info("Detection worker loop started.")
        last_detections: List[DetectionResult] = []
        last_yolo_frame_id = -1
        frame_counter = 0
        fps_counter = 0
        fps_t0 = time.perf_counter()
        fps = 0.0
        log_interval_t = time.perf_counter()

        while not self._stop_worker_event.is_set():
            try:
                t0 = time.perf_counter()

                # Ensure depth worker is running if depth is ready
                if self.is_depth_enabled() and self.depth_estimator.is_loaded:
                    if self._depth_thread is None or not self._depth_thread.is_alive():
                        self._start_depth_thread_if_needed()

                # 0. Get latest frame from camera grabber (non-blocking)
                ok, frame, fid = self.webcam.read_latest_frame()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                infer_ms = 0.0

                # 1. Run YOLO Object Detection (respects frame_skip to conserve CPU)
                conf_val = self.get_confidence_threshold()
                frame_skip = max(1, getattr(self.config.model, "frame_skip", 1))
                if self.detector.is_loaded and fid != last_yolo_frame_id:
                    frame_counter += 1
                    if frame_counter % frame_skip == 0:
                        t_inf = time.perf_counter()
                        try:
                            last_detections = self.detector.detect(
                                frame,
                                confidence_override=conf_val,
                            )
                            infer_ms = (time.perf_counter() - t_inf) * 1000.0
                        except Exception as e:
                            logger.error("Detection exception: %s", e)
                    last_yolo_frame_id = fid

                # 2. Read latest depth result (non-blocking, from depth worker)
                depth_status = ""
                depth_ms = 0.0
                depth_enabled = self.is_depth_enabled()
                if depth_enabled and self.depth_estimator.is_loaded:
                    with self._depth_lock:
                        current_depth = self._latest_depth_map
                        depth_ms = self._depth_ms

                    depth_status = "Metric VITS"
                    detections_with_dist = self.depth_estimator.attach_distances_to_detections(
                        last_detections,
                        current_depth,
                        current_time=time.time(),
                    )
                else:
                    depth_status = "Disabled" if not depth_enabled else "Offline"
                    detections_with_dist = last_detections

                # 3. Attach Horizontal Position Guidance Metadata
                if frame is not None and detections_with_dist:
                    positioned_dets = [
                        self.filter_mgr.attach_position_metadata(d, frame_width=frame.shape[1])
                        for d in detections_with_dist
                    ]
                else:
                    positioned_dets = []

                # 4. Closest Object Prioritization & Temporal Distance Smoothing
                interval_val = self.get_interval_seconds()
                cool_val = self.get_cooldown_seconds()
                self.nearest_mgr.announcement_interval_seconds = interval_val
                self.nearest_mgr.cooldown_seconds = cool_val
                enriched_dets, nearest_obj = self.nearest_mgr.process_frame(
                    positioned_dets,
                    current_time=time.time(),
                    frame_width=frame.shape[1],
                )

                # 5. Evaluate Voice Guidance Alerts
                if depth_enabled and nearest_obj is not None:
                    voice_alert = self.nearest_mgr.evaluate_voice_alert(
                        nearest_obj,
                        current_time=time.time(),
                    )
                    if voice_alert:
                        phrase, det_obj, is_urgent = voice_alert
                        self.speech_engine.speak_async(phrase, priority=is_urgent)
                else:
                    self.filter_mgr.cooldown_seconds = cool_val
                    self.filter_mgr.announcement_interval_seconds = interval_val
                    alert_result = self.filter_mgr.evaluate_alert(
                        enriched_dets or positioned_dets,
                        current_time=time.time(),
                        confidence_threshold=conf_val,
                        frame_width=frame.shape[1],
                    )
                    if alert_result:
                        phrase, det_obj = alert_result
                        self.speech_engine.speak_async(phrase)

                # 6. Draw Visual Overlay with Closest Object Badge & Depth Telemetry
                grab_fps = self.webcam.grab_fps
                annotated_frame = self.detector.draw_overlays(
                    frame,
                    enriched_dets or positioned_dets,
                    fps=fps,
                    infer_ms=infer_ms,
                    depth_ms=depth_ms,
                    depth_status=depth_status,
                )

                # 7. Enqueue Frame for GUI Rendering (bounded, drop-oldest)
                queue_item = (annotated_frame, enriched_dets or positioned_dets, fid)
                try:
                    self._frame_queue.put_nowait(queue_item)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(queue_item)
                    except Exception:
                        pass

                # FPS tracking
                fps_counter += 1
                fps_elapsed = time.perf_counter() - fps_t0
                if fps_elapsed >= 1.0:
                    fps = fps_counter / fps_elapsed
                    fps_counter = 0
                    fps_t0 = time.perf_counter()

                # Diagnostics log every 1.0 second (Requirement 2)
                now_t = time.perf_counter()
                if now_t - log_interval_t >= 1.0:
                    cur_wall_time = time.time()
                    cam_alive = self.webcam.grabber_alive
                    cam_fid = self.webcam.frame_id
                    disp_fid = self._latest_displayed_fid
                    cap_ts = self.webcam.latest_capture_timestamp
                    disp_ts = self._latest_displayed_timestamp
                    time_since_cam = cur_wall_time - cap_ts if cap_ts > 0 else 0.0
                    time_since_gui = cur_wall_time - disp_ts if disp_ts > 0 else 0.0
                    yolo_alive = True
                    depth_alive = self._depth_thread is not None and self._depth_thread.is_alive()
                    depth_busy_flag = self._depth_busy.locked()
                    ui_dur = self._last_ui_callback_duration_ms
                    ram = _get_ram_mb()
                    ram_str = f"{ram:.0f}MB" if ram >= 0 else "N/A"

                    # Detect when preview stops updating
                    if disp_ts > 0 and time_since_gui > 1.5:
                        logger.warning(
                            "[FREEZE DETECTED] GUI preview stopped updating! Elapsed=%.2fs | CamFID=%d | DispFID=%d | CamAlive=%s | YoloAlive=%s | DepthAlive=%s",
                            time_since_gui, cam_fid, disp_fid, cam_alive, yolo_alive, depth_alive,
                        )

                    logger.info(
                        "[GUI DIAGNOSTICS] CamAlive=%s | CamFID=%d | DispFID=%d | LastCapTS=%.3f | LastDispTS=%.3f | TimeSinceCam=%.2fs | TimeSinceGUI=%.2fs | CamFPS=%.1f | DetFPS=%.1f | YoloAlive=%s | DepthAlive=%s | DepthBusy=%s | YOLO=%.1fms | Depth=%.1fms | UIDuration=%.2fms | RAM=%s",
                        cam_alive, cam_fid, disp_fid, cap_ts, disp_ts, time_since_cam, time_since_gui, grab_fps, fps, yolo_alive, depth_alive, depth_busy_flag, infer_ms, depth_ms, ui_dur, ram_str,
                    )
                    log_interval_t = now_t

                # Yield to keep CPU usage reasonable
                time.sleep(0.003)
            except Exception as loop_e:
                logger.error("Unexpected exception in _camera_worker_loop: %s", loop_e, exc_info=True)
                time.sleep(0.02)

        logger.info("Detection worker loop exited.")

    def _ui_update_loop(self) -> None:
        """Periodic GUI refresh for video frames and detection list."""
        if getattr(self, "_is_closing", False):
            return

        t_ui_start = time.perf_counter()
        try:
            item = self._frame_queue.get_nowait()
            if len(item) == 3:
                frame, detections, fid = item
            else:
                frame, detections = item
                fid = 0
            self._render_video_frame(frame, fid)
            self._update_detections_table(detections)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error("Error in UI update loop: %s", e, exc_info=True)
        finally:
            self._last_ui_callback_duration_ms = (time.perf_counter() - t_ui_start) * 1000.0

        if not getattr(self, "_is_closing", False):
            try:
                self.root.after(30, self._ui_update_loop)
            except Exception:
                pass

    def _render_video_frame(self, frame_bgr: np.ndarray, fid: int = 0) -> None:
        self._latest_displayed_timestamp = time.time()
        self._latest_displayed_fid = fid
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            return

        h, w, _ = frame_bgr.shape
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb_frame)
        self._photo_image = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        pos_x = (canvas_w - new_w) // 2
        pos_y = (canvas_h - new_h) // 2
        self.canvas.create_image(pos_x, pos_y, anchor=tk.NW, image=self._photo_image)

    def _render_placeholder_video(self, message: str) -> None:
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or 600
        canvas_h = self.canvas.winfo_height() or 440
        self.canvas.create_text(
            canvas_w // 2,
            canvas_h // 2,
            text=message,
            font=("Segoe UI", 13),
            fill="#94A3B8",
            justify=tk.CENTER,
        )

    def _show_empty_detection_state(self) -> None:
        for item in self.tree_detections.get_children():
            self.tree_detections.delete(item)
        self.tree_detections.insert("", tk.END, values=("No objects detected", "—", "—", "—", "—"))

    def _update_detections_table(self, detections: List[DetectionResult]) -> None:
        if getattr(self, "_is_closing", False):
            return

        deduped = self.filter_mgr.deduplicate(detections) if detections else []
        deduped.sort(key=lambda d: (not d.is_closest, -d.confidence))
        depth_active = self.is_depth_enabled()

        for item in self.tree_detections.get_children():
            self.tree_detections.delete(item)

        if not deduped:
            self._show_empty_detection_state()
            return

        for d in deduped:
            name = self.filter_mgr.format_class_name(d.class_name)
            pos_label = d.position or "Center"

            if d.distance is not None and d.distance_available:
                dist_str = f"~{d.distance:.1f} m"
            elif depth_active:
                dist_str = "Distance unavailable"
            else:
                dist_str = "—"

            closest_str = "[CLOSEST]" if d.is_closest else "—"
            conf_str = f"{int(d.confidence * 100)}%"

            self.tree_detections.insert(
                "",
                tk.END,
                values=(name, pos_label, dist_str, closest_str, conf_str),
            )

    def _on_toggle_voice(self) -> None:
        enabled = self.var_voice_enabled.get()
        self.update_settings(voice_enabled=enabled)
        self.speech_engine.set_enabled(enabled)
        if enabled:
            self.set_status("Voice alerts enabled.")
        else:
            self.set_status("Voice alerts turned off.")

    def _on_toggle_mute(self) -> None:
        new_muted = not self.var_muted.get()
        self.var_muted.set(new_muted)
        self.speech_engine.set_muted(new_muted)

        if new_muted:
            self.btn_mute.configure(text="Unmute", bg="#FEF3C7", fg="#92400E")
            self.set_status("Audio output muted.")
        else:
            self.btn_mute.configure(text="Mute", bg="#F1F5F9", fg="#475569")
            self.set_status("Audio output unmuted.")

    def _toggle_camera_shortcut(self) -> None:
        if self._is_camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def _toggle_voice_shortcut(self) -> None:
        self.var_voice_enabled.set(not self.var_voice_enabled.get())
        self._on_toggle_voice()

    def on_close(self) -> None:
        """Gracefully releases resources and exits."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        logger.info("Closing NaviGlass Desktop...")
        try:
            self.stop_camera()
        except Exception:
            pass
        try:
            self.detector.close()
        except Exception:
            pass
        try:
            self.depth_estimator.close()
        except Exception:
            pass
        try:
            self.speech_engine.stop()
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        logger.info("NaviGlass Desktop closed cleanly.")


def launch_gui(config: Optional[AppConfig] = None) -> None:
    cfg = config or AppConfig.load()
    root = tk.Tk()
    app = NaviGlassGUI(root, cfg)
    root.mainloop()
