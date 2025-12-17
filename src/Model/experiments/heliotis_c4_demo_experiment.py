"""
heliotis_c4_demo_experiment.py

ONE FILE (experiment only) that matches your framework’s “Adding New Experiments” pattern:
  - Put in: src/Model/experiments/
  - Inherits Experiment
  - Defines: _DEFAULT_SETTINGS, _DEVICES, _EXPERIMENTS
  - Implements: _function() (experiment logic), _plot(), _update_plot()

IMPORTANT:
- This file intentionally contains **ZERO** code from your heliotis_c4_device.py device wrapper.
- It ONLY imports the device class and uses its public API: update(...), acquire(...), connect()/disconnect().

What it does:
- Demo sweep with illuminator directly facing the sensor:
    Sweep either modulation FREQUENCY, AMPLITUDE, or OFFSET
- For each sweep point:
    Acquire I/Q frames
    Compute lock-in amplitude image using the same RMS-AC method as your FastAPI /plot endpoint:
        I -= mean(I, axis=0), Q -= mean(Q, axis=0)
        amplitude = sqrt(mean(I^2 + Q^2, axis=0))
    Compute a scalar ROI metric (mean amplitude in ROI)
- Stores:
    self.data["amplitude_image"] (latest)
    self.data["sweep_values"], self.data["roi_metric"]
- Plots:
    axes_list[0] -> amplitude image
    axes_list[1] -> ROI metric vs sweep value (if you provide a second figure)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.experiment import Experiment
from src.core.parameter import Parameter

# ---- Import your device wrapper (NO device code included here) ----
# Adjust these import paths if your repo places the file elsewhere.
try:
    from src.Controller.heliotis_c4_device import HeliotisC4
except Exception:
    # fallback for running without the package layout (e.g., directly in a dev folder)
    from heliotis_c4_device import HeliotisC4  # type: ignore


def _make_sweep(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive sweep with sane handling of step sign."""
    start = float(start)
    stop = float(stop)
    step = float(step)

    if step == 0:
        return np.array([start], dtype=float)

    # Ensure step points toward stop
    if (stop - start) * step < 0:
        step = -step

    n = int(np.floor((stop - start) / step)) + 1
    if n < 1:
        return np.array([start], dtype=float)

    vals = start + step * np.arange(n, dtype=float)

    # Guard against a tiny floating rounding overshoot
    if step > 0:
        vals = vals[vals <= stop + 1e-12]
    else:
        vals = vals[vals >= stop - 1e-12]
    return vals


def _roi_mask(h: int, w: int, mode: str, settings: Dict[str, Any]) -> np.ndarray:
    """
    ROI mask generator.
    mode:
      - "full"
      - "center_circle"
      - "box"
    """
    mode = str(mode)

    if mode == "full":
        return np.ones((h, w), dtype=bool)

    if mode == "box":
        x0 = int(settings.get("roi_x0", 0))
        y0 = int(settings.get("roi_y0", 0))
        x1 = int(settings.get("roi_x1", w))
        y1 = int(settings.get("roi_y1", h))
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    # default: center_circle
    cx = int(settings.get("roi_cx", -1))
    cy = int(settings.get("roi_cy", -1))
    r = int(settings.get("roi_radius", 50))

    if cx < 0:
        cx = w // 2
    if cy < 0:
        cy = h // 2
    r = max(1, r)

    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _lockin_amplitude_rms_ac(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Match your FastAPI /plot behavior:
      I,Q: (frames, h, w)
      subtract per-pixel mean across frames, then RMS amplitude:
        sqrt(mean(I^2 + Q^2, axis=0))
    """
    I = I.astype(float, copy=False)
    Q = Q.astype(float, copy=False)
    I = I - I.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    return np.sqrt(np.mean(I ** 2 + Q ** 2, axis=0))


class HeliotisC4DemoSweep(Experiment):
    """
    Demo experiment for the Heliotis C4:
      - Sweeps illumination parameter
      - Measures lock-in amplitude and an ROI metric
    """

    _DEFAULT_SETTINGS = [
        # Sweep control
        Parameter("sweep_mode", "frequency", ["frequency", "amplitude", "offset"], "What to sweep"),
        Parameter("sweep_start", 9000.0, float, "Sweep start (Hz for frequency, % for amp/offset)"),
        Parameter("sweep_stop", 11000.0, float, "Sweep stop (Hz for frequency, % for amp/offset)"),
        Parameter("sweep_step", 250.0, float, "Sweep step (Hz for frequency, % for amp/offset)"),
        Parameter("settle_s", 0.15, float, "Wait after changing setting before acquiring"),

        # Acquisition
        Parameter("frames", 10, int, "Frames per acquisition burst"),
        Parameter("timeout_s", 30.0, float, "Fetch timeout (s)"),

        # ROI
        Parameter("roi_mode", "center_circle", ["full", "center_circle", "box"], "ROI type for scalar metric"),
        Parameter("roi_cx", -1, int, "ROI circle center x (-1 = image center)"),
        Parameter("roi_cy", -1, int, "ROI circle center y (-1 = image center)"),
        Parameter("roi_radius", 50, int, "ROI circle radius (px)"),
        Parameter("roi_x0", 0, int, "ROI box x0 (px)"),
        Parameter("roi_y0", 0, int, "ROI box y0 (px)"),
        Parameter("roi_x1", 0, int, "ROI box x1 (px, 0 = auto width)"),
        Parameter("roi_y1", 0, int, "ROI box y1 (px, 0 = auto height)"),

        # Device baseline config (pushed into the device before sweep)
        Parameter("trigger_mode", "On", ["On", "Off"], "FrameStart trigger mode"),
        Parameter("trigger_source", "Software", str, "Trigger source when trigger is On"),

        Parameter("lia_sensitivity", 0.2, float, "Lock-in sensitivity fraction"),
        Parameter("lia_n_periods", 10, int, "Lock-in time constant (periods)"),
        Parameter("lia_coupling", "DC", ["AC", "DC"], "Lock-in coupling"),
        Parameter("lia_ref_source", "Internal", ["Internal", "External"], "Reference source type"),
        Parameter("lia_exp_freq_dev_percent", 5, int, "Expected reference frequency deviation (%)"),

        Parameter("illum_mode", "On", ["On", "Off"], "Signal generator mode"),
        Parameter("illum_offset_percent", 20, int, "Signal generator offset (%)"),
        Parameter("illum_amplitude_percent", 10, int, "Signal generator amplitude (%)"),
        Parameter("illum_frequency_hz", 10000.0, float, "Signal generator frequency (Hz)"),

        # Cleanup
        Parameter("turn_off_illum_after", True, bool, "Turn illumination Off when experiment ends"),
        Parameter("disconnect_after", False, bool, "Disconnect camera when experiment ends"),
    ]

    # Declare required devices (framework uses the KEYS; values are not used by Experiment base here)
    _DEVICES = {"heliotis": HeliotisC4}

    _EXPERIMENTS = {}

    def __init__(self, devices: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(devices=devices, *args, **kwargs)

        # If the manager didn’t pass a device instance, create one locally.
        if "heliotis" not in self._devices or self._devices["heliotis"] is None:
            # Prevent any auto-connect surprises by explicitly setting auto_connect False if supported.
            try:
                self._devices["heliotis"] = HeliotisC4(name="heliotis", settings={"auto_connect": False})
            except Exception:
                self._devices["heliotis"] = HeliotisC4()

        # Plot state
        self._pg_image_item = None
        self._pg_curve_item = None

    def _function(self):
        dev = self._devices["heliotis"]

        # Connect if needed
        if hasattr(dev, "is_connected"):
            if not dev.is_connected:
                if hasattr(dev, "connect"):
                    dev.connect()
        else:
            if hasattr(dev, "connect"):
                dev.connect()

        # Push baseline config into device
        base_cfg = dict(
            trigger_mode=self.settings["trigger_mode"],
            trigger_source=self.settings["trigger_source"],

            lia_sensitivity=self.settings["lia_sensitivity"],
            lia_n_periods=self.settings["lia_n_periods"],
            lia_coupling=self.settings["lia_coupling"],
            lia_ref_source=self.settings["lia_ref_source"],
            lia_exp_freq_dev_percent=self.settings["lia_exp_freq_dev_percent"],

            illum_mode=self.settings["illum_mode"],
            illum_offset_percent=int(self.settings["illum_offset_percent"]),
            illum_amplitude_percent=int(self.settings["illum_amplitude_percent"]),
            illum_frequency_hz=float(self.settings["illum_frequency_hz"]),

            # acquisition defaults (your device wrapper uses these keys)
            acq_burst_frames=int(self.settings["frames"]),
            fetch_timeout_s=float(self.settings["timeout_s"]),
        )
        dev.update(base_cfg)

        # Build sweep
        sweep_mode = str(self.settings["sweep_mode"])
        sweep_vals = _make_sweep(self.settings["sweep_start"], self.settings["sweep_stop"], self.settings["sweep_step"])

        roi_metric = []
        roi_mask = None

        # Initialize data dict (so plotting can work mid-run)
        self.data = {
            "sweep_mode": sweep_mode,
            "sweep_values": sweep_vals.copy(),
            "roi_metric": np.array([], dtype=float),
            "amplitude_image": None,
        }

        n_total = len(sweep_vals) if len(sweep_vals) > 0 else 1

        for i, val in enumerate(sweep_vals):
            if self._abort:
                break

            # Update the swept parameter
            if sweep_mode == "frequency":
                # Keep reference and illumination in sync (common-sense demo behavior)
                dev.update({
                    "illum_frequency_hz": float(val),
                    "lia_ref_hz": int(round(float(val))),
                })
            elif sweep_mode == "amplitude":
                dev.update({"illum_amplitude_percent": int(round(float(val)))})
            elif sweep_mode == "offset":
                dev.update({"illum_offset_percent": int(round(float(val)))})
            else:
                raise ValueError(f"Unknown sweep_mode: {sweep_mode}")

            time.sleep(float(self.settings["settle_s"]))

            # Acquire
            I, Q = dev.acquire(frames=int(self.settings["frames"]), timeout_s=float(self.settings["timeout_s"]))

            # Compute amplitude image (RMS AC)
            amp = _lockin_amplitude_rms_ac(I, Q)

            # Build ROI mask once we know image shape
            if roi_mask is None:
                h, w = amp.shape
                # auto box end coords if left at 0
                if int(self.settings["roi_x1"]) == 0:
                    self.settings.update({"roi_x1": w})
                if int(self.settings["roi_y1"]) == 0:
                    self.settings.update({"roi_y1": h})
                roi_mask = _roi_mask(h, w, str(self.settings["roi_mode"]), dict(self.settings))

            m = float(np.mean(amp[roi_mask]))
            roi_metric.append(m)

            # Update experiment data for live plotting
            self.data["amplitude_image"] = amp
            self.data["roi_metric"] = np.array(roi_metric, dtype=float)

            # Progress update
            self.updateProgress.emit(int(round(100.0 * (i + 1) / n_total)))

        # Cleanup options
        if bool(self.settings["turn_off_illum_after"]):
            try:
                dev.update({"illum_mode": "Off"})
            except Exception:
                pass

        if bool(self.settings["disconnect_after"]) and hasattr(dev, "disconnect"):
            try:
                dev.disconnect()
            except Exception:
                pass

    # ---- Plotting (pyqtgraph) ----
    def _plot(self, axes_list):
        """
        axes_list comes from Experiment.get_axes_layout(...)
        - axes_list[0]: image plot
        - axes_list[1]: sweep curve plot (if a second figure is provided)
        """
        if not self.data:
            return

        # Import here so headless runs don’t require pyqtgraph
        try:
            import pyqtgraph as pg
        except Exception:
            return

        amp = self.data.get("amplitude_image", None)
        x = self.data.get("sweep_values", None)
        y = self.data.get("roi_metric", None)
        sweep_mode = self.data.get("sweep_mode", "")

        # ---- Image plot ----
        if len(axes_list) >= 1 and amp is not None:
            ax_img = axes_list[0]
            ax_img.clear()

            img_item = pg.ImageItem(amp)
            ax_img.addItem(img_item)
            ax_img.setAspectLocked(True)

            if hasattr(ax_img, "setLabel"):
                ax_img.setLabel("bottom", "X (px)")
                ax_img.setLabel("left", "Y (px)")
            if hasattr(ax_img, "setTitle"):
                ax_img.setTitle("Heliotis lock-in amplitude (RMS AC)")

        # ---- Curve plot ----
        if len(axes_list) >= 2 and x is not None and y is not None:
            ax_curve = axes_list[1]
            ax_curve.clear()

            y = np.asarray(y, dtype=float)
            x = np.asarray(x, dtype=float)
            x_used = x[: len(y)] if len(y) <= len(x) else x

            ax_curve.plot(x_used, y, pen=None, symbol="o", symbolSize=6)

            if hasattr(ax_curve, "setLabel"):
                ax_curve.setLabel("bottom", f"{sweep_mode}")
                ax_curve.setLabel("left", "ROI mean amplitude")
            if hasattr(ax_curve, "setTitle"):
                ax_curve.setTitle("ROI metric vs sweep")

    def _update_plot(self, axes_list):
        # Simple and safe: redraw. If you want faster updates later, swap to persistent ImageItem/PlotDataItem.
        self._plot(axes_list)
