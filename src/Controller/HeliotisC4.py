"""
HeliotisC4.py — Heliotis C4 lock-in camera device class (pittqlabsys format)

Drop this file into: src/Controller/  (or wherever your project expects device classes)

Requirements:
- harvesters (GenICam/GenTL)
- numpy
- A valid GenTL producer .cti file for the camera

Driver path:
- Either set env var DIAPHUS_GENTL64_FILE to the .cti path
- OR set the device setting 'cti_path' to the .cti path
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from harvesters.core import Harvester

from src.core.device import Device
from src.core.parameter import Parameter


class HeliotisC4(Device):
    """
    Device wrapper for a Heliotis C4 lock-in camera using Harvester (GenTL/GenICam).

    Exposes:
      - update(settings): apply settings and (if connected) push to hardware
      - read_probes(key=None): read cached acquisition data + basic camera state
      - is_connected: True if camera handle is alive
      - acquire(frames=None, timeout_s=None): trigger and fetch I/Q frames
      - connect(), disconnect()
    """

    _DEFAULT_SETTINGS = Parameter(
        [
            Parameter(
                "auto_connect",
                True,
                bool,
                "Connect to the camera during __init__",
            ),
            Parameter(
                "cti_env_var",
                "DIAPHUS_GENTL64_FILE",
                str,
                "Env var name containing the GenTL producer (.cti) path",
            ),
            Parameter(
                "cti_path",
                "",
                str,
                "Explicit .cti path (overrides env var if non-empty)",
            ),
            # Trigger
            Parameter(
                "trigger_mode",
                "On",
                ["On", "Off"],
                "FrameStart trigger mode",
            ),
            Parameter(
                "trigger_source",
                "Software",
                str,
                "Trigger source when trigger_mode=On (e.g. Software, Line1)",
            ),
            # LIA
            Parameter(
                "lia_sensitivity",
                0.2,
                float,
                "Lock-in sensitivity (fraction of max; e.g. 0.2 for S40U)",
            ),
            Parameter(
                "lia_n_periods",
                10,
                int,
                "Lock-in target time constant in number of periods",
            ),
            Parameter(
                "lia_coupling",
                "DC",
                ["AC", "DC"],
                "Lock-in coupling",
            ),
            Parameter(
                "lia_ref_hz",
                10000,
                int,
                "Target reference frequency in Hz",
            ),
            Parameter(
                "lia_ref_source",
                "Internal",
                ["Internal", "External"],
                "Reference source type",
            ),
            Parameter(
                "lia_exp_freq_dev_percent",
                5,
                int,
                "Expected frequency deviation (%)",
            ),
            # Illumination / signal generator
            Parameter(
                "illum_offset_percent",
                20,
                int,
                "Signal generator DC offset (%)",
            ),
            Parameter(
                "illum_amplitude_percent",
                10,
                int,
                "Signal generator amplitude (%)",
            ),
            Parameter(
                "illum_frequency_hz",
                9975.0,
                float,
                "Signal generator frequency (Hz)",
            ),
            Parameter(
                "illum_mode",
                "On",
                ["On", "Off"],
                "Signal generator mode",
            ),
            # Acquisition
            Parameter(
                "acq_burst_frames",
                10,
                int,
                "AcquisitionBurstFrameCount",
            ),
            Parameter(
                "fetch_timeout_s",
                30.0,
                float,
                "Buffer fetch timeout (seconds)",
            ),
        ]
    )

    _PROBES_TEMPLATE = {
        "i_data": "Last acquired I (in-phase) frames (frames x H x W)",
        "q_data": "Last acquired Q (quadrature) frames (frames x H x W)",
        "iq_data": "Last acquired combined array (2 x frames x H x W)",
        "height": "Current camera Height (pixels)",
        "width": "Current camera Width (pixels)",
        "acq_burst_frames": "Current AcquisitionBurstFrameCount (frames)",
        "device_info": "Harvester device info string for the opened device",
        "connected": "True if device is connected",
    }

    # Probes are defined as a CLASS attribute (Device base class exposes them read-only)
    _PROBES = deepcopy(_PROBES_TEMPLATE)

    def __init__(self, name: Optional[str] = None, settings: Optional[Dict[str, Any]] = None):
        # Internal handles / cached data
        self.harvester: Optional[Harvester] = None
        self.camera = None  # Harvester ImageAcquirer
        self._device_info_str: Optional[str] = None

        self.last_i_data: Optional[np.ndarray] = None
        self.last_q_data: Optional[np.ndarray] = None
        self.last_iq_data: Optional[np.ndarray] = None

        super().__init__(name=name, settings=None)


        if settings:
            self.update(settings)

        if "auto_connect" in self.settings and bool(self.settings["auto_connect"]):
            self.connect()

    # -----------------------------
    # Connection management
    # -----------------------------
    def connect(self) -> None:
        """Open the first available Heliotis C4 camera and apply current settings."""
        if self.camera is not None:
            self._is_connected = True
            return

        cti_path = str(self.settings.get("cti_path", "")).strip() if hasattr(self.settings, "get") else str(self.settings["cti_path"]).strip()
        if not cti_path:
            env_var = str(self.settings.get("cti_env_var", "DIAPHUS_GENTL64_FILE")) if hasattr(self.settings, "get") else str(self.settings["cti_env_var"])
            try:
                cti_path = os.environ[env_var]
            except KeyError as e:
                raise RuntimeError(
                    f"Missing GenTL producer path. Set env var {env_var} to your .cti file path "
                    f"or set device setting 'cti_path'."
                ) from e

        self.harvester = Harvester()
        self.harvester.add_file(cti_path)
        self.harvester.update()

        if len(self.harvester.device_info_list) == 0:
            self._safe_reset_harvester()
            raise RuntimeError("No Heliotis C4 camera found (Harvester device_info_list is empty).")

        self._device_info_str = str(self.harvester.device_info_list[0])
        self.camera = self.harvester.create(0)

        # Apply all current settings to hardware
        self._apply_all_settings_to_hardware()

        self._is_connected = True

    def disconnect(self) -> None:
        """Stop acquisition, destroy camera handle, reset harvester."""
        self._is_connected = False

        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None

        self._safe_reset_harvester()

        self.last_i_data = None
        self.last_q_data = None
        self.last_iq_data = None

    def _safe_reset_harvester(self) -> None:
        if self.harvester is not None:
            try:
                self.harvester.reset()
            except Exception:
                pass
            self.harvester = None

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    # -----------------------------
    # Required interface
    # -----------------------------
    def update(self, settings: Dict[str, Any]) -> None:
        """
        Update internal settings and push any relevant changes to the camera if connected.
        """
        super().update(settings)

        # Optional convenience toggles
        if "auto_connect" in settings and bool(settings["auto_connect"]) and not self.is_connected:
            self.connect()

        if not self.is_connected:
            return

        # Push changed settings to the camera
        self._apply_settings_to_hardware(settings)

    def read_probes(self, key: Optional[str] = None):
        """
        Read device probes.

        - read_probes() -> dict of all probes
        - read_probes('i_data') -> returns last I frames (or None)
        """
        if key is None:
            return {k: self.read_probes(k) for k in self._PROBES.keys()}

        if key not in self._PROBES:
            raise KeyError(f"Unknown probe '{key}'. Valid probes: {list(self._PROBES.keys())}")

        if key == "i_data":
            return self.last_i_data
        if key == "q_data":
            return self.last_q_data
        if key == "iq_data":
            return self.last_iq_data
        if key == "connected":
            return self.is_connected
        if key == "device_info":
            return self._device_info_str

        if not self.is_connected:
            return None

        nm = self.camera.remote_device.node_map
        if key == "height":
            return int(nm.Height.value)
        if key == "width":
            return int(nm.Width.value)
        if key == "acq_burst_frames":
            return int(nm.AcquisitionBurstFrameCount.value)

        return None

    @property
    def is_connected(self) -> bool:
        return bool(self._is_connected and self.camera is not None)

    # -----------------------------
    # Public acquisition API
    # -----------------------------
    def acquire(
        self,
        frames: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trigger a burst acquisition and return (I, Q) arrays.

        Data shape:
          I.shape == (frames, H, W)
          Q.shape == (frames, H, W)
        """
        if not self.is_connected:
            self.connect()

        nm = self.camera.remote_device.node_map

        if frames is not None:
            nm.AcquisitionBurstFrameCount.value = int(frames)

        if timeout_s is None:
            timeout_s = float(self.settings["fetch_timeout_s"])

        num_frames = int(nm.AcquisitionBurstFrameCount.value)
        height = int(nm.Height.value)
        width = int(nm.Width.value)
        output_shape = (2, num_frames, height, width)

        self.camera.start()

        # Trigger only if we're in triggered mode and using software trigger
        try:
            nm.TriggerSelector.value = "FrameStart"
            if str(self.settings["trigger_mode"]) == "On" and str(self.settings["trigger_source"]).lower() == "software":
                nm.TriggerSoftware.execute()
        except Exception:
            # Some configurations/drivers may not expose these nodes exactly as expected.
            # If trigger fails but free-run is possible, fetch may still work.
            pass

        with self.camera.fetch(timeout=float(timeout_s)) as buffer:
            data_components = []
            for component in buffer.payload.components:
                # Vendor demo behavior: keep 15 bits, then divide by 4
                arr = component.data % (2**15)
                arr = arr // 4
                data_components.append(arr)

            data = np.array(data_components).reshape(output_shape)

        self.camera.stop()

        self.last_iq_data = data
        self.last_i_data = data[0]
        self.last_q_data = data[1]
        return self.last_i_data, self.last_q_data

    # -----------------------------
    # Hardware configuration helpers
    # -----------------------------
    def _apply_all_settings_to_hardware(self) -> None:
        """Push *all* current settings to the camera."""
        self._apply_settings_to_hardware(dict(self.settings))

    def _apply_settings_to_hardware(self, changed: Dict[str, Any]) -> None:
        """Push a subset of settings to the camera (only if connected)."""
        if not self.is_connected:
            return

        nm = self.camera.remote_device.node_map

        # Always safe to re-apply config blocks if any key in that block changed
        trigger_keys = {"trigger_mode", "trigger_source"}
        lia_keys = {
            "lia_sensitivity",
            "lia_n_periods",
            "lia_coupling",
            "lia_ref_hz",
            "lia_ref_source",
            "lia_exp_freq_dev_percent",
        }
        illum_keys = {
            "illum_offset_percent",
            "illum_amplitude_percent",
            "illum_frequency_hz",
            "illum_mode",
        }

        if trigger_keys.intersection(changed.keys()):
            self._configure_trigger()

        if lia_keys.intersection(changed.keys()):
            self._configure_lia()

        if illum_keys.intersection(changed.keys()):
            self._configure_illumination()

        if "acq_burst_frames" in changed:
            nm.AcquisitionBurstFrameCount.value = int(self.settings["acq_burst_frames"])

    def _configure_trigger(self) -> None:
        nm = self.camera.remote_device.node_map

        # Disable RecordingStart trigger (avoid external gating unless you explicitly want it)
        nm.TriggerSelector.value = "RecordingStart"
        nm.TriggerMode.value = "Off"

        # FrameStart trigger
        nm.TriggerSelector.value = "FrameStart"
        nm.TriggerMode.value = str(self.settings["trigger_mode"])

        if str(self.settings["trigger_mode"]) == "On":
            nm.TriggerSource.value = str(self.settings["trigger_source"])

    def _configure_lia(self) -> None:
        nm = self.camera.remote_device.node_map

        nm.DeviceOperationMode.value = "LockInCam"
        nm.Scan3dExtractionMethod.value = "rawIQ"

        nm.LockInSensitivity.value = float(self.settings["lia_sensitivity"])
        nm.LockInTargetTimeConstantNPeriods.value = int(self.settings["lia_n_periods"])
        nm.LockInCoupling.value = str(self.settings["lia_coupling"])

        nm.LockInExpectedFrequencyDeviation.value = int(self.settings["lia_exp_freq_dev_percent"])
        nm.LockInTargetReferenceFrequency.value = int(self.settings["lia_ref_hz"])
        nm.LockInReferenceSourceType.value = str(self.settings["lia_ref_source"])

        if str(self.settings["lia_ref_source"]) == "External":
            nm.LockInReferenceFrequencyScaler.value = "Off"
            nm.LockInReferenceSourceSignal.value = "FI2"

    def _configure_illumination(self) -> None:
        nm = self.camera.remote_device.node_map

        nm.SignalGeneratorOffset.value = int(self.settings["illum_offset_percent"])
        nm.SignalGeneratorAmplitude.value = int(self.settings["illum_amplitude_percent"])
        nm.SignalGeneratorFrequency.value = float(self.settings["illum_frequency_hz"])
        nm.SignalGeneratorMode.value = str(self.settings["illum_mode"])

        nm.LightControllerSelector.value = "LightController0"
        nm.LightControllerSource.value = "SignalGenerator"
