"""Energy measurement.

Primary backend (Apple Silicon, no admin rights needed): Zeus `zeus-apple-silicon`, which reads the
on-chip energy counters (IOReport) for CPU, GPU, GPU SRAM, ANE and DRAM. CodeCarbon cannot see Apple GPU
power without root (it reports 0 W for the GPU), so it is used only as a last-resort fallback and
results are flagged via `energy_mode`.

Carbon = energy x grid carbon intensity. CARBON_INTENSITY (kgCO2e/kWh) defaults to 0.20, a round
placeholder for the UK grid. VERIFY and set the exact value from the UK DESNZ greenhouse-gas conversion
factors for the year you cite before reporting numbers. Energy (J, kWh) is the primary result; CO2e is
just a linear rescaling of it.
"""
from __future__ import annotations

import os
import time
import uuid

CARBON_INTENSITY = float(os.environ.get("CARBON_INTENSITY", 0.20))  # kgCO2e per kWh

try:
    from zeus_apple_silicon import AppleEnergyMonitor
    _MON = AppleEnergyMonitor()
except Exception:  # not Apple Silicon / package missing
    _MON = None


def zeus_available() -> bool:
    return _MON is not None


class PowerMeter:
    """Context manager measuring energy (J) of the enclosed block, with a per-component breakdown."""

    idle_watts: float | None = None  # shared idle baseline (W)

    def __init__(self):
        self.mode = "zeus_apple_silicon" if _MON is not None else "codecarbon_cpu_estimate"
        self.parts: dict[str, float] = {}

    def __enter__(self):
        self.t0 = time.time()
        if self.mode == "zeus_apple_silicon":
            self._win = f"w{uuid.uuid4().hex}"
            _MON.begin_window(self._win)
        else:
            from codecarbon import OfflineEmissionsTracker
            self.tracker = OfflineEmissionsTracker(country_iso_code="GBR", log_level="error",
                                                   save_to_file=False, measure_power_secs=1)
            self.tracker.start()
        return self

    def __exit__(self, *exc):
        if self.mode == "zeus_apple_silicon":
            e = _MON.end_window(self._win)
            self.seconds = time.time() - self.t0
            mj = lambda v: float(v or 0.0)
            self.parts = {"cpu": mj(e.cpu_total_mj) / 1e3, "gpu": (mj(e.gpu_mj) + mj(e.gpu_sram_mj)) / 1e3,
                          "dram": mj(e.dram_mj) / 1e3, "ane": mj(e.ane_mj) / 1e3}
            self.joules = sum(self.parts.values())
        else:
            self.tracker.stop()
            self.seconds = time.time() - self.t0
            self.joules = float(self.tracker.final_emissions_data.energy_consumed) * 3.6e6
        self.mean_watts = self.joules / max(self.seconds, 1e-9)
        idle = PowerMeter.idle_watts or 0.0
        self.dynamic_joules = max(self.joules - idle * self.seconds, 0.0)
        return False

    def result(self, prefix: str = "") -> dict:
        kwh = self.joules / 3.6e6
        out = {f"{prefix}seconds": self.seconds, f"{prefix}joules": self.joules,
               f"{prefix}dynamic_joules": self.dynamic_joules, f"{prefix}kwh": kwh,
               f"{prefix}kgco2e": kwh * CARBON_INTENSITY, f"{prefix}mean_watts": self.mean_watts,
               f"{prefix}energy_mode": self.mode}
        out.update({f"{prefix}{k}_joules": v for k, v in self.parts.items()})
        return out


def measure_idle(seconds: float = 15.0) -> float:
    """Measure and store idle power (W) so we can also report dynamic (model-attributable) energy."""
    with PowerMeter() as pm:
        time.sleep(seconds)
    PowerMeter.idle_watts = pm.mean_watts
    return pm.mean_watts
