"""Télémétrie GPU en direct (NVML, repli nvidia-smi) pour l'interface.

Renvoie un petit dict stable : nom, utilisation %, VRAM utilisée/totale (Mo),
température (°C), puissance (W). Si aucun GPU NVIDIA n'est visible :
{"available": False}.
"""
from __future__ import annotations

import subprocess

_nvml = None


def _nvml_handle():
    global _nvml
    if _nvml is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            _nvml = False
    return _nvml or None


def gpu_stats() -> dict:
    h = _nvml_handle()
    if h:
        nv, dev = h
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(dev)
            name = nv.nvmlDeviceGetName(dev)
            return {
                "available": True,
                "name": name.decode() if isinstance(name, bytes) else name,
                "util": int(nv.nvmlDeviceGetUtilizationRates(dev).gpu),
                "mem_used_mb": int(mem.used // 2**20),
                "mem_total_mb": int(mem.total // 2**20),
                "temp_c": int(nv.nvmlDeviceGetTemperature(dev, nv.NVML_TEMPERATURE_GPU)),
                "power_w": round(nv.nvmlDeviceGetPowerUsage(dev) / 1000, 1),
            }
        except Exception:
            pass
    try:  # repli : nvidia-smi
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()[0]
        name, util, used, total, temp, power = [p.strip() for p in out.split(",")]
        return {"available": True, "name": name, "util": int(float(util)),
                "mem_used_mb": int(float(used)), "mem_total_mb": int(float(total)),
                "temp_c": int(float(temp)), "power_w": round(float(power), 1)}
    except Exception:
        return {"available": False}
