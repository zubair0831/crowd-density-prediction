# ═══════════════════════════════════════════════════════════════════════════════
# zones.py — Dynamic zone engine, adaptive threshold v2.2, pressure field
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from constants import GRID_COLS, GRID_ROWS, EMA_ALPHA
from analyzers import fruin_los


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC ZONE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicZoneEngine:
    def __init__(
        self,
        grid_cols: int = GRID_COLS,
        grid_rows: int = GRID_ROWS,
        pixels_per_meter: Optional[float] = None,
    ) -> None:
        self.grid_cols        = grid_cols
        self.grid_rows        = grid_rows
        self.pixels_per_meter = pixels_per_meter

    def compute(
        self,
        coords:        List[List[float]],
        img_w:         int,
        img_h:         int,
        walkable_mask: np.ndarray,
        flow_zones:    Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Dict]:
        cell_w = img_w / self.grid_cols
        cell_h = img_h / self.grid_rows
        H, W   = walkable_mask.shape

        # Vectorised per-cell walkable pixel counts
        cell_walk = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        for r in range(self.grid_rows):
            r0 = int(r * H / self.grid_rows); r1 = int((r+1) * H / self.grid_rows)
            for c in range(self.grid_cols):
                c0 = int(c * W / self.grid_cols); c1 = int((c+1) * W / self.grid_cols)
                cell_walk[r, c] = int(walkable_mask[r0:r1, c0:c1].sum())

        cell_total_px = (H / self.grid_rows) * (W / self.grid_cols)

        # Assign persons to cells
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        for x, y in coords:
            col = min(int(x / cell_w), self.grid_cols - 1)
            row = min(int(y / cell_h), self.grid_rows - 1)
            grid_counts[row, col] += 1

        flow_map = {(fz["row"], fz["col"]): fz for fz in (flow_zones or [])}

        # Normalisation denominator
        wf_arr = cell_walk / cell_total_px
        dn_raw = np.where(wf_arr >= 0.10, grid_counts / np.maximum(wf_arr, 0.10), 0.0)
        norm_max = max(float(dn_raw.max()), 1.0)

        ppm   = self.pixels_per_meter
        zones: List[Dict] = []
        total_walkable_m2 = 0.0
        hot_zones = 0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                fz  = flow_map.get((r, c), {})
                cnt = int(grid_counts[r, c])
                wf  = float(wf_arr[r, c])
                is_walkable = wf >= 0.10

                cell_area_m2 = dens_sqm = None
                if ppm and ppm > 0 and is_walkable:
                    cell_area_m2   = int(cell_walk[r, c]) / (ppm ** 2)
                    dens_sqm       = cnt / max(cell_area_m2, 0.001)
                    total_walkable_m2 += cell_area_m2

                dn  = round(min(float(dn_raw[r, c]) / norm_max, 1.0), 3) if is_walkable else 0.0
                dx  = round(fz.get("mean_dx",    0.0), 2)
                dy  = round(fz.get("mean_dy",    0.0), 2)
                div = round(fz.get("divergence", 0.0), 4)
                mag = round(fz.get("magnitude",  0.0), 4)

                comp = max(0.0, -div / 0.5)
                turb = min(mag / 10.0, 1.0)
                risk = round(min(1.0, 0.50*dn + 0.30*comp + 0.20*turb), 3)
                if risk > 0.68: hot_zones += 1

                zones.append({
                    "row":           r, "col":          c,
                    "count":         cnt,
                    "density_norm":  dn,
                    "density_sqm":   round(dens_sqm, 2)   if dens_sqm    else None,
                    "walkable_frac": round(wf, 3),
                    "is_walkable":   is_walkable,
                    "cell_area_m2":  round(cell_area_m2, 2) if cell_area_m2 else None,
                    "los":           fruin_los(dens_sqm)   if dens_sqm    else None,
                    "dx": dx, "dy": dy, "divergence": div, "magnitude": mag,
                    "risk": risk,
                })

        aggregate = {
            "total_walkable_m2":   round(total_walkable_m2, 1),
            "hot_zones":           hot_zones,
            "walkable_zone_count": sum(1 for z in zones if z["is_walkable"]),
            "crush_risk_zones":    [z for z in zones if z["risk"] > 0.85],
        }
        return zones, aggregate


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD v2.2 — physically-grounded hard floor
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveThresholdV2:
    """
    Multi-signal adaptive threshold.

    FIX v3.2 hard floor:
      • Calibrated:     floor = walkable_m² × LOS-D density (2.7 p/m²) — alert fires
                        exactly when the crowd reaches uncomfortable density, regardless
                        of historical baseline.
      • Uncalibrated:   floor = max(window) × 1.10 + 3  (previous: ×1.25+8, too loose)

    This eliminates the previous pathological behaviour where a venue running at
    steady 60 people would need to reach 83 before any alert, while Fruin LOS-D
    fires correctly at 2.7 p/m².
    """

    TARGET_DENSITY_SQM = 2.7   # Fruin LOS-D — alert boundary
    PROX_ALPHA         = 0.12
    ZONE_ALPHA         = 0.08

    def __init__(
        self,
        smooth_win: int   = 15,
        ema_alpha:  float = EMA_ALPHA,
        z_factor:   float = 3.5,
    ) -> None:
        self.smooth_win = smooth_win
        self.ema_alpha  = ema_alpha
        self.z_factor   = z_factor
        self.window:   List[float] = []
        self.ema:      Optional[float] = None
        self.prev:     Optional[float] = None

    def reset(self) -> None:
        self.window = []; self.ema = self.prev = None

    def update(
        self,
        count:            int,
        walkable_meta:    Optional[Dict] = None,
        distance_meta:    Optional[Dict] = None,
        zone_aggregate:   Optional[Dict] = None,
        zone_list:        Optional[List[Dict]] = None,
        pixels_per_meter: Optional[float] = None,
    ) -> Dict:
        self.window.append(float(count))
        if len(self.window) > self.smooth_win: self.window.pop(0)
        n = len(self.window)

        if n < 5:
            naive = round(max(self.window) * 1.30 + 2)
            return {"threshold": naive, "mode": "warmup",
                    "mean": None, "std": None, "samples": n,
                    "signals": {"baseline": naive}}

        arr  = np.array(self.window)
        mean = float(arr.mean())
        std  = float(arr.std())
        med  = float(np.median(arr))

        self.ema = (self.ema_alpha * med + (1 - self.ema_alpha) * self.ema
                    if self.ema is not None else med)

        eff_std    = max(std, self.ema * 0.05)
        t_baseline = self.ema + self.z_factor * eff_std
        signals    = {"baseline": round(t_baseline, 1),
                      "mean": round(mean, 1), "std": round(std, 1)}

        # S2: area capacity cap
        t_area     = None
        walkable_m2 = zone_aggregate.get("total_walkable_m2", 0.0) if zone_aggregate else 0.0
        if walkable_m2 > 0 and pixels_per_meter:
            t_area = walkable_m2 * self.TARGET_DENSITY_SQM
            signals["area_cap"]    = round(t_area, 1)
            signals["walkable_m2"] = round(walkable_m2, 1)

        t_raw = min(t_baseline, t_area) if t_area else t_baseline
        signals["raw"] = round(t_raw, 1)

        # S3: proximity
        prox_score  = (distance_meta or {}).get("proximity_score", 0.0)
        prox_factor = 1.0 - self.PROX_ALPHA * prox_score
        signals["proximity_score"] = prox_score

        # S4: zone imbalance
        zone_factor = 1.0
        if zone_list:
            dn_vals = [z["density_norm"] for z in zone_list if z.get("is_walkable")]
            if len(dn_vals) > 1:
                zone_cv = float(np.std(dn_vals) / max(np.mean(dn_vals), 0.01))
                zone_cv = min(zone_cv, 1.0)
                zone_factor = 1.0 - self.ZONE_ALPHA * zone_cv
                signals["zone_cv"] = round(zone_cv, 3)

        t_adj = t_raw * prox_factor * zone_factor
        signals.update({
            "prox_factor": round(prox_factor, 3),
            "zone_factor": round(zone_factor, 3),
            "adjusted":    round(t_adj, 1),
        })

        # S5: temporal damping (max 2 % drop per frame)
        if self.prev is not None:
            t_final = max(t_adj, self.prev * 0.98)
        else:
            t_final = t_adj

        # ── Hard floor (v3.2 FIX) ────────────────────────────────────────────
        if t_area is not None:
            # Calibrated: floor = Fruin LOS-D area capacity
            hard_floor = int(walkable_m2 * self.TARGET_DENSITY_SQM)
        else:
            # Uncalibrated: 10 % above observed peak + small buffer
            hard_floor = int(max(self.window) * 1.10) + 3

        t_final = int(round(max(t_final, hard_floor, 1.0)))
        self.prev = t_final

        return {
            "threshold": t_final,
            "mode":      "dynamic_v2",
            "mean":      round(mean, 1),
            "std":       round(std, 1),
            "samples":   n,
            "signals":   signals,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PRESSURE FIELD
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pressure(zone_data: List[Dict], count_history: List[int]) -> Dict:
    if len(count_history) >= 3:
        n   = min(len(count_history), 5)
        win = np.array(count_history[-n:], dtype=float)
        x   = np.arange(n) - (n - 1) / 2
        y_m = win.mean()
        den = float((x * x).sum()) or 1e-9
        trend = max(-1.0, min(1.0, float((x * (win - y_m)).sum() / den) / max(y_m, 1) * 5))
    else:
        trend = 0.0

    cells: List[Dict] = []
    crush_zones: List[Dict] = []
    max_p = 0.0

    for z in zone_data:
        if not z.get("is_walkable", True): continue
        comp = max(0.0, -z["divergence"] / 0.5)
        p    = round(min(1.0, 0.50*z["density_norm"] + 0.30*comp + 0.20*max(0.0, trend)), 3)
        max_p = max(max_p, p)
        cells.append({"row": z["row"], "col": z["col"], "pressure": p})
        if p > 0.70:
            crush_zones.append({"row": z["row"], "col": z["col"]})

    return {
        "cells":            cells,
        "max_pressure":     round(max_p, 3),
        "crush_risk_zones": crush_zones,
        "trend_factor":     round(trend, 3),
    }