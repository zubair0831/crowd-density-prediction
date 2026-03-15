# ═══════════════════════════════════════════════════════════════════════════════
# zones.py — Dynamic zone engine, adaptive threshold v2, pressure field
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
    """
    Replaces fixed 8×6 grid with walkable-area-aware zones.

    - Computes walkable fraction per cell from the mask.
    - Assigns persons to cells; density uses only walkable area.
    - Cells with walkable_fraction < 0.10 are excluded from threshold calc.
    - Zone risk: 50 % density_norm + 30 % flow compression + 20 % turbulence.
    """

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
        walkable_mask: np.ndarray,           # (H, W) uint8
        flow_zones:    Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Dict]:
        cell_w = img_w / self.grid_cols
        cell_h = img_h / self.grid_rows
        H, W   = walkable_mask.shape

        # Per-cell walkable pixel counts
        cell_walk = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                r0 = int(r * H / self.grid_rows);  r1 = int((r + 1) * H / self.grid_rows)
                c0 = int(c * W / self.grid_cols);  c1 = int((c + 1) * W / self.grid_cols)
                cell_walk[r, c] = int(walkable_mask[r0:r1, c0:c1].sum())

        cell_total_px = (H / self.grid_rows) * (W / self.grid_cols)
        max_walk = float(cell_total_px)

        # Assign persons to cells
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        for x, y in coords:
            col = min(int(x / cell_w), self.grid_cols - 1)
            row = min(int(y / cell_h), self.grid_rows - 1)
            grid_counts[row, col] += 1

        # Flow lookup
        flow_map: Dict[Tuple[int, int], Dict] = {}
        if flow_zones:
            for fz in flow_zones:
                flow_map[(fz["row"], fz["col"])] = fz

        # Max walkable-normalised count (for density_norm denominator)
        walkable_counts = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                wf = cell_walk[r, c] / max_walk
                if wf >= 0.10 and grid_counts[r, c] > 0:
                    walkable_counts.append(grid_counts[r, c] / max(wf, 0.10))
        norm_max = max(walkable_counts) if walkable_counts else 1.0

        ppm   = self.pixels_per_meter
        zones: List[Dict] = []
        total_walkable_m2 = 0.0
        hot_zones = 0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                fz  = flow_map.get((r, c), {})
                cnt = int(grid_counts[r, c])
                wp  = int(cell_walk[r, c])
                wf  = wp / max_walk
                is_walkable = wf >= 0.10

                # Calibrated density
                if ppm and ppm > 0 and is_walkable:
                    cell_area_m2 = wp / (ppm ** 2)
                    dens_sqm     = cnt / max(cell_area_m2, 0.001)
                    total_walkable_m2 += cell_area_m2
                else:
                    dens_sqm     = None
                    cell_area_m2 = None

                dn = (cnt / max(wf, 0.10)) / max(norm_max, 1) if is_walkable else 0.0
                dn = round(min(dn, 1.0), 3)

                dx  = round(fz.get("mean_dx", 0.0), 2)
                dy  = round(fz.get("mean_dy", 0.0), 2)
                div = round(fz.get("divergence", 0.0), 4)
                mag = round(fz.get("magnitude", 0.0), 4)

                comp = max(0.0, -div / 0.5)
                turb = min(mag / 10.0, 1.0)
                risk = round(min(1.0, 0.50 * dn + 0.30 * comp + 0.20 * turb), 3)

                if risk > 0.68:
                    hot_zones += 1

                zones.append({
                    "row":           r,
                    "col":           c,
                    "count":         cnt,
                    "density_norm":  dn,
                    "density_sqm":   round(dens_sqm, 2) if dens_sqm is not None else None,
                    "walkable_frac": round(wf, 3),
                    "is_walkable":   is_walkable,
                    "cell_area_m2":  round(cell_area_m2, 2) if cell_area_m2 else None,
                    "los":           fruin_los(dens_sqm) if dens_sqm else None,
                    "dx":            dx,
                    "dy":            dy,
                    "divergence":    div,
                    "magnitude":     mag,
                    "risk":          risk,
                })

        aggregate = {
            "total_walkable_m2":    round(total_walkable_m2, 1),
            "hot_zones":            hot_zones,
            "walkable_zone_count":  sum(1 for z in zones if z["is_walkable"]),
            "crush_risk_zones":     [z for z in zones if z["risk"] > 0.85],
        }
        return zones, aggregate


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD v2
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveThresholdV2:
    """
    Multi-signal threshold calculation.

    Signal stack
    ────────────
    S1  EMA baseline       (rolling median + exponential smoothing)
    S2  Walkable area cap  (walkable_m² × Fruin LOS-C target density 1.7 p/m²)
    S3  Proximity adjust   (high proximity_score lowers threshold)
    S4  Zone imbalance     (uneven distribution lowers threshold)
    S5  Temporal damping   (≤ 4 % drop per frame — safety bias)

    T_raw   = min(T_baseline, T_area_cap) if area cap available
    T_adj   = T_raw × (1 − α_prox × proximity_score) × (1 − α_zone × zone_cv)
    T_final = max(T_adj, prev_T × 0.96)
    """

    TARGET_DENSITY_SQM = 2.7   # Fruin LOS-D: threshold triggers only at uncomfortable density
    PROX_ALPHA         = 0.12  # reduced — proximity alone shouldn't cut threshold sharply
    ZONE_ALPHA         = 0.08  # reduced — zone imbalance is a weaker signal

    def __init__(
        self,
        smooth_win: int   = 10,  # wider window → smoother, less reactive baseline
        ema_alpha:  float = EMA_ALPHA,
        z_factor:   float = 3.0,  # 3σ above mean — far fewer false positives
    ) -> None:
        self.smooth_win = smooth_win
        self.ema_alpha  = ema_alpha
        self.z_factor   = z_factor
        self.window:    List[float] = []
        self.ema:       Optional[float] = None
        self.prev:      Optional[float] = None

    def reset(self) -> None:
        self.window = []
        self.ema    = None
        self.prev   = None

    def update(
        self,
        count:            int,
        walkable_meta:    Optional[Dict] = None,
        distance_meta:    Optional[Dict] = None,
        zone_aggregate:   Optional[Dict] = None,
        zone_list:        Optional[List[Dict]] = None,
        pixels_per_meter: Optional[float] = None,
    ) -> Dict:
        # S1: EMA baseline
        self.window.append(float(count))
        if len(self.window) > self.smooth_win:
            self.window.pop(0)
        n = len(self.window)

        if n < 5:
            naive = round(max(self.window) * 1.30 + 2)
            return {
                "threshold": naive, "mode": "warmup",
                "mean": None, "std": None, "samples": n,
                "signals": {"baseline": naive},
            }

        sw   = sorted(self.window)
        med  = (sw[n // 2 - 1] + sw[n // 2]) / 2 if n % 2 == 0 else sw[n // 2]
        mean = sum(self.window) / n
        std  = math.sqrt(sum((v - mean) ** 2 for v in self.window) / n)
        self.ema = (
            self.ema_alpha * med + (1 - self.ema_alpha) * self.ema
            if self.ema is not None else med
        )
        eff_std    = max(std, self.ema * 0.05)
        t_baseline = self.ema + self.z_factor * eff_std
        signals    = {"baseline": round(t_baseline, 1), "mean": round(mean, 1), "std": round(std, 1)}

        # S2: Walkable area capacity
        t_area = None
        if (
            walkable_meta and pixels_per_meter and pixels_per_meter > 0
            and zone_aggregate and zone_aggregate.get("total_walkable_m2")
        ):
            walkable_m2 = zone_aggregate["total_walkable_m2"]
            if walkable_m2 > 0:
                t_area = walkable_m2 * self.TARGET_DENSITY_SQM
                signals["area_cap"]    = round(t_area, 1)
                signals["walkable_m2"] = round(walkable_m2, 1)

        t_raw = min(t_baseline, t_area) if t_area else t_baseline
        signals["raw"] = round(t_raw, 1)

        # S3: Proximity adjustment
        prox_score = 0.0
        if distance_meta:
            prox_score = distance_meta.get("proximity_score", 0.0)
            signals["proximity_score"] = prox_score
        prox_factor = 1.0 - self.PROX_ALPHA * prox_score

        # S4: Zone imbalance
        zone_factor = 1.0
        if zone_list:
            dn_vals = [z["density_norm"] for z in zone_list if z.get("is_walkable")]
            if len(dn_vals) > 1:
                zone_cv = np.std(dn_vals) / max(np.mean(dn_vals), 0.01)
                zone_cv = min(zone_cv, 1.0)
                zone_factor = 1.0 - self.ZONE_ALPHA * zone_cv
                signals["zone_cv"] = round(float(zone_cv), 3)

        t_adj = t_raw * prox_factor * zone_factor
        signals["prox_factor"] = round(prox_factor, 3)
        signals["zone_factor"] = round(zone_factor, 3)
        signals["adjusted"]    = round(t_adj, 1)

        # S5: Temporal damping (max 2 % drop per frame — stronger safety bias)
        if self.prev is not None:
            t_final = max(t_adj, self.prev * 0.98)
        else:
            t_final = t_adj
        t_final    = int(round(max(t_final, 1.0)))
        # Hard floor: threshold must always exceed the highest count seen so far
        # by at least 3 people, preventing alerts when crowd is simply at baseline
        if self.window:
            hard_floor = int(max(self.window)) + 3
            t_final = max(t_final, hard_floor)
        self.prev  = t_final

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
        n = min(len(count_history), 5)
        win = count_history[-n:]
        x_m = (n - 1) / 2
        y_m = sum(win) / n
        num = sum((i - x_m) * (win[i] - y_m) for i in range(n))
        den = sum((i - x_m) ** 2 for i in range(n)) or 1e-9
        trend = max(-1.0, min(1.0, (num / den) / max(y_m, 1) * 5))
    else:
        trend = 0.0

    cells: List[Dict] = []
    crush_zones: List[Dict] = []
    max_p = 0.0

    for z in zone_data:
        if not z.get("is_walkable", True):
            continue
        comp = max(0.0, -z["divergence"] / 0.5)
        p    = round(min(1.0, 0.50 * z["density_norm"] + 0.30 * comp + 0.20 * max(0.0, trend)), 3)
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