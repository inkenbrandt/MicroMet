"""
easyflux_footprint.py
=====================
Python translation of the EasyFlux-DL CRBASIC footprint subroutines for
recalculating FETCH values from eddy-covariance station data.

Implements:
    - FootprintCharacteristics_Kljun  (Kljun et al., 2004)
    - FootprintCharacteristics_KormannMeixner (Kormann & Meixner, 2001)
    - Footprint model selection logic from the EasyFlux-DL main program

References:
    Kljun, N., et al. (2004). A simple parameterisation for flux footprint
        predictions. Boundary-Layer Meteorology, 112, 503–523.
    Kormann, R. & Meixner, F.X. (2001). An analytical footprint model for
        non-neutral stratification. Boundary-Layer Meteorology, 99, 207–224.
    Nemes, G. (2007). Approximation of the Gamma function (Stirling-type).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

# ---------------------------------------------------------------------------
# Physical & numerical constants
# ---------------------------------------------------------------------------
K_VON_KARMAN = 0.4          # von Kármán constant
PI = np.pi
NMBR_INT_INTERV_SEGMENT = 100  # Default integration sub-intervals per segment


# ---------------------------------------------------------------------------
# Site configuration
# ---------------------------------------------------------------------------
@dataclass
class SiteConfig:
    """Site-specific parameters for footprint recalculation.

    Parameters
    ----------
    z : float
        Aerodynamic measurement height [m] (height above zero-plane displacement).
    z0 : float
        Roughness length [m].  Set to 0 to enable automatic estimation under
        neutral conditions (as in EasyFlux-DL when roughness_user=0).
    sonic_azimuth : float
        Compass bearing of the sonic anemometer's positive-x axis [degrees].
        Used to convert compass WD back to WD_SONIC for sector selection.
    dist_intrst : dict
        Upwind distance of interest [m] per wind-direction sector keyed by
        sector name.  Default sectors follow EasyFlux-DL convention:
        ``{'60_300': d1, '60_170': d2, '170_190': d3, '190_300': d4}``
        If a single float is given, it is used for all sectors.
    n_int : int
        Number of integration sub-intervals per segment (default 100).
    """
    z: float = 1.64
    z0: float = 0.01
    sonic_azimuth: float = 0.0
    dist_intrst: Dict[str, float] = field(default_factory=lambda: {
        '60_300': 500.0,
        '60_170': 500.0,
        '170_190': 500.0,
        '190_300': 500.0,
    })
    n_int: int = NMBR_INT_INTERV_SEGMENT

    def set_uniform_dist(self, d: float):
        """Set the same upwind distance of interest for all sectors."""
        for k in self.dist_intrst:
            self.dist_intrst[k] = d


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FootprintResult:
    """Container for a single-period footprint calculation."""
    fetch_max: float = np.nan
    fetch_90: float = np.nan
    fetch_55: float = np.nan
    fetch_40: float = np.nan
    fp_dist_intrst: float = np.nan   # cumulative footprint [%] at upwind dist of interest
    fp_equation: str = ''


# ---------------------------------------------------------------------------
#  Kljun et al. (2004) footprint model
# ---------------------------------------------------------------------------

def _pbl_height_kljun(obukhov: float) -> float:
    """Estimate planetary boundary-layer height from Obukhov length.

    Piecewise-linear interpolation based on Table I of Kljun et al. (2004),
    exactly as implemented in the EasyFlux-DL CRBASIC subroutine.
    """
    if obukhov <= 0.0:
        if obukhov < -1013.3:
            return 1000.0
        elif obukhov <= -650.0:
            return 1200.0 - 200.0 * ((obukhov + 650.0) / (-1013.3 + 650.0))
        elif obukhov <= -30.0:
            return 1500.0 - 300.0 * ((obukhov + 30.0) / (-650.0 + 30.0))
        elif obukhov <= -5.0:
            return 2000.0 - 500.0 * ((obukhov + 5.0) / (-30.0 + 5.0))
        else:  # -5 < obukhov <= 0
            return 2000.0 + 20.0 * (obukhov + 5.0)
    else:  # obukhov > 0
        if obukhov > 1316.4:
            return 1000.0
        elif obukhov >= 1000.0:
            return 800.0 + 200.0 * ((obukhov - 1000.0) / (1316.4 - 1000.0))
        elif obukhov >= 130.0:
            return 250.0 + 550.0 * ((obukhov - 130.0) / (1000.0 - 130.0))
        elif obukhov >= 84.0:
            return 200.0 + 50.0 * ((obukhov - 84.0) / (130.0 - 84.0))
        else:  # 0 < obukhov < 84
            return 200.0 - (84.0 - obukhov) * (50.0 / 46.0)


def _kljun_footprint_value(x: float, k1_suz_zh: float, suz: float,
                           k2: float, k3: float, k4: float) -> float:
    """Evaluate the Kljun footprint density at distance x."""
    t = (suz * x + k4) / k3
    return k1_suz_zh * (t ** k2) * np.exp(k2 * (1.0 - t))


def footprint_kljun(u_star: float, sigma_w: float, z: float,
                    obukhov: float, z0: float, upwnd_dist: float,
                    n_int: int = NMBR_INT_INTERV_SEGMENT) -> FootprintResult:
    """Kljun et al. (2004) footprint model.

    Parameters
    ----------
    u_star : float   – friction velocity [m/s]
    sigma_w : float  – std dev of vertical velocity [m/s]
    z : float        – aerodynamic measurement height [m]
    obukhov : float  – Monin-Obukhov length [m]
    z0 : float       – roughness length [m]
    upwnd_dist : float – upwind distance of interest [m]
    n_int : int      – integration sub-intervals per segment

    Returns
    -------
    FootprintResult with fetch_max, fetch_90/55/40, fp_dist_intrst.
    """
    res = FootprintResult(fp_equation='Kljun et al')

    if np.isnan(u_star) or np.isnan(sigma_w) or np.isnan(obukhov):
        return res

    # Model parameters (eqs. 13–16 in Kljun et al. 2004)
    ln_z0_term = 3.418 - np.log(z0)
    k1 = 0.175 / ln_z0_term
    k2 = 3.68254
    k3 = 4.277 * ln_z0_term
    k4 = 1.685 * ln_z0_term

    # PBL height
    h_pbl = _pbl_height_kljun(obukhov)

    # Composite variables
    zh_ratio = z / h_pbl
    suz = ((sigma_w / u_star) ** 0.8) / z
    k1_suz_zh = k1 * suz * (1.0 - zh_ratio)

    # Peak location and inflection points
    x_max = (k3 - k4) / suz
    x_infl_L = x_max * (k3 * ((np.sqrt(k2) - 1.0) / np.sqrt(k2)) - k4) / (k3 - k4)
    x_infl_R = x_max * (k3 * ((np.sqrt(k2) + 1.0) / np.sqrt(k2)) - k4) / (k3 - k4)

    res.fetch_max = x_max

    # Accumulators
    fp_cum = 0.0
    fp_win = 0.0
    fp_90 = 0.0
    fp_55 = 0.0
    fp_40 = 0.0

    # Helper for trapezoidal accumulation + threshold detection
    def _accum_trap(x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
                    fp_win, fp_90, fp_55, fp_40):
        fp_cum += intv * (fp_L + fp_R) / 2.0
        if x_L < upwnd_dist <= x_R:
            fp_win = 100.0 * (fp_cum_prev + (fp_cum - fp_cum_prev) *
                              (upwnd_dist - x_L) / intv)
        if fp_cum >= 0.4 and fp_40 == 0.0:
            fp_40 = x_R - intv * (fp_cum - 0.4) / (fp_cum - fp_cum_prev) if fp_cum != fp_cum_prev else x_R
        if fp_cum >= 0.55 and fp_55 == 0.0:
            fp_55 = x_R - intv * (fp_cum - 0.55) / (fp_cum - fp_cum_prev) if fp_cum != fp_cum_prev else x_R
        if fp_cum >= 0.9 and fp_90 == 0.0:
            fp_90 = x_R - intv * (fp_cum - 0.9) / (fp_cum - fp_cum_prev) if fp_cum != fp_cum_prev else x_R
        return fp_cum, fp_win, fp_90, fp_55, fp_40

    # Helper for Boole's rule accumulation
    def _accum_boole(x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
                     fp_win, fp_90, fp_55, fp_40, eval_fn):
        fp_m1 = eval_fn(x_L + 0.25 * intv)
        fp_m2 = eval_fn(x_L + 0.50 * intv)
        fp_m3 = eval_fn(x_L + 0.75 * intv)
        fp_cum += intv * (7.0 * fp_L + 32.0 * fp_m1 + 12.0 * fp_m2 +
                          32.0 * fp_m3 + 7.0 * fp_R) / 90.0
        if x_L < upwnd_dist <= x_R:
            fp_win = 100.0 * (fp_cum_prev + (fp_cum - fp_cum_prev) *
                              (upwnd_dist - x_L) / intv)
        if fp_cum >= 0.4 and fp_40 == 0.0:
            fp_40 = x_R - intv * (fp_cum - 0.4) / (fp_cum - fp_cum_prev) if fp_cum != fp_cum_prev else x_R
        if fp_cum >= 0.55 and fp_55 == 0.0:
            fp_55 = x_R - intv * (fp_cum - 0.55) / (fp_cum - fp_cum_prev) if fp_cum != fp_cum_prev else x_R
        if fp_cum >= 0.9 and fp_90 == 0.0:
            fp_90 = x_R - intv * (fp_cum - 0.9) / (fp_cum - fp_cum_prev) if fp_cum != fp_cum_prev else x_R
        return fp_cum, fp_win, fp_90, fp_55, fp_40

    eval_kljun = lambda x: _kljun_footprint_value(x, k1_suz_zh, suz, k2, k3, k4)

    # --- Segment 1: start (-k4/suz) → left inflection point ---
    x_R = -k4 / suz
    intv = (x_infl_L - x_R) / n_int
    fp_R = 0.0

    for _ in range(n_int):
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_kljun(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_trap(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)

    # --- Segment 2: left inflection → x_max ---
    intv = (x_max - x_infl_L) / n_int
    for _ in range(n_int):
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_kljun(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_trap(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)

    # --- Segment 3: x_max → past right inflection (2× segment width) ---
    intv = (x_infl_R - x_max) / n_int
    found_90 = False
    for _ in range(2 * n_int):
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_kljun(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_trap(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)
        if fp_90 > 0:
            found_90 = True
            break

    # --- Segment 4: Boole's rule, coarser interval ---
    intv = 4.0 * z
    while fp_cum < 0.9 and (x_R - x_max) < 200.0 * z:
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_kljun(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_boole(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40, eval_kljun)

    # Check 90% after segment 4
    if fp_cum >= 0.9 and fp_90 == 0.0:
        fp_90 = x_R - intv * (fp_cum - 0.9) / max(fp_cum - fp_cum_prev, 1e-30)

    if fp_cum < 0.9 and fp_90 == 0.0:
        fp_90 = np.nan

    # --- Segment 5: extend if upwind distance not yet reached ---
    if x_R < upwnd_dist:
        if (upwnd_dist - x_R) < 100.0 * z:
            interval_count = max(int((upwnd_dist - x_R) / intv), 1)
            intv = (upwnd_dist - x_R) / interval_count
            for _ in range(interval_count):
                x_L = x_R
                x_R += intv
                fp_cum_prev = fp_cum
                fp_L = fp_R
                fp_R = eval_kljun(x_R)
                fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_boole(
                    x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
                    fp_win, fp_90, fp_55, fp_40, eval_kljun)
                if fp_cum >= 0.9 and np.isnan(fp_90):
                    fp_90 = x_R - intv * (fp_cum - 0.9) / max(fp_cum - fp_cum_prev, 1e-30)

            if fp_win == 0.0:
                fp_win = min(100.0 * fp_cum, 99.0) if fp_cum < 1.0 else 99.0
        else:
            fp_win = 99.0

    res.fetch_max = x_max
    res.fetch_90 = fp_90
    res.fetch_55 = fp_55
    res.fetch_40 = fp_40
    res.fp_dist_intrst = fp_win
    return res


# ---------------------------------------------------------------------------
#  Kormann & Meixner (2001) footprint model
# ---------------------------------------------------------------------------

def _gamma_nemes(mu: float) -> float:
    """Gamma function of mu using Nemes (2007) approximation.

    Identical to the CRBASIC implementation.
    """
    return np.sqrt(2.0 * PI / mu) * (
        ((mu + 1.0 / (12.0 * mu - 0.1 / mu)) / np.e) ** mu
    )


def footprint_kormann_meixner(u_star: float, z: float, stability: float,
                               u_total: float, upwnd_dist: float,
                               n_int: int = NMBR_INT_INTERV_SEGMENT) -> FootprintResult:
    """Kormann & Meixner (2001) footprint model.

    Parameters
    ----------
    u_star : float     – friction velocity [m/s]
    z : float          – aerodynamic measurement height [m]
    stability : float  – z/L (Monin-Obukhov stability parameter)
    u_total : float    – resultant wind speed [m/s]
    upwnd_dist : float – upwind distance of interest [m]
    n_int : int        – integration sub-intervals per segment

    Returns
    -------
    FootprintResult with fetch_max, fetch_90/55/40, fp_dist_intrst.
    """
    res = FootprintResult(fp_equation='KormannMeixner')
    k = K_VON_KARMAN

    if np.isnan(u_star) or np.isnan(stability) or np.isnan(u_total):
        return res

    # Vertical profile exponents depending on stability
    if stability > 0:
        stab_clamped = min(stability, 4.0)
        m_km = (u_star / (k * u_total)) * (1.0 + 5.0 * stab_clamped)
        n_km = 1.0 / (1.0 + 5.0 * stab_clamped)
        phi_c = 1.0 + 5.0 * stab_clamped
    else:
        stab_clamped = max(stability, -4.0)
        m_km = (u_star / (k * u_total)) / ((1.0 - 16.0 * stab_clamped) ** 0.25)
        n_km = (1.0 - 24.0 * stab_clamped) / (1.0 - 16.0 * stab_clamped)
        phi_c = 1.0 / np.sqrt(1.0 - 16.0 * stab_clamped)

    # Composite variables
    wnd_const = u_total / (z ** m_km)
    r_km = 2.0 + m_km - n_km
    kp = (k * u_star * z ** (1.0 - n_km)) / phi_c
    xi = wnd_const / (kp * r_km * r_km)
    mu = (m_km + 1.0) / r_km

    gamma_mu = _gamma_nemes(mu)
    xgz = ((xi ** mu) * (z ** (m_km + 1.0))) / gamma_mu
    xz = xi * (z ** r_km)

    # Peak and inflection points
    x_max = xz / (mu + 1.0)
    x_infl_L = x_max * (1.0 - 1.0 / np.sqrt(mu + 2.0))
    x_infl_R = x_max * (1.0 + 1.0 / np.sqrt(mu + 2.0))

    res.fetch_max = x_max

    # Footprint density function for K&M
    def eval_km(x):
        if x <= 0:
            return 0.0
        return xgz * np.exp(-xz / x) / (x ** (mu + 1.0))

    # Accumulators
    fp_cum = 0.0
    fp_win = 0.0
    fp_90 = 0.0
    fp_55 = 0.0
    fp_40 = 0.0

    def _accum_trap(x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
                    fp_win, fp_90, fp_55, fp_40):
        fp_cum += intv * (fp_L + fp_R) / 2.0
        if x_L < upwnd_dist <= x_R:
            fp_win = 100.0 * (fp_cum_prev + (fp_cum - fp_cum_prev) *
                              (upwnd_dist - x_L) / intv)
        if fp_cum >= 0.4 and fp_40 == 0.0:
            fp_40 = x_R - intv * (fp_cum - 0.4) / max(fp_cum - fp_cum_prev, 1e-30)
        if fp_cum >= 0.55 and fp_55 == 0.0:
            fp_55 = x_R - intv * (fp_cum - 0.55) / max(fp_cum - fp_cum_prev, 1e-30)
        if fp_cum >= 0.9 and fp_90 == 0.0:
            fp_90 = x_R - intv * (fp_cum - 0.9) / max(fp_cum - fp_cum_prev, 1e-30)
        return fp_cum, fp_win, fp_90, fp_55, fp_40

    def _accum_boole(x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
                     fp_win, fp_90, fp_55, fp_40):
        fp_m1 = eval_km(x_L + 0.25 * intv)
        fp_m2 = eval_km(x_L + 0.50 * intv)
        fp_m3 = eval_km(x_L + 0.75 * intv)
        fp_cum += intv * (7.0 * fp_L + 32.0 * fp_m1 + 12.0 * fp_m2 +
                          32.0 * fp_m3 + 7.0 * fp_R) / 90.0
        if x_L < upwnd_dist <= x_R:
            fp_win = 100.0 * (fp_cum_prev + (fp_cum - fp_cum_prev) *
                              (upwnd_dist - x_L) / intv)
        if fp_cum >= 0.4 and fp_40 == 0.0:
            fp_40 = x_R - intv * (fp_cum - 0.4) / max(fp_cum - fp_cum_prev, 1e-30)
        if fp_cum >= 0.55 and fp_55 == 0.0:
            fp_55 = x_R - intv * (fp_cum - 0.55) / max(fp_cum - fp_cum_prev, 1e-30)
        if fp_cum >= 0.9 and fp_90 == 0.0:
            fp_90 = x_R - intv * (fp_cum - 0.9) / max(fp_cum - fp_cum_prev, 1e-30)
        return fp_cum, fp_win, fp_90, fp_55, fp_40

    # --- Segment 1: 0+ → left inflection ---
    intv = x_infl_L / n_int
    x_R = 0.0
    fp_R = 0.0

    for _ in range(n_int):
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_km(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_trap(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)

    # --- Segment 2: left inflection → x_max ---
    intv = (x_max - x_infl_L) / n_int
    for _ in range(n_int):
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_km(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_trap(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)

    # --- Segment 3: x_max → past right inflection (2× segment) ---
    intv = (x_infl_R - x_max) / n_int
    found_90 = False
    for _ in range(2 * n_int):
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_km(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_trap(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)
        if fp_90 > 0:
            found_90 = True
            break

    # --- Segment 4: Boole's rule, 5×z interval ---
    intv = 5.0 * z
    while fp_cum < 0.9 and (x_R - x_max) < 1000.0 * z:
        x_L = x_R
        x_R += intv
        fp_cum_prev = fp_cum
        fp_L = fp_R
        fp_R = eval_km(x_R)
        fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_boole(
            x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
            fp_win, fp_90, fp_55, fp_40)

    if fp_cum >= 0.9 and fp_90 == 0.0:
        fp_90 = x_R - intv * (fp_cum - 0.9) / max(fp_cum - fp_cum_prev, 1e-30)

    # --- Segment 5: extend if needed ---
    if (x_R < upwnd_dist) or (fp_90 == 0.0):
        intv = 2.0 * intv
        for _ in range(100):
            x_L = x_R
            x_R += intv
            fp_cum_prev = fp_cum
            fp_L = fp_R
            fp_R = eval_km(x_R)
            fp_cum, fp_win, fp_90, fp_55, fp_40 = _accum_boole(
                x_L, x_R, fp_L, fp_R, intv, fp_cum, fp_cum_prev,
                fp_win, fp_90, fp_55, fp_40)
            if fp_cum >= 0.9 and fp_90 == 0.0:
                fp_90 = x_R - intv * (fp_cum - 0.9) / max(fp_cum - fp_cum_prev, 1e-30)
            if x_R > upwnd_dist and fp_90 > x_max:
                break

        if fp_win == 0.0:
            fp_win = min(100.0 * fp_cum, 99.0) if fp_cum < 1.0 else 99.0

        if fp_cum < 0.9 and fp_90 == 0.0:
            fp_90 = x_R

    res.fetch_max = x_max
    res.fetch_90 = fp_90
    res.fetch_55 = fp_55
    res.fetch_40 = fp_40
    res.fp_dist_intrst = fp_win
    return res


# ---------------------------------------------------------------------------
#  Wind-direction sector helper
# ---------------------------------------------------------------------------

def _get_upwind_dist(wd_sonic: float, dist_intrst: Dict[str, float]) -> float:
    """Select the upwind distance of interest based on WD_SONIC sector.

    Follows the EasyFlux-DL convention exactly.
    """
    if wd_sonic <= 60.0:
        return dist_intrst['60_300']
    elif wd_sonic <= 170.0:
        return dist_intrst['60_170']
    elif wd_sonic < 190.0:
        return dist_intrst['170_190']
    elif wd_sonic < 300.0:
        return dist_intrst['190_300']
    else:
        return dist_intrst['60_300']


def wd_compass_to_sonic(wd_compass: float, sonic_azimuth: float) -> float:
    """Convert compass wind direction to WD_SONIC (CSAT coordinate system).

    Inverse of: WD = (sonic_azimuth - WD_SONIC + 360) % 360
    """
    return (sonic_azimuth - wd_compass + 360.0) % 360.0


# ---------------------------------------------------------------------------
#  Main dispatcher: calculate footprint for a single period
# ---------------------------------------------------------------------------

def calc_footprint(ustar: float, w_sigma: float, zl: float,
                   mo_length: float, ws_rslt: float, wd_compass: float,
                   cfg: SiteConfig) -> FootprintResult:
    """Calculate footprint characteristics for one averaging period.

    This replicates the footprint selection logic from the EasyFlux-DL
    main program (lines 330–361 of main_code_snippet.txt).

    Parameters
    ----------
    ustar : float      – friction velocity after freq corrections [m/s]
    w_sigma : float    – std dev of vertical velocity after rotation [m/s]
    zl : float         – z/L stability parameter
    mo_length : float  – Monin-Obukhov length [m]
    ws_rslt : float    – resultant wind speed [m/s]
    wd_compass : float – wind direction in compass convention [degrees]
    cfg : SiteConfig   – site configuration

    Returns
    -------
    FootprintResult
    """
    z = cfg.z

    if any(np.isnan(v) for v in [ustar, zl, mo_length]):
        return FootprintResult()

    # Convert compass WD to sonic coordinate for sector selection
    wd_sonic = wd_compass_to_sonic(wd_compass, cfg.sonic_azimuth)
    upwnd_dist = _get_upwind_dist(wd_sonic, cfg.dist_intrst)

    # Footprint model selection (Kljun criteria from EasyFlux-DL)
    if (zl >= -200.0) and (zl <= 1.0) and (ustar >= 0.2) and (z >= 1.0):
        # Kljun et al. (2004)
        return footprint_kljun(ustar, w_sigma, z, mo_length, cfg.z0,
                               upwnd_dist, cfg.n_int)
    else:
        # Kormann & Meixner (2001)
        return footprint_kormann_meixner(ustar, z, zl, ws_rslt,
                                         upwnd_dist, cfg.n_int)


# ---------------------------------------------------------------------------
#  Batch processing: apply to a DataFrame
# ---------------------------------------------------------------------------

def recalculate_fetch(df: pd.DataFrame, cfg: SiteConfig,
                      col_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Recalculate FETCH values for an entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (NaN-cleaned, i.e. -9999 already replaced).
    cfg : SiteConfig
        Site configuration with updated parameters.
    col_map : dict, optional
        Mapping from internal names to column names in `df`.
        Defaults assume AmeriFlux-style naming with _1_1_1 suffixes.

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns:
        FETCH_MAX_new, FETCH_90_new, FETCH_55_new, FETCH_40_new,
        FP_DIST_INTRST_new, FP_EQUATION_new
    """
    default_map = {
        'ustar':     'USTAR_1_1_1',
        'w_sigma':   'W_SIGMA_1_1_1',
        'zl':        'ZL_1_1_1',
        'mo_length': 'MO_LENGTH_1_1_1',
        'ws_rslt':   'WS_1_1_1',
        'wd':        'WD_1_1_1',
    }
    cm = {**default_map, **(col_map or {})}

    out = df.copy()
    results = []

    for idx, row in df.iterrows():
        r = calc_footprint(
            ustar=row[cm['ustar']],
            w_sigma=row[cm['w_sigma']],
            zl=row[cm['zl']],
            mo_length=row[cm['mo_length']],
            ws_rslt=row[cm['ws_rslt']],
            wd_compass=row[cm['wd']],
            cfg=cfg,
        )
        results.append(r)

    out['FETCH_MAX_new'] = [r.fetch_max for r in results]
    out['FETCH_90_new'] = [r.fetch_90 for r in results]
    out['FETCH_55_new'] = [r.fetch_55 for r in results]
    out['FETCH_40_new'] = [r.fetch_40 for r in results]
    out['FP_DIST_INTRST_new'] = [r.fp_dist_intrst for r in results]
    out['FP_EQUATION_new'] = [r.fp_equation for r in results]

    return out
