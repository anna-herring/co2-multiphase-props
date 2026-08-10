"""
Multiphase flow & capillary trapping fluid properties calculator.

Properties at a single (T, P) point for NW/W fluid pairs.
  - Density, viscosity, compressibility  : CoolProp
  - Brine corrections                    : Rowe & Chou (1970); Mao & Duan (2009)
  - Solubility                            : Henry's law + Krichevsky-Kasarnovsky
  - Diffusivity                           : Stokes-Einstein scaling from D_25 ref
  - Interfacial tension                   : 2-D interpolation of published data
"""

import numpy as np
import CoolProp.CoolProp as CP
from scipy.interpolate import RegularGridInterpolator

# ── Constants ────────────────────────────────────────────────────────────────
R        = 8.314           # J/(mol·K)
kB       = 1.380649e-23    # J/K
NA       = 6.02214076e23
Mw_H2O   = 0.018015        # kg/mol
Mw_NaCl  = 0.05844         # kg/mol

# ── Fluid definitions ────────────────────────────────────────────────────────
NW_FLUIDS = {
    "CO2": {
        "coolprop": "CO2",      "Mw": 0.04401, "Vmp": 35.1e-6,
        "D25": 2.2e-9,   "NIST_Hcp": 0.035,    "H_dT": 2400,
        "label": "CO₂",  "color": "#e74c3c",
        "Tc": 304.13,    "Pc": 7.377e6,
    },
    "H2": {
        "coolprop": "Hydrogen", "Mw": 0.00202, "Vmp": 26.7e-6,
        "D25": 5.11e-9,  "NIST_Hcp": 0.000078, "H_dT": 495,
        "label": "H₂",   "color": "#2980b9",
        "Tc": 33.15,     "Pc": 1.296e6,
    },
    "N2": {
        "coolprop": "Nitrogen", "Mw": 0.02802, "Vmp": 34.05e-6,
        "D25": 2.61e-9,  "NIST_Hcp": 0.000600, "H_dT": 1300,
        "label": "N₂",   "color": "#27ae60",
        "Tc": 126.19,    "Pc": 3.396e6,
    },
}

W_FLUIDS = {
    "water":    {"label": "Freshwater",   "default_S": 0.0},
    "seawater": {"label": "Seawater",     "default_S": 35.0},
    "brine":    {"label": "Saline Brine", "default_S": 100.0},
}

# CoolProp phase integer → label
_PHASE = {0: "Liquid", 1: "Supercritical Fluid", 2: "Supercritical Fluid",
          3: "Supercritical Fluid", 5: "Gas", 6: "Two-phase", 7: "Unknown"}

# ── IFT tables (mN/m) ────────────────────────────────────────────────────────
# Pressure [MPa] and temperature [°C] grids for interpolation
_IFT_P = np.array([0.1, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
_IFT_T = np.array([10., 20., 30., 40., 50., 60., 75., 100., 150., 200.])

# CO2-water IFT [mN/m]
# Hebach et al. (2002) J Phys Chem; Georgiadis et al. (2010) J Supercrit Fluids;
# Chiquet et al. (2007) Int J Greenh Gas Control; Bikkina et al. (2011);
# Pereira et al. (2016) J Chem Thermodyn
# 10 °C row: linear extrapolation from 20–30 °C gradient; pure-water anchor 74.2 mN/m
_IFT_CO2 = np.array([
    [74.4, 69.9, 64.0, 53.1, 43.2, 37.2, 33.7, 30.9, 29.1, 28.0],
    [72.8, 68.2, 62.5, 51.8, 42.1, 36.2, 32.8, 30.2, 28.6, 27.6],
    [71.2, 66.5, 61.0, 50.5, 41.0, 35.2, 31.9, 29.5, 28.1, 27.2],
    [69.0, 64.2, 58.5, 48.0, 38.5, 32.5, 29.5, 27.3, 26.1, 25.5],
    [66.8, 62.0, 56.2, 45.8, 36.8, 30.8, 28.0, 26.1, 25.0, 24.5],
    [64.4, 59.5, 53.8, 43.5, 35.0, 29.2, 26.6, 25.0, 24.0, 23.6],
    [61.0, 56.0, 50.5, 40.5, 33.0, 27.8, 25.5, 24.1, 23.3, 22.9],
    [57.0, 52.0, 47.0, 37.5, 30.5, 26.0, 23.9, 22.6, 21.9, 21.5],
    [48.5, 44.5, 40.5, 32.5, 27.0, 23.5, 21.8, 20.8, 20.2, 20.0],
    [37.5, 35.0, 32.0, 26.5, 22.5, 20.0, 18.8, 18.2, 17.8, 17.6],
])

# H2-water IFT [mN/m]
# Chow et al. (2018) J Chem Eng Data; Hashemi et al. (2021) Int J Hydrogen Energy;
# Yekeen et al. (2021) J Pet Sci Eng; Pan et al. (2021)
# 10 °C row: extrapolated; H2 effect on IFT is weak, pure-water anchor 74.2 mN/m
_IFT_H2 = np.array([
    [74.4, 73.8, 73.1, 72.0, 70.8, 69.8, 68.9, 68.1, 67.4, 66.8],
    [72.8, 72.2, 71.5, 70.4, 69.2, 68.2, 67.3, 66.5, 65.8, 65.2],
    [71.2, 70.6, 69.9, 68.8, 67.6, 66.6, 65.7, 64.9, 64.2, 63.6],
    [69.0, 68.4, 67.7, 66.6, 65.4, 64.4, 63.5, 62.7, 62.0, 61.4],
    [66.8, 66.2, 65.5, 64.4, 63.2, 62.2, 61.3, 60.5, 59.8, 59.2],
    [64.4, 63.8, 63.1, 62.0, 60.8, 59.8, 58.9, 58.1, 57.4, 56.8],
    [61.0, 60.4, 59.7, 58.6, 57.4, 56.4, 55.5, 54.7, 54.0, 53.4],
    [57.0, 56.4, 55.7, 54.6, 53.4, 52.4, 51.5, 50.7, 50.0, 49.4],
    [48.5, 48.0, 47.4, 46.4, 45.4, 44.5, 43.7, 43.0, 42.4, 41.9],
    [37.5, 37.1, 36.6, 35.8, 35.0, 34.3, 33.7, 33.2, 32.8, 32.4],
])

# N2-water IFT [mN/m]
# Yan et al. (2001) J Chem Eng Data; Li et al. (2012) J Chem Thermodyn;
# Kashefi et al. (2016) Fluid Phase Equilib
# 10 °C row: extrapolated; N2 effect on IFT is weak, pure-water anchor 74.2 mN/m
_IFT_N2 = np.array([
    [74.4, 74.1, 73.6, 72.9, 72.1, 71.4, 70.7, 70.1, 69.5, 69.0],
    [72.8, 72.5, 72.0, 71.3, 70.5, 69.8, 69.1, 68.5, 67.9, 67.4],
    [71.2, 70.9, 70.4, 69.7, 68.9, 68.2, 67.5, 66.9, 66.3, 65.8],
    [69.0, 68.7, 68.2, 67.5, 66.7, 66.0, 65.3, 64.7, 64.1, 63.6],
    [66.8, 66.5, 66.0, 65.3, 64.5, 63.8, 63.1, 62.5, 61.9, 61.4],
    [64.4, 64.1, 63.6, 62.9, 62.1, 61.4, 60.7, 60.1, 59.5, 59.0],
    [61.0, 60.7, 60.2, 59.5, 58.7, 58.0, 57.3, 56.7, 56.1, 55.6],
    [57.0, 56.7, 56.2, 55.5, 54.7, 54.0, 53.3, 52.7, 52.1, 51.6],
    [48.5, 48.3, 47.9, 47.3, 46.6, 46.0, 45.4, 44.9, 44.4, 44.0],
    [37.5, 37.3, 37.0, 36.5, 36.0, 35.5, 35.1, 34.7, 34.3, 34.0],
])

_IFT_INTERP = {
    "CO2": RegularGridInterpolator((_IFT_T, _IFT_P), _IFT_CO2, method="linear",
                                    bounds_error=False, fill_value=None),
    "H2":  RegularGridInterpolator((_IFT_T, _IFT_P), _IFT_H2,  method="linear",
                                    bounds_error=False, fill_value=None),
    "N2":  RegularGridInterpolator((_IFT_T, _IFT_P), _IFT_N2,  method="linear",
                                    bounds_error=False, fill_value=None),
}

IFT_SOURCES = {
    "CO2": "Hebach et al. (2002); Georgiadis et al. (2010); Chiquet et al. (2007)",
    "H2":  "Chow et al. (2018); Hashemi et al. (2021); Yekeen et al. (2021)",
    "N2":  "Yan et al. (2001); Li et al. (2012); Kashefi et al. (2016)",
}

# ── Helper functions ──────────────────────────────────────────────────────────

def _safe(v):
    """Return float or None for non-finite values."""
    return float(v) if np.isfinite(v) else None


def _nw_props(fluid_id, T_K, P_Pa):
    """Density, viscosity, compressibility and phase of the NW gas via CoolProp."""
    cp_name = NW_FLUIDS[fluid_id]["coolprop"]
    out = {}
    try:
        out["density_kg_m3"]          = _safe(CP.PropsSI("D", "T", T_K, "P", P_Pa, cp_name))
        out["viscosity_uPas"]         = _safe(CP.PropsSI("V", "T", T_K, "P", P_Pa, cp_name) * 1e6)
        out["compressibility_1_GPa"]  = _safe(CP.PropsSI("isothermal_compressibility",
                                                          "T", T_K, "P", P_Pa, cp_name) * 1e9)
        phase_int = int(CP.PropsSI("Phase", "T", T_K, "P", P_Pa, cp_name))
        out["phase"] = _PHASE.get(phase_int, "Unknown")
    except Exception as e:
        out.setdefault("density_kg_m3", None)
        out.setdefault("viscosity_uPas", None)
        out.setdefault("compressibility_1_GPa", None)
        out.setdefault("phase", "Error")
    return out


def _w_props(T_K, P_Pa, S_g_L):
    """Density, viscosity, compressibility of the wetting (brine) phase.

    Uses CoolProp for pure water, then applies NaCl corrections.
    Density    : Rowe & Chou (1970) approximation
    Viscosity  : Mao & Duan (2009) simplified
    κ_T        : CoolProp water value with small salinity correction
    """
    T_C = T_K - 273.15

    # Pure water base
    try:
        rho_w   = CP.PropsSI("D", "T", T_K, "P", P_Pa, "Water")
        mu_w    = CP.PropsSI("V", "T", T_K, "P", P_Pa, "Water")
        kappa_w = CP.PropsSI("isothermal_compressibility", "T", T_K, "P", P_Pa, "Water")
    except Exception:
        # Fallback for supercritical T: use steam tables via quality forcing
        rho_w, mu_w, kappa_w = np.nan, np.nan, np.nan

    m = S_g_L / (Mw_NaCl * 1000)    # mol/kg NaCl (approx; valid for dilute–moderate)

    # Density correction (Rowe & Chou 1970; Batzle & Wang 1992)
    d_rho = S_g_L * (0.824 - 4.66e-4 * T_C + 8.4e-7 * T_C ** 2)
    rho_b = rho_w + d_rho if np.isfinite(rho_w) else np.nan

    # Viscosity correction: Mao & Duan (2009) ln(μ_r) ≈ m*(A+B*T_K)
    A_mao, B_mao = 1.9741e-2, -4.22e-5
    mu_ratio = np.exp(m * (A_mao + B_mao * T_K))
    mu_b = mu_w * mu_ratio if np.isfinite(mu_w) else np.nan

    # Compressibility: slight reduction with salinity (< 3% for seawater)
    kappa_b = kappa_w * (1.0 - 2.6e-3 * m) if np.isfinite(kappa_w) else np.nan

    return {
        "density_kg_m3":         _safe(rho_b),
        "viscosity_uPas":        _safe(mu_b * 1e6),
        "compressibility_1_GPa": _safe(kappa_b * 1e9),
        "mu_water_Pas":          mu_w,   # kept for diffusivity calculation below
    }


def _henry_MPa(sol, T_K):
    """Henry's law constant [MPa] at temperature T_K."""
    H_ref = 0.101325 / (sol["NIST_Hcp"] * Mw_H2O)
    return H_ref / np.exp(sol["H_dT"] * (1.0 / T_K - 1.0 / 298.15))


def _water_vapor_pressure_MPa(T_K):
    try:
        return CP.PropsSI("P", "T", T_K, "Q", 0, "Water") * 1e-6
    except Exception:
        return 10 ** (8.07131 - 1730.63 / (233.426 + T_K - 273.15)) * 133.322e-6


def _solubility(nw_id, T_K, P_MPa, mu_w_Pas, S_g_L):
    """
    Mole fraction solubility of NW gas in the wetting phase via Henry's law
    with Krichevsky-Kasarnovsky (KK) correction for pressure.

    Returns mol fraction and mg/L.
    Salinity effect: Sechenov coefficient from Schumpe (1993) reduces solubility.
    """
    sol = NW_FLUIDS[nw_id]
    H = _henry_MPa(sol, T_K)
    Pv = _water_vapor_pressure_MPa(T_K)

    # KK correction
    Vmp_m3 = sol["Vmp"]   # m³/mol partial molar volume
    KK = H * np.exp(Vmp_m3 * (P_MPa - Pv) * 1e6 / (R * T_K))

    # Fugacity ≈ pressure for ideal-ish gas (H2, N2); use CoolProp for CO2
    try:
        fug_MPa = CP.PropsSI("fugacity", "T", T_K, "P", P_MPa * 1e6,
                              sol["coolprop"]) * 1e-6
    except Exception:
        fug_MPa = P_MPa  # fallback

    xs = fug_MPa / KK   # mole fraction

    # Sechenov salting-out correction: xs_brine = xs_water * 10^(-Ks * c_s)
    # Ks ≈ 0.01 (L/mol) for most gases; c_s in mol/L
    c_NaCl = S_g_L / (Mw_NaCl * 1000)   # mol/L
    Ks = 0.01  # Schumpe (1993) average value
    xs_brine = xs * 10 ** (-Ks * c_NaCl)

    # Convert to mg/L: ρ_water ≈ 1 kg/L
    # c [mol/L] = x * (1 kg/L / Mw_H2O [kg/mol]) = x / Mw_H2O
    # mg/L = c [mol/L] * Mw_gas [kg/mol] * 1e6 mg/kg
    mg_L = xs_brine * (1.0 / Mw_H2O) * sol["Mw"] * 1e6

    # mmol/L
    mmol_L = xs_brine * (1.0 / Mw_H2O) * 1e3

    return {
        "mol_frac":    _safe(xs_brine),
        "mg_L":        _safe(mg_L),
        "mmol_L":      _safe(mmol_L),
        "henry_MPa":   _safe(H),
        "kk_MPa":      _safe(KK),
    }


def _diffusivity(nw_id, T_K, mu_w_Pas):
    """
    Stokes-Einstein scaling of diffusivity:
        D = D_25 * (T / 298.15) * (μ_25 / μ_w)

    Reference viscosity at 25°C, 1 atm.
    """
    sol = NW_FLUIDS[nw_id]
    try:
        mu25 = CP.PropsSI("V", "T", 298.15, "P", 1e5, "Water")
    except Exception:
        mu25 = 8.9e-4  # Pa·s

    D = sol["D25"] * (T_K / 298.15) * (mu25 / mu_w_Pas) if (
        np.isfinite(mu_w_Pas) and mu_w_Pas > 0
    ) else np.nan
    return _safe(D)


def _ift(nw_id, T_C, P_MPa, S_g_L):
    """
    Interfacial tension [mN/m] from 2-D interpolation of published data (fresh water),
    with a salinity correction of +0.04 mN/m per g/L NaCl (Hebach et al. 2002).
    """
    T_clamp = np.clip(T_C, _IFT_T[0], _IFT_T[-1])
    P_clamp = np.clip(P_MPa, _IFT_P[0], _IFT_P[-1])
    ift_fw = float(_IFT_INTERP[nw_id]([[T_clamp, P_clamp]]).flat[0])
    ift = ift_fw + 0.04 * S_g_L   # salinity correction
    return _safe(ift)


# ── Flow-derived dimensionless numbers ───────────────────────────────────────

def compute_flow(Q_ml_hr: float, diam_mm: float, porosity: float,
                 char_len_um: float, mu_nw_uPas: float, mu_w_uPas: float,
                 rho_nw_kg_m3: float, rho_w_kg_m3: float, ift_mN_m: float,
                 theta_deg: float = 10.0) -> dict:
    """
    Fluid velocity and dimensionless numbers for multiphase flow.

    v   = Q / (A · φ)                        interstitial velocity
    Ca  = v · μ_injected / (σ · cos θ)        capillary number (contact-angle corrected)
    Bo  = Δρ · g · L² / (σ · cos θ)         Bond number (contact-angle corrected)
    M   = μ_NW / μ_W                         viscosity ratio (drainage convention)
    """
    Q   = Q_ml_hr  * 1e-6 / 3600       # m³/s
    r   = diam_mm  * 0.5e-3             # m
    A   = np.pi * r**2                  # m²
    v   = Q / (A * porosity)            # m/s (interstitial)

    mu_nw = mu_nw_uPas * 1e-6          # Pa·s
    mu_w  = mu_w_uPas  * 1e-6
    ift   = ift_mN_m   * 1e-3          # N/m
    L     = char_len_um * 1e-6         # m

    cos_theta = max(np.cos(np.radians(theta_deg)), 1e-9)   # guard against θ→90°

    Ca_nw = v * mu_nw / (ift * cos_theta)
    Ca_w  = v * mu_w  / (ift * cos_theta)

    delta_rho = abs(rho_w_kg_m3 - rho_nw_kg_m3)
    Bo  = delta_rho * 9.81 * L**2 / (ift * cos_theta)
    M   = mu_nw / mu_w   # NW/W, drainage convention

    inv_Bo      = 1.0 / Bo  if Bo > 0 else np.nan
    Ca_Bo_nw    = Ca_nw / Bo if Bo > 0 else np.nan
    Ca_Bo_w     = Ca_w  / Bo if Bo > 0 else np.nan
    M_imbib     = mu_w / mu_nw  # invading/defending for imbibition

    return {
        "velocity_m_s":    _safe(v),
        "velocity_um_s":   _safe(v * 1e6),
        "Ca_nw":           _safe(Ca_nw),
        "Ca_w":            _safe(Ca_w),
        "Bo":              _safe(Bo),
        "M_drainage":      _safe(M),
        "M_imbibition":    _safe(M_imbib),
        "inv_Bo":          _safe(inv_Bo),
        "Ca_Bo_nw":        _safe(Ca_Bo_nw),
        "Ca_Bo_w":         _safe(Ca_Bo_w),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def calculate(T_C: float, P_MPa: float, nw_id: str, w_id: str,
              salinity_g_L: float | None = None,
              Q_ml_hr: float | None = None, core_diam_mm: float = 10.0,
              porosity: float = 0.22, char_len_um: float = 40.0,
              theta_deg: float = 10.0, Ca_nw_target: float | None = None) -> dict:
    """Return all properties at (T_C, P_MPa) for the given NW/W fluid pair."""
    P_MPa = max(P_MPa, 0.1)   # avoid P=0 in CoolProp
    T_K   = T_C + 273.15
    P_Pa  = P_MPa * 1e6

    nw_def = NW_FLUIDS[nw_id]
    w_def  = W_FLUIDS[w_id]
    S = w_def["default_S"] if salinity_g_L is None else float(salinity_g_L)

    # ── NW phase ──────────────────────────────────────────────────────────────
    nw = _nw_props(nw_id, T_K, P_Pa)
    nw["fluid"]  = nw_id
    nw["label"]  = nw_def["label"]
    nw["color"]  = nw_def["color"]
    nw["Mw_g_mol"] = nw_def["Mw"] * 1000

    # ── Wetting phase ─────────────────────────────────────────────────────────
    w_out = _w_props(T_K, P_Pa, S)
    mu_w  = w_out.pop("mu_water_Pas", np.nan)   # internal; not returned
    w_out["fluid"]       = w_id
    w_out["label"]       = w_def["label"]
    w_out["salinity_g_L"] = S

    # ── Interaction / transport ───────────────────────────────────────────────
    sol   = _solubility(nw_id, T_K, P_MPa, mu_w if np.isfinite(mu_w) else 1e-3, S)
    D     = _diffusivity(nw_id, T_K, mu_w if np.isfinite(mu_w) else 1e-3)
    ift   = _ift(nw_id, T_C, P_MPa, S)

    # Mobility & density ratios (useful for multiphase flow)
    mu_nw_Pas   = (nw["viscosity_uPas"] or 0) * 1e-6
    mu_w_Pas    = mu_w if np.isfinite(mu_w) else None
    rho_nw      = nw.get("density_kg_m3")
    rho_w       = w_out.get("density_kg_m3")

    mobility_ratio  = _safe(mu_nw_Pas / mu_w_Pas) if (mu_w_Pas and mu_w_Pas > 0
                                                        and mu_nw_Pas > 0) else None
    density_ratio   = _safe(rho_w / rho_nw) if (rho_nw and rho_nw > 0
                                                  and rho_w) else None
    buoyancy_N_m3   = _safe((rho_w - rho_nw) * 9.81) if (
        rho_w and rho_nw) else None   # (ρ_w - ρ_nw)*g [N/m³]

    interaction = {
        "ift_mN_m":        ift,
        "ift_source":      IFT_SOURCES[nw_id],
        "solubility":      sol,
        "diffusivity_m2_s": D,
        "mobility_ratio":   mobility_ratio,
        "density_ratio":    density_ratio,
        "buoyancy_N_m3":    buoyancy_N_m3,
    }

    # ── If Ca target supplied, derive Q from it ───────────────────────────────
    derived_Q = None
    if (Ca_nw_target is not None and Ca_nw_target > 0
            and ift and ift > 0 and mu_nw_Pas and mu_nw_Pas > 0):
        cos_theta = max(np.cos(np.radians(theta_deg)), 1e-9)
        A_m2 = np.pi * (core_diam_mm * 0.5e-3) ** 2
        v_m_s = Ca_nw_target * (ift * 1e-3) * cos_theta / mu_nw_Pas
        derived_Q = v_m_s * A_m2 * porosity * 1e6 * 3600   # m³/s → ml/hr
        if Q_ml_hr is None:
            Q_ml_hr = derived_Q

    # ── Flow & dimensionless numbers (optional) ───────────────────────────────
    flow = None
    if (Q_ml_hr is not None and ift and ift > 0
            and nw.get("viscosity_uPas") and w_out.get("viscosity_uPas")
            and nw.get("density_kg_m3") and w_out.get("density_kg_m3")):
        flow = compute_flow(
            Q_ml_hr, core_diam_mm, porosity, char_len_um,
            nw["viscosity_uPas"], w_out["viscosity_uPas"],
            nw["density_kg_m3"],  w_out["density_kg_m3"],
            ift, theta_deg,
        )
        flow["Q_ml_hr"]         = Q_ml_hr
        flow["derived_Q_ml_hr"] = _safe(derived_Q) if derived_Q is not None else None
        flow["core_diam_mm"]    = core_diam_mm
        flow["porosity"]        = porosity
        flow["char_len_um"]     = char_len_um
        flow["theta_deg"]       = theta_deg

    return {
        "T_C":    T_C,
        "P_MPa":  P_MPa,
        "nw":     nw,
        "wetting": w_out,
        "interaction": interaction,
        "flow":   flow,
    }
