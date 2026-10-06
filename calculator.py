"""
Multiphase flow & capillary trapping fluid properties calculator.

Properties at a single (T, P) point for NW/W fluid pairs, plus property
surfaces over a (T, P) grid (see surface_grid).
  - Density, viscosity, speed of sound, isothermal/isentropic
    compressibility, c_p, c_v, gamma, thermal expansivity, thermal
    conductivity, Joule-Thomson coefficient, Z   : CoolProp
  - Brine corrections                    : Rowe & Chou (1970); Mao & Duan (2009)
  - CO2 solubility (water + NaCl brine)   : Duan & Sun (2006) Marine Chemistry,
        with fugacity from the Duan, Moller & Weare (1992) BWR EOS
  - CO2 solubility (KI), H2/N2            : Henry's law + Krichevsky-Kasarnovsky
  - Diffusivity                           : Stokes-Einstein scaling from D_25 ref
  - Interfacial tension                   : 2-D interpolation of published data

WARNING: CoolProp has NO "fugacity" output key, and AbstractState.fugacity(i)
fails for pure fluids. Both failures are silent if caught, leaving an ideal-gas
value (f = P) that is ~2.7x wrong for CO2 at 50 C / 10 MPa. Get the fugacity
coefficient from residual properties instead -- see _fugacity_MPa below.
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
Mw_KI    = 0.16601         # kg/mol

# ── Fluid definitions ────────────────────────────────────────────────────────
NW_FLUIDS = {
    "CO2": {
        "coolprop": "CO2",      "Mw": 0.04401, "Vmp": 35.1e-6,
        "D25": 2.2e-9,   "NIST_Hcp": 0.035,    "H_dT": 2400,
        "label": "CO₂",  "color": "#e74c3c",
        "Tc": 304.13,    "Pc": 7.377e6,
    },
    # H2 and N2 were removed 2026-09-29 (CO2-only tool). Their IFT tables,
    # Henry constants and solubility path are all still present below, so
    # restoring them means re-adding their entries here, to _IFT_INTERP and
    # to IFT_SOURCES -- nothing else was deleted.
}

W_FLUIDS = {
    "water":    {"label": "Freshwater",   "default_S": 0.0,   "default_ki": 0.0},
    "seawater": {"label": "Seawater",     "default_S": 35.0,  "default_ki": 0.0},
    "brine":    {"label": "Saline Brine", "default_S": 100.0, "default_ki": 0.0},
    "ki_brine": {"label": "KI Brine",     "default_S": 0.0,   "default_ki": 0.5},
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
}

IFT_SOURCES = {
    "CO2": "Hebach et al. (2002); Georgiadis et al. (2010); Chiquet et al. (2007)",
}

# ── Helper functions ──────────────────────────────────────────────────────────

def _safe(v):
    """Return float or None for non-finite values."""
    return float(v) if np.isfinite(v) else None


# ── Thermodynamic property block ─────────────────────────────────────────────
# key -> (display label, unit, use a LOG colour scale on the surface plot)
# The log flags come from the measured dynamic range over the app's (T,P)
# window for CO2: kappa_T spans ~2100x and Cp ~50x (both diverge at the
# critical point), whereas the speed of sound spans only ~3.8x.
PROP_META = {
    "density_kg_m3":            ("Density",                    "kg/m³",   False),
    "viscosity_uPas":           ("Dynamic viscosity",          "µPa·s",   True),
    "speed_sound_m_s":          ("Speed of sound",             "m/s",     False),
    "compressibility_1_GPa":    ("Isothermal compressibility",  "GPa⁻¹",  True),
    "expansivity_1_K":          ("Thermal expansivity",         "1/K",    True),
    "conductivity_W_mK":        ("Thermal conductivity",        "W/m·K",  False),
    "joule_thomson_K_MPa":      ("Joule–Thomson coefficient",   "K/MPa",  False),
    "Z":                        ("Compressibility factor Z",    "—",      False),
}

# Still COMPUTED, but no longer displayed or offered as a surface (removed
# from the UI 2026-10-06 as surplus). gamma cannot simply be dropped: the
# brine sound speed comes from c = sqrt(gamma/(rho*kappa_T)) and kappa_s from
# kappa_T/gamma, so _w_props depends on it. They stay in the returned dict.
_INTERNAL_PROPS = ("compressibility_s_1_GPa", "cp_J_kgK", "cv_J_kgK", "gamma")

# Order the property rows appear in the UI cards
PROP_ORDER = list(PROP_META.keys())


def _coolprop_props(cp_name, T_K, P_Pa, state=None):
    """
    Every thermodynamic property at (T,P) from ONE AbstractState update.

    Using a single AbstractState keeps all properties on the SAME state point
    and is far faster than repeated PropsSI calls, which matters for the
    surface grids (thousands of points).

    Pass `state` to reuse an AbstractState across a grid.

    NOTE: PropsSI("speed_sound", ...) is NOT a valid CoolProp key -- the valid
    one is "A". That is the same silent-failure trap as "fugacity" described in
    the module docstring. AbstractState.speed_sound() is unambiguous, so all
    properties here come from the AbstractState API rather than output strings.
    """
    out = {k: None for k in (*PROP_META, *_INTERNAL_PROPS)}
    out["phase"] = "Error"
    try:
        AS = state if state is not None else CP.AbstractState("HEOS", cp_name)
        AS.update(CP.PT_INPUTS, P_Pa, T_K)

        rho = AS.rhomass()
        cp_ = AS.cpmass()
        cv_ = AS.cvmass()
        kT  = AS.isothermal_compressibility()
        gam = cp_ / cv_ if cv_ else np.nan

        out["density_kg_m3"]           = _safe(rho)
        out["speed_sound_m_s"]         = _safe(AS.speed_sound())
        out["compressibility_1_GPa"]   = _safe(kT * 1e9)
        out["compressibility_s_1_GPa"] = _safe(kT / gam * 1e9)   # κ_s = κ_T/γ
        out["cp_J_kgK"]                = _safe(cp_)
        out["cv_J_kgK"]                = _safe(cv_)
        out["gamma"]                   = _safe(gam)
        out["expansivity_1_K"]         = _safe(AS.isobaric_expansion_coefficient())
        out["Z"]                       = _safe(AS.compressibility_factor())
        out["phase"]                   = _PHASE.get(AS.phase(), "Unknown")

        # Transport properties are unavailable in a few regions for some fluids
        try:
            out["viscosity_uPas"]    = _safe(AS.viscosity() * 1e6)
            out["conductivity_W_mK"] = _safe(AS.conductivity())
        except Exception:
            pass
        try:   # μ_JT = (∂T/∂P)_H, Pa → MPa
            out["joule_thomson_K_MPa"] = _safe(
                AS.first_partial_deriv(CP.iT, CP.iP, CP.iHmass) * 1e6)
        except Exception:
            pass
    except Exception:
        pass
    return out


def _nw_props(fluid_id, T_K, P_Pa):
    """Full thermodynamic property set for the NW fluid via CoolProp."""
    return _coolprop_props(NW_FLUIDS[fluid_id]["coolprop"], T_K, P_Pa)


# Isothermal compressibility of NaCl brine, relative to pure water:
#     κ_T(m) / κ_T(water) = exp(-KAPPA_NACL * m),   m in mol/kg
# Anchored on measured NaCl(aq) at 1.0 mol/kg, 25 °C (c = 1552 m/s,
# ρ = 1037.5 kg/m³): the identity c² = γ/(ρ·κ_T) gives κ_T = 4.05e-10 Pa⁻¹
# against 4.52e-10 for pure water, i.e. -10.6% per mol/kg. The exponential
# form (rather than linear) keeps κ_T positive and reproduces near-saturated
# brine: at 6 mol/kg it gives c ≈ 1918 m/s, consistent with measurement.
# Underlying data: Millero et al., PVT properties of concentrated electrolytes
# (J. Solution Chem.). REPLACES a previous linear coefficient of 2.6e-3 per
# mol/kg, which was ~43x too weak and made brine essentially as compressible
# as pure water.
KAPPA_NACL = 0.112

# Properties that carry NO salt correction and are reported as pure water.
# Anything listed here must be flagged in the UI.
# Only DISPLAYED properties are listed — c_p, c_v and gamma are pure water too
# but are no longer shown, so flagging them would be invisible noise.
_W_PURE_WATER_BASE = ["expansivity_1_K", "conductivity_W_mK",
                      "joule_thomson_K_MPa", "Z"]


def _water_is_liquid(T_K, P_Pa):
    """
    True when water at (T,P) is LIQUID -- the only state in which the brine
    correlations below are defined.

    Compares against the saturation pressure rather than reading CoolProp's
    phase label, because above P_c (22.06 MPa) CoolProp reports compressed
    liquid as "supercritical_liquid", which a label test would wrongly reject.
    """
    try:
        if T_K >= CP.PropsSI("Tcrit", "Water"):
            return False                      # beyond water's critical point
        return P_Pa >= CP.PropsSI("P", "T", T_K, "Q", 0, "Water")
    except Exception:
        return True    # fail open: an EOS hiccup shouldn't blank a valid state


def _w_props(T_K, P_Pa, S_g_L, ki_mol_L=0.0):
    """Thermodynamic properties of the wetting (brine) phase.

    Salt-corrected:
      NaCl: Rowe & Chou (1970) density; Mao & Duan (2009) viscosity;
            isothermal compressibility via KAPPA_NACL above; speed of sound
            derived from those two by the exact identity c² = γ/(ρ·κ_T).
      KI  : CRC Handbook density (~128 kg/m3 per mol/L);
            Jones-Dole viscosity (negative B coefficient -> slight decrease).

    NOT salt-corrected (pure-water values, flagged via "uncorrected_props"):
      c_p, c_v, γ, expansivity, thermal conductivity, Joule-Thomson, Z for
      every brine; PLUS compressibility and speed of sound for KI, for which
      no compressibility coefficient could be sourced. Reporting a pure-water
      value and saying so is better than a KI sound speed built from corrected
      density but uncorrected compressibility, which would trend the wrong way.
    """
    T_C = T_K - 273.15

    # GUARD: every correlation below (Rowe & Chou density, Mao & Duan
    # viscosity, KAPPA_NACL compressibility, Jones-Dole) is a LIQUID
    # correlation. Below the saturation pressure CoolProp hands back steam and
    # these would be applied to it regardless -- at 200 C / 0.1 MPa that turned
    # 0.46 kg/m3 of vapour into a "brine" at 45.1 kg/m3 with a 57 m/s sound
    # speed. Report nothing and say why, rather than a plausible-looking wrong
    # number. This also makes `flow` None downstream, so no Ca or Bo is
    # produced from a non-liquid wetting phase.
    if not _water_is_liquid(T_K, P_Pa):
        out = {k: None for k in (*PROP_META, *_INTERNAL_PROPS)}
        out["water_not_liquid"]  = True
        out["uncorrected_props"] = []
        out["mu_water_Pas"]      = np.nan
        return out

    # Pure water base: full property set, single state point
    w = _coolprop_props("Water", T_K, P_Pa)
    rho_w   = w["density_kg_m3"]
    mu_w    = (w["viscosity_uPas"] / 1e6) if w["viscosity_uPas"] else np.nan
    kappa_w = (w["compressibility_1_GPa"] / 1e9) if w["compressibility_1_GPa"] else np.nan
    gamma_w = w["gamma"] if w["gamma"] else np.nan
    if rho_w is None:
        rho_w = np.nan

    uncorrected = []

    if ki_mol_L > 0:
        # KI density: CRC Handbook data, ~128 kg/m3 per mol/L at 25 C
        d_rho = ki_mol_L * (128.0 - 0.04 * T_C)
        rho_b = rho_w + d_rho if np.isfinite(rho_w) else np.nan

        # KI viscosity: Jones-Dole; negative B -> decreases viscosity
        ln_mu_r = ki_mol_L * (-0.090 + 1.5e-4 * (T_K - 298.15))
        mu_b = mu_w * np.exp(ln_mu_r) if np.isfinite(mu_w) else np.nan

        # No sourced KI compressibility coefficient -> leave as pure water
        kappa_b = kappa_w
        uncorrected = _W_PURE_WATER_BASE + [
            "compressibility_1_GPa", "compressibility_s_1_GPa", "speed_sound_m_s"]
    else:
        m = S_g_L / (Mw_NaCl * 1000)    # mol/kg NaCl

        # Density: Rowe & Chou (1970); Batzle & Wang (1992)
        d_rho = S_g_L * (0.824 - 4.66e-4 * T_C + 8.4e-7 * T_C ** 2)
        rho_b = rho_w + d_rho if np.isfinite(rho_w) else np.nan

        # Viscosity: Mao & Duan (2009)
        A_mao, B_mao = 1.9741e-2, -4.22e-5
        mu_b = mu_w * np.exp(m * (A_mao + B_mao * T_K)) if np.isfinite(mu_w) else np.nan

        kappa_b = kappa_w * np.exp(-KAPPA_NACL * m) if np.isfinite(kappa_w) else np.nan
        if m > 0:
            uncorrected = list(_W_PURE_WATER_BASE)

    # Speed of sound from the EXACT identity c² = γ/(ρ·κ_T), so it stays
    # consistent with the density and compressibility just derived.
    #
    # Only do this when κ_T is ITSELF salt-corrected. Mixing a corrected
    # density with a pure-water κ_T (the KI case) drives c BELOW pure water,
    # which is the wrong direction -- adding salt raises the sound speed. When
    # κ_T is uncorrected we therefore report the pure-water c and flag it,
    # rather than a half-corrected number that trends backwards.
    if "speed_sound_m_s" in uncorrected:
        c_b      = w["speed_sound_m_s"]
        kappa_sb = (kappa_w / gamma_w) if np.isfinite(gamma_w) else np.nan
    elif np.isfinite(kappa_b) and np.isfinite(rho_b) and np.isfinite(gamma_w) \
            and rho_b > 0 and kappa_b > 0:
        c_b      = np.sqrt(gamma_w / (rho_b * kappa_b))
        kappa_sb = kappa_b / gamma_w
    else:
        c_b, kappa_sb = np.nan, np.nan

    out = dict(w)                       # pure-water values for everything else
    out.pop("phase", None)              # the wetting phase is always liquid here
    out["density_kg_m3"]           = _safe(rho_b)
    out["viscosity_uPas"]          = _safe(mu_b * 1e6)
    out["compressibility_1_GPa"]   = _safe(kappa_b * 1e9)
    out["compressibility_s_1_GPa"] = _safe(kappa_sb * 1e9)
    out["speed_sound_m_s"]         = _safe(c_b)
    out["uncorrected_props"]       = uncorrected
    out["water_not_liquid"]        = False
    out["mu_water_Pas"]            = mu_w   # internal, for diffusivity below
    return out


def _henry_MPa(sol, T_K):
    """Henry's law constant [MPa] at temperature T_K."""
    H_ref = 0.101325 / (sol["NIST_Hcp"] * Mw_H2O)
    return H_ref / np.exp(sol["H_dT"] * (1.0 / T_K - 1.0 / 298.15))


def _water_vapor_pressure_MPa(T_K):
    try:
        return CP.PropsSI("P", "T", T_K, "Q", 0, "Water") * 1e-6
    except Exception:
        return 10 ** (8.07131 - 1730.63 / (233.426 + T_K - 273.15)) * 133.322e-6


def _fugacity_MPa(coolprop_name, T_K, P_MPa):
    """
    Fugacity [MPa] via residual Gibbs energy from CoolProp.
    ln φ = (h_res - T·s_res) / (R·T);  f = φ·P
    Falls back to ideal gas if CoolProp raises.
    """
    try:
        AS = CP.AbstractState("HEOS", coolprop_name)
        AS.update(CP.PT_INPUTS, P_MPa * 1e6, T_K)
        h_res = AS.hmolar_residual()   # J/mol
        s_res = AS.smolar_residual()   # J/(mol·K)
        phi = np.exp((h_res - T_K * s_res) / (R * T_K))
        return phi * P_MPa
    except Exception:
        return P_MPa


_DUAN_A = (8.99288497e-2, -4.94783127e-1, 4.77922245e-2, 1.03808883e-2,
           -2.82516861e-2, 9.49887563e-2, 5.20600880e-4, -2.93540971e-4,
           -1.77265112e-3, -2.51101973e-5, 8.93353441e-5, 7.88998563e-5,
           -1.66727022e-2, 1.3980, 2.96000000e-2)


def _co2_fugacity_coeff_duan(T_K, P_MPa):
    """
    CO2 fugacity coefficient from the Duan, Moller & Weare (1992) BWR-type EOS,
    as used inside Duan & Sun (2003, 2006). Must be used instead of Span &
    Wagner: the mu0 parameterisation is calibrated against THIS EOS.

    Reduced variables use the Duan reference constants Tc=304.1282 K,
    Pc=73.773 bar. Returns phi (dimensionless).
    """
    a = _DUAN_A
    Tc, Pc_bar = 304.1282, 73.773
    Tr = T_K / Tc
    Pr = (P_MPa * 10.0) / Pc_bar

    B = a[0] + a[1] / Tr**2 + a[2] / Tr**3
    C = a[3] + a[4] / Tr**2 + a[5] / Tr**3
    D = a[6] + a[7] / Tr**2 + a[8] / Tr**3
    E = a[9] + a[10] / Tr**2 + a[11] / Tr**3
    F = a[12] / Tr**3
    alpha, beta = a[13], a[14]

    # Solve Z from  Z = 1 + B/Vr + C/Vr^2 + D/Vr^4 + E/Vr^5
    #                 + (F/Vr^2)(beta + alpha/Vr^2) exp(-alpha/Vr^2)
    # with Vr = Z*Tr/Pr, by fixed-point iteration on Vr.
    Vr = Tr / Pr          # ideal-gas start
    for _ in range(200):
        Z = (1.0 + B / Vr + C / Vr**2 + D / Vr**4 + E / Vr**5
             + (F / Vr**2) * (beta + alpha / Vr**2) * np.exp(-alpha / Vr**2))
        Vr_new = Z * Tr / Pr
        if abs(Vr_new - Vr) < 1e-12:
            Vr = Vr_new
            break
        Vr = 0.5 * Vr + 0.5 * Vr_new     # damped, keeps liquid root stable

    Z = Pr * Vr / Tr
    ln_phi = (Z - 1.0 - np.log(Z)
              + B / Vr + C / (2.0 * Vr**2) + D / (4.0 * Vr**4) + E / (5.0 * Vr**5)
              + (F / (2.0 * alpha)) * (beta + 1.0
                                       - (beta + 1.0 + alpha / Vr**2)
                                       * np.exp(-alpha / Vr**2)))
    return np.exp(ln_phi)


def _co2_solubility_duan_sun(T_K, P_MPa, m_NaCl):
    """
    CO2 solubility in NaCl brine, Duan & Sun (2006) Marine Chemistry 98, 131-139.
    doi:10.1016/j.marchem.2005.09.001

    ln(m_CO2) = ln(y_CO2 * phi * P) - mu0/RT - 2*lam*m_Na - zeta*m_Na*m_Cl

    Fugacity comes from the Duan (1992) EOS (NOT Span & Wagner) because the
    mu0 polynomial was regressed against it. Returns molality [mol/kg water].
    Valid: 273-533 K, 0-200 MPa, 0-4.5 mol/kg NaCl.
    """
    P_bar = P_MPa * 10.0

    # Standard chemical potential mu0/RT of CO2(aq)  (Table 2, Duan & Sun 2003)
    c = (28.9447706, -0.0354581768, -4770.67077, 1.02782768e-5, 33.8126098,
         9.04037140e-3, -1.14934031e-3, -0.307405726, -0.0907301486,
         9.32713393e-4, 0.0)
    mu0 = (c[0] + c[1] * T_K + c[2] / T_K + c[3] * T_K**2 + c[4] / (630.0 - T_K)
           + c[5] * P_bar + c[6] * P_bar * np.log(T_K) + c[7] * P_bar / T_K
           + c[8] * P_bar / (630.0 - T_K)
           + c[9] * P_bar**2 / (630.0 - T_K)**2)

    lam = (-0.411370585 + 6.07632013e-4 * T_K + 97.5347708 / T_K
           - 0.0237622469 * P_bar / T_K + 0.0170656236 * P_bar / (630.0 - T_K)
           + 1.41335834e-5 * T_K * np.log(P_bar))

    zeta = (3.36389723e-4 - 1.98298980e-5 * T_K
            + 2.12220830e-3 * P_bar / T_K - 5.24873303e-3 * P_bar / (630.0 - T_K))

    # Mole fraction of CO2 in the vapour phase (water vapour dilutes it)
    Pv_bar = _water_vapor_pressure_MPa(T_K) * 10.0
    y_co2 = max((P_bar - Pv_bar) / P_bar, 1e-6)

    phi = _co2_fugacity_coeff_duan(T_K, P_MPa)

    ln_m = (np.log(y_co2 * phi * P_bar) - mu0
            - 2.0 * lam * m_NaCl - zeta * m_NaCl**2)
    return np.exp(ln_m)


def _solubility(nw_id, T_K, P_MPa, mu_w_Pas, S_g_L, ki_mol_L=0.0):
    """
    Mole fraction solubility of NW gas in the wetting phase.

    CO2 (water, NaCl brine, KI brine): Duan & Sun (2006) Marine Chemistry,
      doi:10.1016/j.marchem.2005.09.001. NaCl salting-out is intrinsic to that
      model; KI is applied as a Sechenov factor ON TOP of the Duan & Sun
      pure-water baseline, so freshwater and dilute KI agree in the limit.
    H2 / N2: Henry's law + Krichevsky-Kasarnovsky pressure correction
      + Sechenov salting-out (Schumpe 1993).
    Fugacity from CoolProp residual Gibbs energy (not ideal gas) -- see the
    warning in the module docstring.
    """
    sol = NW_FLUIDS[nw_id]

    H    = _henry_MPa(sol, T_K)
    Pv   = _water_vapor_pressure_MPa(T_K)
    KK   = H * np.exp(sol["Vmp"] * (P_MPa - Pv) * 1e6 / (R * T_K))
    fug_MPa = _fugacity_MPa(sol["coolprop"], T_K, P_MPa)
    xs_pure = fug_MPa / KK          # pure-water mole fraction (Henry+KK)

    if nw_id == "CO2":
        # ── Duan & Sun (2006): NaCl enters the model directly; KI rides on top
        #    of the same pure-water baseline as a Sechenov factor, so that
        #    KI -> 0 reproduces freshwater instead of jumping to Henry's law.
        m_NaCl = 0.0 if ki_mol_L > 0 else S_g_L / (Mw_NaCl * 1000)
        try:
            m_co2 = _co2_solubility_duan_sun(T_K, P_MPa, m_NaCl)
            xs_brine = m_co2 * Mw_H2O / (1.0 + m_co2 * Mw_H2O)
        except Exception:
            xs_brine = xs_pure          # fall back to Henry's law on error
        if ki_mol_L > 0:
            Ks_ki = 0.006   # KI Sechenov constant (Schumpe 1993)
            xs_brine *= 10 ** (-Ks_ki * ki_mol_L)
    elif ki_mol_L > 0:
        Ks_ki = 0.006
        xs_brine = xs_pure * 10 ** (-Ks_ki * ki_mol_L)
    else:
        m_NaCl = S_g_L / (Mw_NaCl * 1000)
        Ks = 0.01       # Schumpe (1993)
        xs_brine = xs_pure * 10 ** (-Ks * m_NaCl)

    mg_L   = xs_brine * (1.0 / Mw_H2O) * sol["Mw"] * 1e6
    mmol_L = xs_brine * (1.0 / Mw_H2O) * 1e3

    return {
        "mol_frac":  _safe(xs_brine),
        "mg_L":      _safe(mg_L),
        "mmol_L":    _safe(mmol_L),
        "henry_MPa": _safe(H),
        "kk_MPa":    _safe(KK),
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


def _ift(nw_id, T_C, P_MPa, S_g_L, ki_mol_L=0.0):
    """
    Interfacial tension [mN/m] from 2-D interpolation of published data (fresh water).
    NaCl correction: +0.04 mN/m per g/L (Hebach et al. 2002).
    KI correction: -0.8 mN/m per mol/L (Weissenborn & Pugh 1996; KI is a
    surface-active chaotrope that slightly decreases surface tension).
    """
    # NOTE: clamped, NOT extrapolated. Outside the table the value is held at
    # the nearest edge, so it stops being a measurement -- callers should use
    # ift_out_of_range() and say so. This matters now that the pressure limit
    # reaches 40 MPa (for the deeper CCS sites) while the table stops at 20.
    T_clamp = np.clip(T_C, _IFT_T[0], _IFT_T[-1])
    P_clamp = np.clip(P_MPa, _IFT_P[0], _IFT_P[-1])
    ift_fw = float(_IFT_INTERP[nw_id]([[T_clamp, P_clamp]]).flat[0])
    if ki_mol_L > 0:
        ift = ift_fw - 0.8 * ki_mol_L
    else:
        ift = ift_fw + 0.04 * S_g_L
    return _safe(ift)


def ift_out_of_range(T_C, P_MPa):
    """
    True when (T, P) falls outside the measured IFT table, i.e. _ift() has
    clamped to an edge value and the returned IFT is no longer interpolated
    from data. Table covers 10-200 °C and 0.1-20 MPa.
    """
    return bool(T_C < _IFT_T[0] or T_C > _IFT_T[-1]
                or P_MPa < _IFT_P[0] or P_MPa > _IFT_P[-1])


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


# ── Property surfaces over (T, P) ────────────────────────────────────────────

# Fluids that can be plotted as a surface. These are PURE CoolProp fluids --
# brine salt corrections are deliberately not applied here, because they touch
# only 5 of the 12 properties and a mixed surface would be mislabelled.
SURFACE_FLUIDS = {
    "CO2":   {"coolprop": "CO2",   "label": "CO₂",          "color": "#e74c3c"},
    "water": {"coolprop": "Water", "label": "Water (pure)", "color": "#0369a1"},
}


def surface_grid(fluid_key, T_min_C, T_max_C, P_min_MPa, P_max_MPa,
                 n_T=70, n_P=70):
    """
    Thermodynamic properties on a (T, P) grid, for 3-D surface plotting.

    Returns z-matrices indexed [i][j] = property at P[i], T[j], which is the
    orientation Plotly's surface trace expects for x=T, y=P.

    In P-T space the two-phase region of a pure fluid is a CURVE, not an area,
    so every grid point off the saturation line is single-phase and CoolProp
    resolves it. Crossing that line the properties jump discontinuously; the
    surface shows a genuine cliff there, and the saturation curve is returned
    separately so it can be drawn.
    """
    if fluid_key not in SURFACE_FLUIDS:
        raise ValueError(f"Unknown surface fluid: {fluid_key}")
    spec    = SURFACE_FLUIDS[fluid_key]
    cp_name = spec["coolprop"]

    T_C   = np.linspace(float(T_min_C), float(T_max_C), int(n_T))
    P_MPa = np.linspace(float(P_min_MPa), float(P_max_MPa), int(n_P))
    # CoolProp cannot evaluate P = 0
    P_MPa = np.clip(P_MPa, 1e-3, None)

    keys  = list(PROP_META.keys())
    props = {k: [[None] * len(T_C) for _ in range(len(P_MPa))] for k in keys}

    AS = CP.AbstractState("HEOS", cp_name)
    for i, P in enumerate(P_MPa):
        for j, T in enumerate(T_C):
            vals = _coolprop_props(cp_name, T + 273.15, P * 1e6, state=AS)
            for k in keys:
                props[k][i][j] = vals[k]

    # Critical point
    try:
        Tc_C   = CP.PropsSI("Tcrit", cp_name) - 273.15
        Pc_MPa = CP.PropsSI("Pcrit", cp_name) / 1e6
    except Exception:
        Tc_C, Pc_MPa = None, None
    in_window = (Tc_C is not None
                 and T_C[0] <= Tc_C <= T_C[-1]
                 and P_MPa[0] <= Pc_MPa <= P_MPa[-1])

    # Saturation curve, clipped to the window (triple point → critical point)
    sat_T, sat_P = [], []
    if Tc_C is not None:
        try:
            T_trip_C = CP.PropsSI("Ttriple", cp_name) - 273.15
            lo = max(T_C[0], T_trip_C)
            hi = min(T_C[-1], Tc_C)
            if hi > lo:
                for Ts in np.linspace(lo, hi, 60):
                    try:
                        Ps = CP.PropsSI("P", "T", Ts + 273.15, "Q", 0, cp_name) / 1e6
                    except Exception:
                        continue
                    if P_MPa[0] <= Ps <= P_MPa[-1]:
                        sat_T.append(float(Ts))
                        sat_P.append(float(Ps))
        except Exception:
            pass

    return {
        "fluid":  fluid_key,
        "label":  spec["label"],
        "color":  spec["color"],
        "T_C":    [float(v) for v in T_C],
        "P_MPa":  [float(v) for v in P_MPa],
        "props":  props,
        "meta":   {k: {"label": PROP_META[k][0],
                       "unit":  PROP_META[k][1],
                       "log":   PROP_META[k][2]} for k in keys},
        "critical": {
            "T_C":       _safe(Tc_C) if Tc_C is not None else None,
            "P_MPa":     _safe(Pc_MPa) if Pc_MPa is not None else None,
            "in_window": bool(in_window),
        },
        "saturation": {"T_C": sat_T, "P_MPa": sat_P},
    }


# ── T,P sensitivity ──────────────────────────────────────────────────────────
#
# Relative (logarithmic) sensitivity:
#
#     S_T = (100/X)·dX/dT  [%/K]        S_P = (100/X)·dX/dP  [%/10 kPa]
#
# Pressure is reported per 10 kPa rather than per MPa because 10 kPa is the
# order of a realistic pressure-control tolerance; per-MPa values run to
# hundreds near the critical point and read as implausible. 1 %/MPa =
# 0.01 %/10 kPa, so the conversion is a factor of 100 (SENS_P_SCALE).
#
# Normalised by X so that properties with different units, and the two phases,
# are directly comparable -- a raw dX/dT comparison between CO2 and brine is
# diluted by their very different densities.
#
# The reason for the log form rather than any other normalisation is that log
# derivatives ADD through products and quotients, so an output's sensitivity
# decomposes exactly into its inputs'. Since
#     Ca = v·mu_inj/(sigma·cos θ)   and   Bo = Δrho·g·L²/(sigma·cos θ)
# and v, g, L, θ do not depend on T or P at fixed flow rate:
#     S(Ca_nw) = S(mu_nw) − S(sigma)        S(Bo) = S(Δrho) − S(sigma)
# Verified numerically to 4–5 significant figures. That decomposition is the
# useful part: it says WHICH property drives an output's sensitivity.
#
# Note the density row is not new physics -- -(1/rho)(drho/dT) is the thermal
# expansivity and (1/rho)(drho/dP) the isothermal compressibility, both already
# reported. It reproduces them to 6 figures, which is a free check on the method.

SENS_DT = 0.05     # K,   converged (see below)
SENS_DP = 0.005    # MPa, the differencing half-step (5 kPa)
SENS_P_SCALE = 0.01   # %/MPa -> %/10 kPa

# Step size matters near the critical point, where the derivative genuinely
# diverges: CO2 density at 32 °C/7.5 MPa reads -63 %/K at a 0.5 K step but
# converges to -32.0 %/K below 0.01 K. Do not coarsen these.

SENS_PROPS = [
    ("density_kg_m3",        "Density"),
    ("viscosity_uPas",       "Viscosity"),
    ("speed_sound_m_s",      "Speed of sound"),
    ("compressibility_1_GPa", "Isothermal compress."),
]


def _rel_sens(f, T_C, P_MPa):
    """
    (S_T, S_P) in %/K and %/MPa for a scalar function f(T_C, P_MPa).
    Returns (None, None) if the value is missing, zero, or the stencil
    straddles a discontinuity (e.g. the water saturation line).
    """
    try:
        x0 = f(T_C, P_MPa)
        if x0 is None or not np.isfinite(x0) or x0 == 0:
            return None, None
        tp, tm = f(T_C + SENS_DT, P_MPa), f(T_C - SENS_DT, P_MPa)
        pp, pm = f(T_C, P_MPa + SENS_DP), f(T_C, P_MPa - SENS_DP)
        if any(v is None or not np.isfinite(v) for v in (tp, tm, pp, pm)):
            return None, None
        return (_safe(100 * (tp - tm) / (2 * SENS_DT) / x0),
                _safe(100 * (pp - pm) / (2 * SENS_DP) / x0 * SENS_P_SCALE))
    except Exception:
        return None, None


def sensitivity_block(T_C, P_MPa, nw_id, S_g_L, ki_mol_L=0.0):
    """
    Relative T and P sensitivity of the NW and wetting-phase properties, plus
    the Ca/Bo decomposition. See the header comment for the definition.

    IFT honesty: above the IFT table's 20 MPa ceiling _ift() clamps, so
    dSigma/dP is identically zero for a NON-PHYSICAL reason. That would read as
    "IFT is pressure-insensitive here", so S_P(sigma) and everything derived
    from it is reported as None with sigma_clamped set, rather than as 0.
    """
    nw_f = lambda k: (lambda T, P: _nw_props(nw_id, T + 273.15, P * 1e6)[k])
    w_f  = lambda k: (lambda T, P: _w_props(T + 273.15, P * 1e6, S_g_L,
                                            ki_mol_L=ki_mol_L)[k])

    rows = []
    for key, label in SENS_PROPS:
        nT, nP = _rel_sens(nw_f(key), T_C, P_MPa)
        wT, wP = _rel_sens(w_f(key),  T_C, P_MPa)
        ratio = (abs(nT / wT) if (nT and wT) else None,
                 abs(nP / wP) if (nP and wP) else None)
        rows.append({"key": key, "label": label,
                     "nw_T": nT, "nw_P": nP, "w_T": wT, "w_P": wP,
                     "ratio_T": _safe(ratio[0]) if ratio[0] else None,
                     "ratio_P": _safe(ratio[1]) if ratio[1] else None})

    # IFT, and whether its pressure derivative is real
    sig_f = lambda T, P: _ift(nw_id, T, P, S_g_L, ki_mol_L=ki_mol_L)
    sigT, sigP = _rel_sens(sig_f, T_C, P_MPa)
    sigma_clamped = ift_out_of_range(T_C, P_MPa) or ift_out_of_range(T_C, P_MPa + SENS_DP)
    if sigma_clamped:
        sigP = None

    # Δrho drives Bo
    def drho(T, P):
        a = _nw_props(nw_id, T + 273.15, P * 1e6)["density_kg_m3"]
        b = _w_props(T + 273.15, P * 1e6, S_g_L, ki_mol_L=ki_mol_L)["density_kg_m3"]
        return None if (a is None or b is None) else abs(b - a)
    dT_, dP_ = _rel_sens(drho, T_C, P_MPa)

    mu_nw_T, mu_nw_P = rows[1]["nw_T"], rows[1]["nw_P"]
    mu_w_T,  mu_w_P  = rows[1]["w_T"],  rows[1]["w_P"]
    sub = lambda a, b: None if (a is None or b is None) else _safe(a - b)

    derived = [
        {"label": "Ca (NW injection)", "S_T": sub(mu_nw_T, sigT), "S_P": sub(mu_nw_P, sigP),
         "parts": "S(μ_NW) − S(σ)"},
        {"label": "Ca (W injection)",  "S_T": sub(mu_w_T, sigT),  "S_P": sub(mu_w_P, sigP),
         "parts": "S(μ_W) − S(σ)"},
        {"label": "Bond number Bo",    "S_T": sub(dT_, sigT),     "S_P": sub(dP_, sigP),
         "parts": "S(Δρ) − S(σ)"},
    ]

    return {
        "dT_K": SENS_DT, "dP_MPa": SENS_DP,
        "unit_T": "%/K", "unit_P": "%/10 kPa",
        "rows": rows,
        "ift":  {"S_T": sigT, "S_P": sigP, "clamped": bool(sigma_clamped)},
        "drho": {"S_T": dT_, "S_P": dP_},
        "derived": derived,
        "note_fixed_Q": True,
    }


# ── Field CCS sites on the CO2 phase diagram ─────────────────────────────────
#
# IN-SITU conditions at the injection interval (NOT wellhead or injected-stream
# conditions, which differ -- at Sleipner the injected CO2 arrives ~12 C warmer
# than the formation). Each entry records where its numbers come from:
#
#   provenance "primary"   -> the cited source was read directly and the T and P
#                             values below are quoted from it
#   provenance "secondary" -> values reported in the site literature and located
#                             via that reference, but the source was not read
#                             directly here; CHECK BEFORE PUBLISHING
#
# To add a site, append a dict with the same keys. T_C and P_MPa are both
# required (a site cannot be placed on the diagram without them).
CCS_SITES = [
    {
        "key": "sleipner", "name": "Sleipner", "country": "Norway",
        "formation": "Utsira Fm (saline aquifer)",
        "depth_m": 1012, "T_C": 35.5, "P_MPa": 10.5,
        "start": 1996, "status": "Operating",
        "scale": "~1 Mt/yr; >13 Mt by 2011",
        "reference": "Chadwick & Eiken (2013), ch. 10 in Gluyas & Mathias (eds), "
                     "Geological Storage of Carbon Dioxide (CO2), Woodhead",
        "provenance": "primary",
        "note": "Quoted: initial temperature ~35.5 °C at the 1012 m injection "
                "depth (~29 °C at reservoir top); hydrostatic pressure 10.5 MPa "
                "at the injection point.",
    },
    {
        "key": "ketzin", "name": "Ketzin", "country": "Germany",
        "formation": "Stuttgart Fm (saline aquifer)",
        "depth_m": 630, "T_C": 33.5, "P_MPa": 6.1,
        "start": 2008, "status": "Completed 2013",
        "scale": "67 kt total (pilot)",
        "reference": "Martens et al. (2012), Environmental Earth Sciences 67, 323–334",
        "provenance": "secondary",
        "note": "Initial target-horizon conditions reported as 33.5 °C / 6.1 MPa "
                "at 630 m. Pressure is BELOW the CO2 critical pressure, so the "
                "stored CO2 is not supercritical here.",
    },
    {
        "key": "in_salah", "name": "In Salah (Krechba)", "country": "Algeria",
        "formation": "Carboniferous sandstone",
        "depth_m": 1880, "T_C": 93.0, "P_MPa": 17.5,
        "start": 2004, "status": "Injection ceased 2011",
        "scale": "~3.8 Mt total",
        "reference": "Iding & Ringrose (2010), Int. J. Greenhouse Gas Control 4, 242–248",
        "provenance": "secondary",
        "note": "Storage unit at 1880 m with initial pressure 17.5 MPa and "
                "temperature 366.15 K (93 °C).",
    },
    {
        "key": "frio", "name": "Frio Brine Pilot", "country": "USA (Texas)",
        "formation": "Frio Fm (Oligocene fluvial sandstone)",
        "depth_m": 1650, "T_C": 55.0, "P_MPa": 16.5,
        "start": 2004, "status": "Completed (pilot)",
        "scale": "~1.6 kt total",
        "reference": "Doughty et al. (2008), Energy 33, 1092–1099 "
                     "(Frio pilot; see also Hovorka et al. 2006, Environmental Geosciences 13)",
        "provenance": "secondary",
        "note": "Brine reservoir reported at ~16.5 MPa and ~55 °C at ~1650 m; "
                "CO2 injected in the supercritical state.",
    },
    {
        "key": "snohvit", "name": "Snøhvit", "country": "Norway",
        "formation": "Tubåen Fm (fluvial sandstone)",
        "depth_m": 2600, "T_C": 95.0, "P_MPa": 33.0,
        "start": 2008, "status": "Tubåen injection ceased 2011",
        "scale": "~1.1 Mt into Tubåen",
        "reference": "Hansen et al. (2013), Energy Procedia 37, 3565–3573 (GHGT-11)",
        "provenance": "primary",
        "note": "T and P supplied by A. Herring (95 °C, ~330 bar). NOTE the "
                "Tubåen pressure was NOT steady: reported initial ~290 bar "
                "rising to ~390 bar, which is why Tubåen injection was stopped "
                "and moved to the Stø Fm. 330 bar is a mid-injection value, so "
                "this site is a band in pressure rather than a point.",
    },
    {
        "key": "quest", "name": "Quest", "country": "Canada (Alberta)",
        "formation": "Basal Cambrian Sands (saline aquifer)",
        "depth_m": 2000, "T_C": 60.0, "P_MPa": 20.45,
        "start": 2015, "status": "Operating",
        "scale": "~1.08 Mt/yr captured",
        "reference": "Quest Carbon Capture and Storage Project, Volume 1: Project "
                     "Description (Shell Canada / Government of Alberta), "
                     "open.alberta.ca",
        "provenance": "primary",
        "note": "Initial reservoir T and P read from the project description by "
                "A. Herring. Depth ~2 km is the commonly quoted BCS figure and is "
                "approximate; T and P are the sourced values.",
    },
    {
        "key": "ibdp", "name": "Illinois Basin – Decatur (Mt. Simon)", "country": "USA (Illinois)",
        "formation": "Mt. Simon Sandstone (saline aquifer)",
        "depth_m": 2140, "T_C": 50.0, "P_MPa": 22.0,
        "start": 2011, "status": "Injection completed 2014",
        "scale": "~1 Mt total (IBDP)",
        "reference": "Illinois Basin – Decatur Project reservoir characterisation "
                     "literature; see Finley (2014), Greenhouse Gases: Sci. Technol. 4, 571–579",
        "provenance": "secondary",
        "note": "APPROXIMATE, assembled from several sources: injection interval "
                "2129.6–2149.4 m; ~2.2e4 kPa (~22 MPa) at 2147 m; reservoir "
                "modelled as isothermal at ~49 °C (lab studies at 51 °C). "
                "Pin the primary reference before publishing.",
    },
]

# Sites located but NOT plotted, because a reservoir T or P could not be
# sourced in this session. Surfaced in the UI so the gap is visible rather
# than silently missing.
CCS_SITES_PENDING = [
    {"name": "Gorgon", "country": "Australia", "have": "Dupuy Fm at ~2300 m; "
     "formation water up to ~100 °C", "missing": "initial reservoir pressure"},
    {"name": "Cranfield", "country": "USA (Mississippi)", "have": "lower Tuscaloosa "
     "Fm at ~3200 m", "missing": "reservoir temperature and pressure"},
]


def co2_phase_diagram(T_min_C=-20.0, T_max_C=140.0,
                      P_min_MPa=0.5, P_max_MPa=40.0,
                      iso_densities=(200, 400, 600, 700, 800, 900)):
    """
    CO2 phase diagram in (T, P) with field CCS sites placed on it.

    Returns the vapour-pressure (saturation) curve, the critical and triple
    points, exact iso-density contours, and each site annotated with the CO2
    density and phase CoolProp gives at its reservoir conditions.

    Iso-density curves are solved EXACTLY rather than contoured from a grid:
    for each target density, P = PropsSI("P", "T", T, "D", rho). States inside
    the two-phase dome are rejected by requiring the density to round-trip
    through a P-T call, which it only does for a genuine single-phase state.
    """
    Tc_C   = CP.PropsSI("Tcrit", "CO2") - 273.15
    Pc_MPa = CP.PropsSI("Pcrit", "CO2") / 1e6
    Tt_C   = CP.PropsSI("Ttriple", "CO2") - 273.15
    Pt_MPa = CP.PropsSI("ptriple", "CO2") / 1e6

    # Vapour-pressure curve, triple point -> critical point, clipped to window
    sat_T, sat_P = [], []
    lo = max(T_min_C, Tt_C)
    for Ts in np.linspace(lo, Tc_C, 240):
        try:
            Ps = CP.PropsSI("P", "T", Ts + 273.15, "Q", 0, "CO2") / 1e6
        except Exception:
            continue
        if P_min_MPa <= Ps <= P_max_MPa:
            sat_T.append(float(Ts))
            sat_P.append(float(Ps))

    # Exact iso-density contours
    iso = []
    for rho in iso_densities:
        cT, cP = [], []
        for Ts in np.linspace(T_min_C, T_max_C, 200):
            T_K = Ts + 273.15
            try:
                P_Pa = CP.PropsSI("P", "T", T_K, "D", float(rho), "CO2")
                # Reject two-phase states: density must round-trip via (P,T)
                rho_back = CP.PropsSI("D", "T", T_K, "P", P_Pa, "CO2")
                if abs(rho_back - rho) / rho > 1e-3:
                    continue
            except Exception:
                continue
            P_MPa_ = P_Pa / 1e6
            if P_min_MPa <= P_MPa_ <= P_max_MPa:
                cT.append(float(Ts))
                cP.append(float(P_MPa_))
        if len(cT) > 1:
            iso.append({"rho": float(rho), "T_C": cT, "P_MPa": cP})

    # Place the sites and evaluate CO2 there
    sites = []
    for s in CCS_SITES:
        entry = dict(s)
        props = _coolprop_props("CO2", s["T_C"] + 273.15, s["P_MPa"] * 1e6)
        entry["co2_density_kg_m3"] = props["density_kg_m3"]
        entry["co2_phase"]         = props["phase"]
        entry["above_Tc"] = bool(s["T_C"] >= Tc_C)
        entry["above_Pc"] = bool(s["P_MPa"] >= Pc_MPa)
        # "Supercritical" in the storage sense: above BOTH critical values.
        entry["supercritical"] = bool(entry["above_Tc"] and entry["above_Pc"])

        # Explicit regime label. CoolProp reports T > Tc with P < Pc as
        # "supercritical gas", which our phase map flattens to "Supercritical
        # Fluid" -- misleading for a storage site, because the density there is
        # gas-like. Ketzin is exactly this case, so label it from the critical
        # comparisons instead.
        if entry["above_Tc"]:
            entry["regime"] = ("Supercritical" if entry["above_Pc"]
                               else "Gas-like (above T_c, below P_c)")
        else:
            try:
                Psat = CP.PropsSI("P", "T", s["T_C"] + 273.15, "Q", 0, "CO2") / 1e6
                entry["regime"] = "Liquid" if s["P_MPa"] >= Psat else "Vapour"
            except Exception:
                entry["regime"] = "—"
        sites.append(entry)

    return {
        "T_range": [float(T_min_C), float(T_max_C)],
        "P_range": [float(P_min_MPa), float(P_max_MPa)],
        "saturation": {"T_C": sat_T, "P_MPa": sat_P},
        "critical": {"T_C": float(Tc_C), "P_MPa": float(Pc_MPa)},
        "triple":   {"T_C": float(Tt_C), "P_MPa": float(Pt_MPa)},
        "iso_density": iso,
        "sites": sites,
        "pending": CCS_SITES_PENDING,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def calculate(T_C: float, P_MPa: float, nw_id: str, w_id: str,
              salinity_g_L: float | None = None,
              Q_ml_hr: float | None = None, core_diam_mm: float = 10.0,
              porosity: float = 0.22, char_len_um: float = 40.0,
              theta_deg: float = 10.0, Ca_nw_target: float | None = None,
              ki_mol_L: float | None = None,
              Ca_w_target: float | None = None) -> dict:
    """Return all properties at (T_C, P_MPa) for the given NW/W fluid pair."""
    P_MPa = max(P_MPa, 0.1)   # avoid P=0 in CoolProp
    T_K   = T_C + 273.15
    P_Pa  = P_MPa * 1e6

    nw_def = NW_FLUIDS[nw_id]
    w_def  = W_FLUIDS[w_id]
    S  = w_def["default_S"]  if salinity_g_L is None else float(salinity_g_L)
    ki = w_def["default_ki"] if ki_mol_L     is None else float(ki_mol_L)
    # KI brine overrides NaCl salinity
    if w_id == "ki_brine":
        S = 0.0

    # ── NW phase ──────────────────────────────────────────────────────────────
    nw = _nw_props(nw_id, T_K, P_Pa)
    nw["fluid"]  = nw_id
    nw["label"]  = nw_def["label"]
    nw["color"]  = nw_def["color"]
    nw["Mw_g_mol"] = nw_def["Mw"] * 1000

    # ── Wetting phase ─────────────────────────────────────────────────────────
    w_out = _w_props(T_K, P_Pa, S, ki_mol_L=ki)
    mu_w  = w_out.pop("mu_water_Pas", np.nan)   # internal; not returned
    w_out["fluid"]       = w_id
    w_out["label"]       = w_def["label"]
    w_out["salinity_g_L"] = S
    w_out["ki_mol_L"]    = ki

    # ── Interaction / transport ───────────────────────────────────────────────
    sol   = _solubility(nw_id, T_K, P_MPa, mu_w if np.isfinite(mu_w) else 1e-3, S, ki_mol_L=ki)
    D     = _diffusivity(nw_id, T_K, mu_w if np.isfinite(mu_w) else 1e-3)
    ift   = _ift(nw_id, T_C, P_MPa, S, ki_mol_L=ki)

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
        "ift_out_of_range": ift_out_of_range(T_C, P_MPa),
        "solubility":      sol,
        "diffusivity_m2_s": D,
        "mobility_ratio":   mobility_ratio,
        "density_ratio":    density_ratio,
        "buoyancy_N_m3":    buoyancy_N_m3,
    }

    # ── If a Ca target is supplied, derive Q from it ──────────────────────────
    # Ca = v·μ_injected/(σ·cos θ), so the viscosity used to invert for v is the
    # INJECTED phase: μ_NW for drainage (Ca_NW), μ_W for imbibition (Ca_W).
    # NW takes precedence if both are somehow supplied.
    # Use the SAME viscosities compute_flow uses, so the returned Ca reproduces
    # the target exactly. Note mu_w_Pas above is the pure-water viscosity (for
    # solubility/diffusivity); the wetting phase in the flow calculation is the
    # brine, whose viscosity differs once salt is present.
    mu_w_flow_Pas = (w_out.get("viscosity_uPas") or 0) * 1e-6

    derived_Q    = None
    derived_from = None
    if Ca_nw_target is not None and Ca_nw_target > 0:
        ca_target, mu_inj, derived_from = Ca_nw_target, mu_nw_Pas, "Ca_nw"
    elif Ca_w_target is not None and Ca_w_target > 0:
        ca_target, mu_inj, derived_from = Ca_w_target, mu_w_flow_Pas, "Ca_w"
    else:
        ca_target, mu_inj = None, None

    if ca_target is not None and ift and ift > 0 and mu_inj and mu_inj > 0:
        cos_theta = max(np.cos(np.radians(theta_deg)), 1e-9)
        A_m2 = np.pi * (core_diam_mm * 0.5e-3) ** 2
        v_m_s = ca_target * (ift * 1e-3) * cos_theta / mu_inj
        derived_Q = v_m_s * A_m2 * porosity * 1e6 * 3600   # m³/s → ml/hr
        if Q_ml_hr is None:
            Q_ml_hr = derived_Q
    else:
        derived_from = None

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
        flow["derived_from"]    = derived_from   # "Ca_nw" | "Ca_w" | None
        flow["core_diam_mm"]    = core_diam_mm
        flow["porosity"]        = porosity
        flow["char_len_um"]     = char_len_um
        flow["theta_deg"]       = theta_deg

    try:
        sens = sensitivity_block(T_C, P_MPa, nw_id, S, ki_mol_L=ki)
    except Exception:
        sens = None

    return {
        "T_C":    T_C,
        "P_MPa":  P_MPa,
        "nw":     nw,
        "wetting": w_out,
        "interaction": interaction,
        "sensitivity": sens,
        "flow":   flow,
    }
