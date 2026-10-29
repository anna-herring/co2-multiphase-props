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
        (solubility is NO LONGER REPORTED as of 2026-10-08 -- the validated
        path is retained unused; see the note in calculate())
  - Diffusivity                           : Stokes-Einstein scaling from D_25 ref
  - Interfacial tension                   : 2-D interpolation of published data

WARNING: CoolProp has NO "fugacity" output key, and AbstractState.fugacity(i)
fails for pure fluids. Both failures are silent if caught, leaving an ideal-gas
value (f = P) that is ~2.7x wrong for CO2 at 50 C / 10 MPa. Get the fugacity
coefficient from residual properties instead -- see _fugacity_MPa below.
"""

import numpy as np
import CoolProp.CoolProp as CP
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

try:                                  # package import (uvicorn main:app) ...
    from . import mutailipu as _mut
except ImportError:                   # ... or run from this directory
    import mutailipu as _mut

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
        # Modified Stokes-Einstein constants for CO2 in pure water, Cadogan,
        # Maitland & Trusler (2014), eq 8: a = a298 [1 + alpha (T/K - 298)]
        "SE_a298_pm": 168.0, "SE_alpha": 2.0e-3,
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

# CO2-water IFT: see _PEREIRA_TABLE6 / _PereiraIFT below (measured data, Pereira 2016).
# The axes above and the H2 / N2 grids below are DORMANT (CO2-only tool).

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

_IFT_INTERP = {}   # dormant: H2 / N2 grids would be registered here if restored

IFT_SOURCES = {
    "CO2": "Mutailipu et al. (2024)",
}

# ── CO2/H2O interfacial tension: Pereira et al. (2016), Table 6 ──────────────
# Pereira, L.M.C.; Chapoy, A.; Burgass, R.; Oliveira, M.B.; Coutinho, J.A.P.;
# Tohidi, B. "Study of the impact of high temperatures and pressures on the
# equilibrium densities and interfacial tension of the carbon dioxide/water
# system", J. Chem. Thermodyn. 2016, 93, 404-415. Axisymmetric drop shape
# analysis, 298-469 K, up to 69 MPa, pure water.
#
# (T [K], P [MPa], IFT [mN/m], Uc [mN/m]); Uc is the combined expanded
# uncertainty (>95 % confidence; mean 0.31, maximum 1.87 at the three-phase
# point). This REPLACES an earlier hand-built 10x10 grid whose per-cell
# provenance was not recorded.
_PEREIRA_TABLE6 = (
    (298.5, 0.70, 66.95, 0.21),
    (298.5, 1.69, 60.56, 0.17),
    (298.5, 3.68, 48.70, 0.14),
    (298.5, 5.64, 37.74, 0.19),
    (298.5, 6.06, 35.06, 0.22),
    (298.5, 6.40, 32.97, 0.52),
    (298.6, 6.49, 31.69, 1.15),
    (298.6, 6.49, 30.79, 1.87),
    (298.4, 6.83, 31.00, 0.49),
    (298.5, 7.20, 30.99, 0.40),
    (298.5, 8.10, 31.09, 0.33),
    (298.6, 8.97, 30.98, 0.34),
    (298.6, 10.63, 30.73, 0.34),
    (298.6, 13.18, 30.62, 0.37),
    (298.6, 15.15, 30.81, 0.42),
    (298.6, 20.83, 29.84, 0.49),
    (313.3, 0.46, 68.52, 0.33),
    (313.3, 1.85, 59.60, 0.31),
    (313.3, 3.57, 50.67, 0.41),
    (313.3, 5.34, 42.75, 0.22),
    (313.3, 7.01, 36.65, 0.36),
    (313.2, 9.30, 30.62, 0.82),
    (313.2, 10.44, 29.91, 0.58),
    (313.3, 13.92, 29.59, 0.40),
    (313.3, 20.82, 29.20, 0.32),
    (313.3, 25.68, 28.96, 0.41),
    (313.3, 29.11, 29.05, 0.46),
    (313.2, 31.19, 28.88, 0.51),
    (333.0, 0.92, 62.31, 0.15),
    (333.0, 1.79, 58.98, 0.17),
    (333.0, 3.66, 52.28, 0.16),
    (333.0, 5.00, 47.96, 0.13),
    (333.0, 5.85, 45.30, 0.15),
    (333.0, 7.18, 41.29, 0.20),
    (333.0, 8.89, 36.76, 0.21),
    (333.1, 10.04, 34.08, 0.26),
    (333.2, 11.42, 30.10, 0.46),
    (333.1, 15.01, 29.46, 0.35),
    (333.2, 21.12, 28.13, 0.30),
    (333.2, 24.27, 27.39, 0.33),
    (333.2, 29.58, 27.00, 0.35),
    (333.2, 31.29, 27.25, 0.41),
    (333.2, 39.64, 26.26, 0.41),
    (333.2, 50.31, 26.27, 0.54),
    (333.2, 59.91, 26.16, 0.75),
    (373.8, 0.34, 56.79, 0.23),
    (373.9, 1.30, 55.02, 0.18),
    (373.9, 3.86, 49.03, 0.17),
    (374.0, 5.60, 45.62, 0.18),
    (374.0, 7.57, 41.88, 0.15),
    (374.0, 12.09, 34.14, 0.19),
    (374.0, 16.96, 29.00, 0.20),
    (374.0, 25.15, 26.41, 0.18),
    (373.9, 40.03, 24.35, 0.27),
    (374.0, 50.18, 23.64, 0.27),
    (374.1, 62.07, 23.38, 0.36),
    (374.0, 69.12, 23.42, 0.35),
    (424.4, 1.55, 46.34, 0.14),
    (424.4, 5.81, 40.10, 0.16),
    (424.4, 10.21, 34.60, 0.12),
    (424.4, 16.43, 29.00, 0.13),
    (424.5, 23.83, 25.55, 0.15),
    (424.6, 30.05, 22.82, 0.11),
    (424.6, 39.68, 20.89, 0.11),
    (424.5, 50.11, 19.82, 0.25),
    (424.6, 60.19, 18.83, 0.24),
    (424.6, 69.14, 18.42, 0.26),
    (469.0, 5.90, 32.91, 0.15),
    (469.2, 10.08, 28.69, 0.15),
    (469.1, 10.79, 28.49, 0.11),
    (469.2, 14.91, 24.88, 0.11),
    (469.2, 17.70, 23.14, 0.10),
    (469.2, 20.71, 21.82, 0.11),
    (469.2, 29.97, 18.22, 0.18),
    (469.2, 40.06, 15.45, 0.14),
    (469.2, 50.16, 13.84, 0.13),
    (469.4, 59.96, 12.73, 0.12),
    (469.2, 69.09, 12.65, 0.18)
)


def _water_surface_tension_mN_m(T_K):
    """Surface tension of water, IAPWS R1-76(2014): B*tau^mu*(1 + b*tau)."""
    tau = 1.0 - T_K / 647.096
    return 235.8 * tau ** 1.256 * (1.0 - 0.625 * tau)


class _PereiraIFT:
    """
    CO2-water IFT interpolated from Pereira et al. (2016) Table 6.

    Six isotherms (25.4, 40.1, 60.0, 100.8, 151.4, 196.0 °C). Along each
    isotherm IFT(P) is a shape-preserving (PCHIP) interpolant of the measured
    points, extended down to the water vapour pressure with the pure-water
    surface tension there (no CO2 present => no CO2 effect), because the data
    stop at 0.34-5.9 MPa. At a given P the isotherms that REACH that pressure
    are interpolated across temperature, again with PCHIP.

    This is interpolation only. Outside the data -- below 25.4 °C or above
    196 °C, or at a pressure beyond the isotherms that reach it (25.4 °C stops
    at 20.8 MPa, 40.1 °C at 31.2 MPa) -- the temperature is clamped to the
    nearest covered value and the second return value is False. Nothing is
    extrapolated.

    Two rows share P = 6.49 MPa at 298.6 K (footnote a: the vapour + liquid +
    liquid point, IFT 31.69 gas/water and 30.79 liquid CO2/water); their mean
    is used there, i.e. up to ~0.45 mN/m from either side of that one pressure.
    """
    GROUP_K = 1.5      # rows within this many K are one isotherm

    def __init__(self, table):
        rows = sorted(table)
        groups = []
        for r in rows:
            if groups and abs(r[0] - groups[-1][-1][0]) <= self.GROUP_K:
                groups[-1].append(r)
            else:
                groups.append([r])
        self.T_C, self.P_lo, self.P_hi, self._fn = [], [], [], []
        self.P_meas_lo = []     # lowest measured P per isotherm (P_lo is the extended one)
        for g in groups:
            T_K = float(np.mean([r[0] for r in g]))
            byP = {}
            for _, P, s, _u in g:
                byP.setdefault(round(P, 6), []).append(s)
            P = np.array(sorted(byP))
            s = np.array([np.mean(byP[p]) for p in P])
            P_sat = CP.PropsSI("P", "T", T_K, "Q", 0, "Water") / 1e6
            self.P_meas_lo.append(float(P[0]))       # lowest MEASURED pressure
            if P_sat < P[0]:
                P = np.r_[P_sat, P]
                s = np.r_[_water_surface_tension_mN_m(T_K), s]
            self.T_C.append(T_K - 273.15)
            self.P_lo.append(float(P[0]))
            self.P_hi.append(float(P[-1]))
            self._fn.append(PchipInterpolator(P, s))
        self.T_C = np.array(self.T_C)
        self.P_lo = np.array(self.P_lo)
        self.P_hi = np.array(self.P_hi)

    def at_isotherm(self, i, P_MPa):
        return float(self._fn[i](min(max(P_MPa, self.P_lo[i]), self.P_hi[i])))

    def __call__(self, T_C, P_MPa):
        """-> (IFT [mN/m], True if (T, P) lies inside the measured data)."""
        idx = [i for i in range(len(self.T_C)) if self.P_hi[i] >= P_MPa]
        in_range = True
        if not idx:                    # beyond every isotherm: clamp P (each
            idx = list(range(len(self.T_C)))   # isotherm is read at its own top
            in_range = False                   # pressure), keep T interpolation
        Ts = self.T_C[idx]
        T_use = min(max(T_C, Ts.min()), Ts.max())
        if abs(T_use - T_C) > 0.5:      # 0.5 K tolerance at the edges of the data
            in_range = False
        vals = np.array([self.at_isotherm(i, P_MPa) for i in idx])
        if len(idx) == 1:
            return float(vals[0]), in_range
        return float(PchipInterpolator(Ts, vals)(T_use)), in_range

    def explain(self, T_C, P_MPa):
        """Why __call__ is NOT a plain interpolation of measurements at (T, P):
        a list of dict(outside=..., estimated_by=...). Mirrors __call__'s own
        logic, so it describes exactly what the number is made of."""
        notes = []
        n = len(self.T_C)
        idx = [i for i in range(n) if self.P_hi[i] >= P_MPa]
        if not idx:
            notes.append(dict(
                outside=f"P = {P_MPa:.1f} MPa is above the highest measured "
                        f"pressure ({self.P_hi.max():.0f} MPa)",
                estimated_by="each isotherm is read at its own highest measured "
                             "pressure, then interpolated in temperature"))
            idx = list(range(n))
        Ts = self.T_C[idx]
        T_use = min(max(T_C, Ts.min()), Ts.max())
        if abs(T_use - T_C) > 0.5:
            notes.append(dict(
                outside=f"T = {T_C:.0f} °C, P = {P_MPa:.1f} MPa has no measurements "
                        f"(at this pressure the data span {Ts.min():.1f}–{Ts.max():.1f} °C)",
                estimated_by=f"temperature held at {T_use:.1f} °C: the IFT at the "
                             f"nearest measured temperature for this pressure"))
        plo = float(np.interp(T_use, Ts, np.array(self.P_meas_lo)[idx]))
        if P_MPa < plo - 1e-6:
            T_K = T_use + 273.15
            P_sat = CP.PropsSI("P", "T", T_K, "Q", 0, "Water") / 1e6
            notes.append(dict(
                outside=f"P = {P_MPa:.2f} MPa is below the lowest measured pressure "
                        f"at {T_use:.0f} °C (about {plo:.1f} MPa)",
                estimated_by=f"interpolated between the lowest measured point and the "
                             f"surface tension of pure water at its vapour pressure "
                             f"({P_sat:.3f} MPa, {_water_surface_tension_mN_m(T_K):.1f} mN/m)"))
        return notes


_PEREIRA = None


def _pereira_ift():
    """Built on first use (it needs CoolProp, and import order is top-down)."""
    global _PEREIRA
    if _PEREIRA is None:
        _PEREIRA = _PereiraIFT(_PEREIRA_TABLE6)
    return _PEREIRA


# ── NaCl effect on CO2/H2O IFT: Pereira et al. (2017), Tables 4 and 5 ────────
# Pereira, L.M.C.; Chapoy, A.; Burgass, R.; Tohidi, B. "Interfacial tension of
# CO2 + brine systems: Experiments and predictive modelling", Adv. Water
# Resour. 2017, 103, 64-75. Pendant-drop (ADSA) IFT of CO2 + NaCl(aq) at 0.98
# and 1.98 mol/kg, 298.6-423 K, 3-69.5 MPa. NaCl only: the paper has no KCl or
# KI measurements (KCl appears only in a literature NaCl+KCl mixture).
#
# (m [mol/kg], T [K], P [MPa], IFT [mN/m], Uc [mN/m]); Uc as in Table 6 above.
_PEREIRA2017_NACL = (
    (0.98, 298.6, 3.00, 53.39, 0.16),
    (0.98, 298.6, 4.96, 42.37, 0.17),
    (0.98, 298.6, 5.66, 38.43, 0.17),
    (0.98, 298.6, 10.31, 32.98, 0.39),
    (0.98, 298.6, 15.58, 32.35, 0.42),
    (0.98, 298.6, 20.21, 32.33, 0.54),
    (0.98, 333.2, 3.91, 52.10, 0.16),
    (0.98, 333.2, 7.40, 41.70, 0.16),
    (0.98, 333.2, 18.84, 31.70, 0.27),
    (0.98, 333.2, 40.94, 29.40, 0.54),
    (0.98, 333.2, 55.07, 29.20, 0.68),
    (0.98, 373.4, 7.19, 44.04, 0.17),
    (0.98, 373.4, 12.32, 35.94, 0.22),
    (0.98, 373.4, 20.14, 30.60, 0.18),
    (0.98, 373.4, 40.89, 26.81, 0.26),
    (0.98, 373.4, 69.50, 25.10, 0.44),
    (0.98, 423.3, 5.69, 42.30, 0.14),
    (0.98, 423.3, 11.12, 35.60, 0.14),
    (0.98, 423.3, 22.37, 27.30, 0.16),
    (0.98, 423.3, 43.07, 22.20, 0.19),
    (0.98, 423.3, 69.07, 19.90, 0.25),
    (1.98, 298.6, 2.96, 54.98, 0.16),
    (1.98, 298.6, 4.12, 48.43, 0.15),
    (1.98, 298.6, 5.69, 39.59, 0.17),
    (1.98, 298.6, 10.25, 34.09, 0.35),
    (1.98, 298.6, 14.82, 33.74, 0.40),
    (1.98, 298.6, 19.90, 33.61, 0.46),
    (1.98, 333.2, 4.09, 53.20, 0.19),
    (1.98, 333.2, 7.86, 42.14, 0.15),
    (1.98, 333.2, 18.85, 33.34, 0.27),
    (1.98, 333.2, 41.72, 31.22, 0.40),
    (1.98, 333.2, 54.76, 30.99, 0.53),
    (1.98, 373.3, 4.61, 50.69, 0.19),
    (1.98, 373.3, 11.49, 39.01, 0.24),
    (1.98, 373.3, 21.94, 31.76, 0.22),
    (1.98, 373.3, 43.18, 28.31, 0.32),
    (1.98, 373.3, 69.51, 27.16, 0.38),
    (1.98, 423.2, 5.65, 44.36, 0.17),
    (1.98, 423.2, 12.17, 36.00, 0.18),
    (1.98, 423.2, 21.88, 29.07, 0.15),
    (1.98, 423.2, 44.35, 23.25, 0.22),
    (1.98, 423.2, 69.00, 21.64, 0.23)
)


class _NaClIFTIncrement:
    """
    dSigma(T, P, m) = IFT(CO2 + NaCl brine) - IFT(CO2 + pure water), from the
    42 measured brine points minus the pure-water interpolant at the same
    (T, P). Reconstructs the paper's own mean increases (1.38 / 1.95 mN/m at
    0.98 mol/kg and 2.69 / 3.66 at 1.98 mol/kg, against its quoted 1.42 / 2.01
    and 2.67 / 3.73 for 298 K / 333-423 K).

    Four isotherms (25.5, 60.1, 100.2, 150.2 C), two molalities. Along each
    isotherm the increment is a PCHIP interpolant of the measured points; at a
    given P the isotherms that cover it are interpolated across temperature;
    then the increment is interpolated in molality through (0, 0), (0.98, d1),
    (1.98, d2). The increment is NOT linear in molality: at the same (T, P) the
    1.98 mol/kg value is typically only 1.5-1.8x the 0.98 mol/kg value.

    Coverage is tracked per molality level, and only the level(s) the request
    needs are required: m <= 0.98 uses d1 alone; larger m uses d1 and d2.
    Nothing is extrapolated silently. The second return value is False when T
    lies outside 25.5-150.2 C, P below the lowest measured pressure (3-7 MPa
    depending on isotherm and level) or above the highest (20 MPa at 25 C,
    55 MPa at 60 C, 69 MPa above) by more than P_SLACK, or m exceeds 1.98
    mol/kg by more than 10 %. Then T and P are clamped, and m beyond 1.98 is
    extrapolated linearly from the last segment.
    """
    GROUP_K = 1.5
    M_SLACK = 1.10
    P_SLACK = 0.5      # MPa above the highest measured P; the increment is flat there

    def __init__(self, table, water):
        self.m = sorted({r[0] for r in table})
        Ts = sorted({r[1] for r in table})
        groups = []
        for t in Ts:
            if groups and t - groups[-1][-1] <= self.GROUP_K:
                groups[-1].append(t)
            else:
                groups.append([t])
        n = len(groups)
        self.T_C = np.array([np.mean(g) - 273.15 for g in groups])
        self.P_lo = np.zeros((len(self.m), n))
        self.P_hi = np.zeros((len(self.m), n))
        self._fn = {}
        for i, g in enumerate(groups):
            for j, m in enumerate(self.m):
                pts = sorted((P, ift - water(T_K - 273.15, P)[0])
                             for (mm, T_K, P, ift, _u) in table if mm == m and T_K in g)
                P = np.array([p for p, _ in pts])
                d = np.array([x for _, x in pts])
                self._fn[(j, i)] = PchipInterpolator(P, d)
                self.P_lo[j, i] = P[0]
                self.P_hi[j, i] = P[-1]

    def _level(self, j, T_C, P_MPa):
        """Increment at molality level j, and whether (T, P) is covered."""
        n = len(self.T_C)
        idx = [i for i in range(n)
               if self.P_lo[j, i] <= P_MPa <= self.P_hi[j, i] + self.P_SLACK]
        in_range = True
        if not idx:
            idx = list(range(n))
            in_range = False
        Ts = self.T_C[idx]
        T_use = min(max(T_C, Ts.min()), Ts.max())
        if abs(T_use - T_C) > 0.5:      # 0.5 K tolerance at the edges of the data
            in_range = False
        vals = np.array([float(self._fn[(j, i)](min(max(P_MPa, self.P_lo[j, i]), self.P_hi[j, i])))
                         for i in idx])
        if len(idx) == 1:
            return float(vals[0]), in_range
        if len(idx) == 2:
            return float(np.interp(T_use, Ts, vals)), in_range
        return float(PchipInterpolator(Ts, vals)(T_use)), in_range

    def __call__(self, T_C, P_MPa, m):
        """-> (increment [mN/m], True if inside the measured data)."""
        if m <= 0:
            return 0.0, True
        m1, m2 = self.m
        d1, ok = self._level(0, T_C, P_MPa)
        if m <= m1:
            return float(d1 * m / m1), bool(ok)
        d2, ok2 = self._level(1, T_C, P_MPa)
        if m <= m2:
            d = d1 + (d2 - d1) * (m - m1) / (m2 - m1)
        else:
            d = d2 + (d2 - d1) / (m2 - m1) * (m - m2)
        return float(d), bool(ok and ok2 and m <= m2 * self.M_SLACK)

    def explain(self, T_C, P_MPa, m, check_molality=True):
        """List of dict(outside=..., estimated_by=...) for the parts of the salt
        increment at (T, P, m) that are not a plain interpolation of Pereira
        (2017). Mirrors _level()/__call__ (same coverage tests)."""
        notes, seen = [], set()
        if m <= 0:
            return notes
        n = len(self.T_C)
        for j in ((0,) if m <= self.m[0] else (0, 1)):
            idx = [i for i in range(n)
                   if self.P_lo[j, i] <= P_MPa <= self.P_hi[j, i] + self.P_SLACK]
            if not idx:
                lo, hi = self.P_lo[j].min(), self.P_hi[j].max()
                new = [dict(
                    outside=f"P = {P_MPa:.1f} MPa is {'below' if P_MPa < lo else 'above'} "
                            f"the measured NaCl-brine pressures ({lo:.1f}–{hi:.0f} MPa)",
                    estimated_by="salt increment held at its value at the nearest "
                                 "measured pressure on each isotherm, then interpolated "
                                 "in temperature")]
                idx = list(range(n))
            else:
                new = []
            Ts = self.T_C[idx]
            T_use = min(max(T_C, Ts.min()), Ts.max())
            if abs(T_use - T_C) > 0.5:
                new.append(dict(
                    outside=f"T = {T_C:.0f} °C, P = {P_MPa:.1f} MPa has no NaCl-brine "
                            f"measurements (at this pressure the data span "
                            f"{Ts.min():.1f}–{Ts.max():.1f} °C)",
                    estimated_by=f"salt increment taken at {T_use:.1f} °C, the nearest "
                                 f"measured temperature for this pressure"))
            for nt in new:
                if nt["outside"] not in seen:
                    seen.add(nt["outside"])
                    notes.append(nt)
        if check_molality and m > self.m[-1] * self.M_SLACK:
            notes.append(dict(
                outside=f"molality {m:.2f} mol/kg is above the highest measured "
                        f"({self.m[-1]:.2f} mol/kg)",
                estimated_by="salt increment extrapolated linearly in molality from "
                             "the last two measured levels"))
        return notes


_NACL_INC = None


def _nacl_increment():
    global _NACL_INC
    if _NACL_INC is None:
        _NACL_INC = _NaClIFTIncrement(_PEREIRA2017_NACL, _pereira_ift())
    return _NACL_INC


# ── Brine density and viscosity: Laliberte (2009) ────────────────────────────
# Laliberte, M. (2009) J. Chem. Eng. Data 54(6), 1725-1760, doi:10.1021/je8008123
# (supersedes Laliberte & Cooper 2004 density and Laliberte 2007 viscosity).
#   Eq 1  1/rho_m = w_w/rho_w + w_s/rho_app
#   Eq 3  rho_app = (c0(1-w_w)+c1) exp(1e-6 (t+c4)^2) / ((1-w_w) + c2 + c3 t)
#   Eq 5  eta_m   = eta_w^w_w * eta_s^w_s
#   Eq 7  eta_s   = exp((V1(1-w_w)^V2 + V3)/(V4 t + 1)) / (V5 (1-w_w)^V6 + 1)
# t in degC, w = mass fraction.  The coefficients are in the paper's Supporting
# Information spreadsheet (not the PDF); they were taken from the `chemicals`
# package's copy of that table and checked against the fitted ranges printed in
# the paper's Tables 53 (KI) and 84 (NaCl) -- t and w_max match exactly.
# The fits are at ~atmospheric pressure, so the model is applied as a RATIO to
# pure water and multiplied onto CoolProp's water at the actual (T, P): the
# freshwater limit is then exactly CoolProp, and water's P dependence is kept.
_LAL = {
    "NaCl": dict(
        M=Mw_NaCl,
        c=(-0.00324112223655149, 0.0636354335906616, 1.01371399467365,
           0.0145951015210159, 3317.34854426537),
        v=(16.221788633396, 1.32293086770011, 1.48485985010431,
           0.00746912559657377, 30.7802007540575, 2.05826852322558),
        rho_T=(0.0, 140.0), rho_wmax=0.26589930421877,
        mu_T=(5.0, 154.0),  mu_wmax=0.264456748962402),
    "KI": dict(
        M=Mw_KI,
        c=(278.943640132031, 1407.61912896448, 2.50114423686534,
           0.0138511632772227, 1387.54108036637),
        v=(2.98583852051338, 2.37765263680441, 0.49977486681403,
           0.00934248088339622, 1.70893863931587, -0.00165048180254392),
        rho_T=(0.0, 100.02), rho_wmax=0.6669,
        mu_T=(5.0, 95.0),    mu_wmax=0.6274217585693),
}


def _lal_rho_w(t):
    """Laliberte's own pure-water density (Eq 2), kg/m3 -- the reference the
    apparent-density fit was made against."""
    return ((((((-2.8054253e-10 * t + 1.0556302e-7) * t - 4.6170461e-5) * t
               - 0.0079870401) * t + 16.945176) * t + 999.83952)
            / (1.0 + 0.01687985 * t))


def _lal_mu_w(t):
    """Laliberte's own pure-water viscosity (Eq 6), mPa.s."""
    return (t + 246.0) / ((0.05594 * t + 5.2842) * t + 137.37)


def _lal_ratios(salt, T_C, w):
    """(rho_m/rho_w, mu_m/mu_w, extrapolated) of a `salt` solution of mass
    fraction w, from Laliberte's own water -- a pure salt effect.  T is held at
    the edge of each fitted range (never extrapolated); `extrapolated` is True
    if either T or w left the range the fit was made on."""
    p = _LAL[salt]
    c0, c1, c2, c3, c4 = p["c"]
    V1, V2, V3, V4, V5, V6 = p["v"]
    ww = 1.0 - w
    tr = min(max(T_C, p["rho_T"][0]), p["rho_T"][1])
    tm = min(max(T_C, p["mu_T"][0]), p["mu_T"][1])
    rho_app = (((c0 * w + c1) * np.exp(1e-6 * (tr + c4) ** 2)) / (w + c2 + c3 * tr))
    rho_w = _lal_rho_w(tr)
    rho_m = 1.0 / (ww / rho_w + w / rho_app)
    eta_s = (np.exp((V1 * w ** V2 + V3) / (V4 * tm + 1.0)) / (V5 * w ** V6 + 1.0))
    eta_w = _lal_mu_w(tm)
    eta_m = eta_w ** ww * eta_s ** w
    extrap = bool(T_C != tr or T_C != tm or w > p["rho_wmax"] or w > p["mu_wmax"])
    return float(rho_m / rho_w), float(eta_m / eta_w), extrap


def _brine_state(salt, conc, T_C, rho_water):
    """Solve a brine of given composition for its mass fraction.

    conc : g/L of solution for NaCl, mol/L of solution for KI (the UI inputs).
    The concentration fixes the solute mass per m3 of SOLUTION, so w = c/rho_m
    and rho_m depends on w: a contraction, solved by fixed-point iteration.

    Returns dict(w, molality [mol/kg water], rho [kg/m3], mu_ratio,
    extrapolated), with rho = CoolProp water density x Laliberte's ratio.
    """
    if conc <= 0:
        return dict(w=0.0, molality=0.0, rho=float(rho_water), mu_ratio=1.0,
                    extrapolated=False)
    p = _LAL[salt]
    c_kg_m3 = conc if salt == "NaCl" else conc * p["M"] * 1000.0
    rho = float(rho_water) * 1.05
    for _ in range(40):
        w = min(c_kg_m3 / rho, 0.95)
        rr, _, _ = _lal_ratios(salt, T_C, w)
        new = float(rho_water) * rr
        if abs(new - rho) < 1e-9 * rho:
            rho = new
            break
        rho = new
    w = min(c_kg_m3 / rho, 0.95)
    rr, mr, extrap = _lal_ratios(salt, T_C, w)
    return dict(w=w, molality=w / ((1.0 - w) * p["M"]), rho=float(rho_water) * rr,
                mu_ratio=mr, extrapolated=extrap)


def _rho_water_cp(T_C, P_MPa):
    """CoolProp liquid-water density for the molality helpers (kg/m3)."""
    T_K = T_C + 273.15
    try:
        P_Pa = max(P_MPa * 1e6, CP.PropsSI("P", "T", T_K, "Q", 0, "Water") * 1.001)
        return CP.PropsSI("D", "T", T_K, "P", P_Pa, "Water")
    except Exception:
        return 1000.0


def _nacl_molality(S_g_L, T_C, P_MPa):
    """g/L of solution -> mol NaCl per kg of water (Laliberte density)."""
    if S_g_L <= 0:
        return 0.0
    return _brine_state("NaCl", S_g_L, T_C, _rho_water_cp(T_C, P_MPa))["molality"]


def _ki_molality(ki_mol_L, T_C, P_MPa):
    """mol/L of solution -> mol KI per kg of water (Laliberte density)."""
    if ki_mol_L <= 0:
        return 0.0
    return _brine_state("KI", ki_mol_L, T_C, _rho_water_cp(T_C, P_MPa))["molality"]


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


# ============================================================================
# UNUSED since 2026-10-09 -- brine corrections of kappa_T, alpha, c_p and gamma.
#
# Anna chose to correct ONLY density and viscosity for salt and to report every
# other wetting-phase property at pure-water values, labelled as such in the UI
# ("Pure water thermodynamic properties"). Nothing below, down to and including
# _brine_alpha_cp(), is called any more. It is kept (validated, but never
# committed elsewhere) in case the brine thermodynamics are wanted again:
# kappa_T fits KAPPA_NACL / KAPPA_KI; expansivity and heat-capacity fits
# _BRINE_FIT; the validity ranges _BRINE_FIT_RANGE; and _WITHHELD_WITH_SALT.
# Source for all of it: Apelblat (2017), J. Mol. Liq., supplementary tables.
# ============================================================================
# Isothermal compressibility of brine, relative to pure water:
#     κ_T(m) / κ_T(water) = exp(-KAPPA_SALT * m),   m in mol/kg WATER
#
# Both coefficients are fitted to MEASURED κ_T, tabulated directly in
#   Apelblat, A. (2017), "Effect of temperature on compressibility properties
#   of 0.1, 0.5 and 1.0 molal solutions of alkali metal halides. Part 1 ...",
#   Journal of Molecular Liquids — supplementary Table S1 (NaCl) and
#   Table 11 (KI; captioned "TABLE 11", the S is missing in the supplement).
# Measured at 0.1, 0.5 and 1.0 mol/kg over 278–353 K; κ_T there is in
# 1e-10 Pa⁻¹. Each coefficient is the least-squares fit of ln(κ_T/κ_water)
# against m, averaged over temperature.
#
# NOTE KI suppresses compressibility slightly MORE than NaCl per mol/kg
# (0.122 vs 0.106) — the opposite of what was assumed when KI was previously
# left uncorrected.
#
# The coefficient is genuinely temperature dependent (NaCl 0.136 at 5 °C
# falling to 0.091 at 80 °C), but holding it constant costs at most ~3% in
# κ_T at 1 mol/kg across that range, against a correction of ~11%. A constant
# is therefore used; if that 3% ever matters, the per-temperature values are
# in the paper's tables.
#
# Supersedes an anchor-on-one-point estimate of 0.112 for NaCl, which this
# measured data confirms to 2% (0.110 at 25 °C), and a previous coefficient of
# 2.6e-3 that was ~43x too weak.
KAPPA_NACL = 0.106
KAPPA_KI   = 0.122

# Thermal expansivity and heat capacity of NaCl(aq) and KI(aq), fitted to the
# SAME paper's supplementary Tables S2 (NaCl) and S12 (KI): density, viscosity,
# specific heat capacity and isobaric thermal expansivity at 0.1, 0.5 and
# 1.0 mol/kg, 5-80 °C, ~0.1 MPa.
#
#   expansivity  alpha_brine = alpha_water(T,P) + m**p * poly3(t) * 1e-4   [1/K]
#   heat cap.    cp_brine = (cp_water(T,P) + m*phi(t)) / (1 + m*M)         [J/kg/K]
#                with phi = apparent molar heat capacity [J/mol/K]
# t = (T_C - 40)/40, polynomial coefficients in increasing powers. Residuals
# against the tables: NaCl alpha rms 0.06e-4 /K (0.02e-4 up to 60 °C); KI alpha
# rms 0.13e-4 /K (0.07e-4 up to 60 °C).
#
# Heat capacity is used ONLY inside c_p - c_v = T*alpha^2/(rho*kappa_T), i.e. for
# the brine heat-capacity ratio gamma. It is deliberately NOT displayed: the
# supplement's heat-capacity columns fail an ionic-additivity test (KI is 33
# J/mol/K away from NaI + KCl - NaCl; RbCl is wildly off), so they are not
# trustworthy as measurements. gamma - 1 is ~0.02 and scales as 1/c_v, so a
# 1% error in c_p moves gamma by ~2e-4 -- immaterial.
#
# Fitted only inside the measured range; outside it the temperature polynomial
# is HELD at its boundary value (a cubic diverges quickly) and the response
# carries brine_fit_extrapolated = True.
_BRINE_FIT = {
    "NaCl": {"M": 0.05844, "p": 0.82710423,
             "a":   [0.31743469, -0.81175858, 0.50114249, -0.18259056],
             "phi": [-71.135048, 24.760999, -47.064616, 28.521852]},
    "KI":   {"M": 0.16600, "p": 1.29844540,
             "a":   [0.68019851, -0.83792974, -0.34823445, -0.95369127],
             "phi": [-71.133111, 24.982246, -46.828793, 28.190801]},
}
# m_max is the top of the measured data; m_flag is where the UI starts calling
# it extrapolated. The 10% slack matters: 1.0 mol/L NaCl is 1.014 mol/kg and
# 1.0 mol/L KI is 1.044 mol/kg, i.e. the default state sits AT the edge of the
# data, and flagging that would be a spurious warning on every first load.
#
# P_MPa: the supplement does not state a pressure for its tables; they are
# ASSUMED atmospheric (~0.1 MPa). The salt increments are applied on top of
# CoolProp's pure-water value at the actual pressure, which has not been
# validated, so anything above P_MPa is marked as extrapolated too.
_BRINE_FIT_RANGE = {"T_C": (5.0, 80.0), "m_max": 1.0, "m_flag": 1.1, "P_MPa": 1.0}

# Properties withheld (reported as None) whenever salt is present. They have no
# sourced brine correction; reporting the pure-water value, even flagged, was
# rejected. Z is also not meaningful for a liquid. For freshwater they ARE the
# exact pure-water values and are shown.
_WITHHELD_WITH_SALT = ["conductivity_W_mK", "joule_thomson_K_MPa", "Z"]


def _brine_alpha_cp(salt, m, T_C, alpha_w, cp_w):
    """Brine thermal expansivity [1/K] and specific heat [J/kg/K] from the
    Apelblat (2017) supplementary-table fits above."""
    f = _BRINE_FIT[salt]
    lo, hi = _BRINE_FIT_RANGE["T_C"]
    t = (min(max(T_C, lo), hi) - 40.0) / 40.0         # hold at the fitted edge
    d_alpha = (m ** f["p"]) * np.polyval(f["a"][::-1], t) * 1e-4
    phi = np.polyval(f["phi"][::-1], t)
    return alpha_w + d_alpha, (cp_w + m * phi) / (1.0 + m * f["M"])


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
    """Properties of the wetting (water / brine) phase.

    ONLY density and viscosity carry a salt correction, both from
    Laliberte (2009) for NaCl and KI (see `_LAL`): its solution/water ratio
    is multiplied onto CoolProp water at (T, P). `brine_extrapolated` is set
    when T or the mass fraction leaves the range the fit was made on.

    EVERY OTHER property -- speed of sound, compressibility, expansivity, heat
    capacities, γ, thermal conductivity, Joule-Thomson, Z -- is the PURE-WATER
    value from CoolProp at (T, P), for freshwater and brine alike. The UI labels
    that section "Pure water thermodynamic properties". A brine correction of
    κ_T, α and γ fitted to Apelblat (2017) was built and deliberately dropped
    on 2026-10-09; the unused fits are kept above.
    """
    T_C = T_K - 273.15

    # GUARD: the Laliberte correlations are LIQUID correlations. Below the saturation
    # pressure CoolProp hands back steam and these would be applied to it
    # regardless -- at 200 C / 0.1 MPa that turned 0.46 kg/m3 of vapour into a
    # "brine" at 45.1 kg/m3. Report nothing and say why, rather than a
    # plausible-looking wrong number. This also makes `flow` None downstream,
    # so no Ca or Bo is produced from a non-liquid wetting phase.
    if not _water_is_liquid(T_K, P_Pa):
        out = {k: None for k in (*PROP_META, *_INTERNAL_PROPS)}
        out["water_not_liquid"]  = True
        out["mu_water_Pas"]      = np.nan
        return out

    # Pure water: full property set, single state point
    w = _coolprop_props("Water", T_K, P_Pa)
    rho_w = w["density_kg_m3"]
    mu_w  = (w["viscosity_uPas"] / 1e6) if w["viscosity_uPas"] else np.nan
    if rho_w is None:
        rho_w = np.nan

    # Salt correction: Laliberte (2009) ratio applied to CoolProp water.
    T_C_ = T_K - 273.15
    if ki_mol_L > 0:
        st = _brine_state("KI", ki_mol_L, T_C_, rho_w)
    elif S_g_L > 0:
        st = _brine_state("NaCl", S_g_L, T_C_, rho_w)
    else:
        st = dict(rho=rho_w, mu_ratio=1.0, extrapolated=False)
    rho_b = st["rho"] if np.isfinite(rho_w) else np.nan
    mu_b = mu_w * st["mu_ratio"] if np.isfinite(mu_w) else np.nan

    out = dict(w)                       # pure-water values for everything else
    out.pop("phase", None)              # the wetting phase is always liquid here
    out["density_kg_m3"]           = _safe(rho_b)
    out["viscosity_uPas"]          = _safe(mu_b * 1e6)
    out["water_not_liquid"]        = False
    out["brine_extrapolated"]      = bool(st["extrapolated"])
    out["mu_water_Pas"]            = mu_w   # internal, for the viscosity ratio below
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


# UNUSED since 2026-10-14 (Anna: diffusion coefficient not useful for this tool;
# removed from the API response, UI and methods draft). Kept, validated, so that
# restoring it is one `_diffusivity(nw_id, T_K)` call in calculate() plus the
# `diffusivity_*` keys in `interaction` and a UI row.
# Diffusion coefficient of CO2 in PURE WATER: Cadogan, Maitland & Trusler (2014),
# J. Chem. Eng. Data 59, 519-525, doi:10.1021/je401008s. Taylor-dispersion data,
# 298-423 K, 14-50 MPa, standard relative uncertainty 2.3 %; D was found
# essentially independent of pressure. Their modified Stokes-Einstein relation
# (eqs 7a and 8, n_SE = 4):
#     D = k_B T / (n_SE pi eta a),   a = a298 [1 + alpha (T/K - 298)]
# with a298 = 168 pm, alpha = 2.0e-3 for CO2 (2.9 % rms against their data).
# Because D is pressure-independent in the data, eta is the pure-water
# viscosity at a FIXED reference pressure (0.1 MPa, or just above P_sat where
# that is higher): that reproduces their data to 2.7 % rms, whereas using
# eta(T, P) drifts to -9 % at 48 MPa (4.5 % rms) because water's viscosity
# rises with pressure and D does not.
KB = 1.380649e-23
DIFF_T_RANGE_K = (298.15 - 0.5, 423.15 + 0.5)   # measured, with the 0.5 K edge slack used elsewhere
DIFF_P_RANGE_MPa = (14.0, 50.0)                 # measured


def _diffusivity(nw_id, T_K):
    """CO2 diffusion coefficient in pure water [m2/s] (Cadogan et al. 2014)."""
    sol = NW_FLUIDS[nw_id]
    try:
        P_ref = max(1e5, CP.PropsSI("P", "T", T_K, "Q", 0, "Water") * 1.01)
        eta = CP.PropsSI("V", "T", T_K, "P", P_ref, "Water")
    except Exception:
        return None
    a = sol["SE_a298_pm"] * 1e-12 * (1.0 + sol["SE_alpha"] * (T_K - 298.0))
    return _safe(KB * T_K / (4.0 * np.pi * eta * a))


def diffusivity_out_of_range(T_K, P_MPa):
    """True outside the measured T (298-423 K) or p (14-50 MPa) of the
    diffusion data. The correlation itself is pressure-free, so a low pressure
    means the measured pressure-independence is being assumed, not tested."""
    return not (DIFF_T_RANGE_K[0] <= T_K <= DIFF_T_RANGE_K[1]
                and DIFF_P_RANGE_MPa[0] <= P_MPa <= DIFF_P_RANGE_MPa[1])


def _brine_ions(T_C, P_MPa, S_g_L, ki_mol_L):
    """Ion molalities [mol/kg water] for the Mutailipu model: NaCl from g/L,
    KI from mol/L (Laliberte density). KI is mapped to a 1:1 potassium salt
    (iodide is not in the model's anion set) -- an ESTIMATE, see _ift()."""
    if ki_mol_L > 0:
        return _mut.ions_from_salt("KI", _ki_molality(ki_mol_L, T_C, P_MPa))
    if S_g_L > 0:
        return _mut.ions_from_salt("NaCl", _nacl_molality(S_g_L, T_C, P_MPa))
    return {}


def _ift(nw_id, T_C, P_MPa, S_g_L, ki_mol_L=0.0):
    """
    Interfacial tension [mN/m].

    CO2: the correlation of Mutailipu, Song, Yao, Liu & Trusler (2024), Fuel
    357, 129712 -- see mutailipu.py. Validated here against the measurements of
    Pereira et al. (2016) (CO2-water, 78 points: mean +0.2, rms 1.0 mN/m) and
    Pereira et al. (2017) (CO2-NaCl(aq), 42 points: mean 0.0, rms 0.9 mN/m).
    Valid 273-473 K, up to 70 MPa, up to 5 mol/kg salt; outside that,
    ift_out_of_range() is True and range_flags() says how it is estimated.
    KI: an ESTIMATE (reported as such via interaction.ift_is_estimate). The
    model has no iodide, and its brine term depends only on the charge-weighted
    CATION molality, so KI is treated as a 1:1 potassium salt at its own
    molality. Whether iodide behaves like chloride at the CO2/water interface
    is not established by any measurement known to us.
    The earlier implementation interpolated the measured data directly; it is
    kept as _ift_interpolated() as an independent cross-check.

    Returns None when water is not liquid at (T, P) (below its saturation
    pressure it is steam, and there is no CO2-water interface); the solubility
    solver does not converge there either, so any number would be meaningless.
    """
    if not _water_is_liquid(T_C + 273.15, P_MPa * 1e6):
        return None
    if nw_id == "CO2":
        r = _mut.ift(T_C + 273.15, P_MPa, _brine_ions(T_C, P_MPa, S_g_L, ki_mol_L))
        return _safe(r["ift"])
    return _ift_interpolated(nw_id, T_C, P_MPa, S_g_L, ki_mol_L)


def _ift_interpolated(nw_id, T_C, P_MPa, S_g_L, ki_mol_L=0.0):
    """
    Interfacial tension by DIRECT INTERPOLATION of measured data (the app's
    method until 2026-10-27; now a cross-check only).

    CO2: pure-water IFT interpolated from the measured data of Pereira et al.
    (2016); see _PereiraIFT. NaCl: the measured increment of Pereira et al.
    (2017); see _NaClIFTIncrement. Outside the data the temperature (and, for
    the NaCl increment, the pressure) is clamped, NOT extrapolated.
    KI: the NaCl increment at the same molality.
    """
    if nw_id == "CO2":
        ift_fw, _ = _pereira_ift()(T_C, P_MPa)
    else:                       # dormant H2 / N2 grid path
        T_clamp = np.clip(T_C, _IFT_T[0], _IFT_T[-1])
        P_clamp = np.clip(P_MPa, _IFT_P[0], _IFT_P[-1])
        ift_fw = float(_IFT_INTERP[nw_id]([[T_clamp, P_clamp]]).flat[0])
    if ki_mol_L > 0:
        if nw_id == "CO2":
            d, _ = _nacl_increment()(T_C, P_MPa, _ki_molality(ki_mol_L, T_C, P_MPa))
        else:
            d = -0.8 * ki_mol_L     # dormant non-CO2 path (old air/water proxy)
        ift = ift_fw + d
    elif S_g_L > 0:
        if nw_id == "CO2":
            d, _ = _nacl_increment()(T_C, P_MPa, _nacl_molality(S_g_L, T_C, P_MPa))
        else:
            d = 0.04 * S_g_L    # dormant non-CO2 path (old coefficient)
        ift = ift_fw + d
    else:
        ift = ift_fw
    return _safe(ift)


def ift_out_of_range(T_C, P_MPa, S_g_L=0.0, ki_mol_L=0.0):
    """
    True when the state lies outside the validity range of the IFT model
    (Mutailipu 2024: 273-473 K, up to 70 MPa, up to 5 mol/kg salt).
    """
    return bool(_ift_notes(T_C, P_MPa, S_g_L, ki_mol_L))


def _ift_notes(T_C, P_MPa, S_g_L=0.0, ki_mol_L=0.0):
    ions = _brine_ions(T_C, P_MPa, S_g_L, ki_mol_L) if (S_g_L > 0 or ki_mol_L > 0) else {}
    return _mut.validity_notes(T_C + 273.15, P_MPa, ions)


def ift_three_phase_between(T_C, P_MPa, dT, dP):
    """True if the (T +/- dT, P +/- dP) stencil straddles the CO2-H2O three-phase
    line (T < 304.63 K only). The IFT really jumps there (b11 |tau|^1.25), so a
    finite-difference derivative across it is meaningless."""
    signs = set()
    for a in (-dT, 0.0, dT):
        T_K = T_C + a + 273.15
        if T_K >= _mut.T_UCEP:
            continue
        p3 = _mut.p_three_phase_MPa(T_K)
        for b in (-dP, 0.0, dP):
            signs.add((P_MPa + b) >= p3)
    return len(signs) == 2


def ift_branch_between(T_C, P_MPa, dT, dP):
    """
    None, "three_phase" or "branch_point": does the (T +/- dT, P +/- dP) stencil
    cross a place where the IFT correlation is not differentiable?

    * "three_phase": below 304.63 K the IFT jumps across the CO2-H2O three-phase
      line (a real step, Delta-gamma-3).
    * "branch_point": at every temperature the correlation switches from the
      gas-liquid form (x1 <= x3) to the liquid-liquid/dense form (x1 > x3), which
      agree in VALUE at x1 = x3 but not in SLOPE, so a central difference across
      it is a blend of two one-sided slopes. x1 and x3 are salt-free, so the
      branch is located with pure water.
    """
    if ift_three_phase_between(T_C, P_MPa, dT, dP):
        return "three_phase"
    regions = set()
    for a in (-dT, 0.0, dT):
        for b in (-dP, 0.0, dP):
            regions.add(_mut.ift(T_C + a + 273.15, P_MPa + b, {})["region"])
    return "branch_point" if len(regions) > 1 else None


def _ift_out_of_range_interpolated(T_C, P_MPa, S_g_L=0.0, ki_mol_L=0.0):
    """
    Coverage test for _ift_interpolated(): the pure-water data (Pereira 2016:
    25-196 C, pressure reach rising with T), and, with NaCl (or KI, via the NaCl
    data) present, the brine data (Pereira 2017: 25.5-150.2 C, about 3-6 MPa and
    up, at most 2.2 mol/kg).
    """
    ok = _pereira_ift()(T_C, P_MPa)[1]
    if ok and ki_mol_L > 0:
        # KI IFT is flagged as an ESTIMATE everywhere (ift_is_estimate), so its
        # concentration is deliberately NOT tested against the NaCl molality
        # limit (1.98 mol/kg): that would add a "clamped" flag for what is a
        # mild linear extrapolation in molality. T and P coverage still counts,
        # because outside it the increment really is held at an edge value.
        ok = _nacl_increment()(T_C, P_MPa, 1.0)[1]
    elif ok and S_g_L > 0:
        ok = _nacl_increment()(T_C, P_MPa, _nacl_molality(S_g_L, T_C, P_MPa))[1]
    return not ok


# ── Out-of-source-range flags ────────────────────────────────────────────────
# Published T/P limits of the equation of state behind each CoolProp fluid.
_CP_LIMITS = {
    "nw": dict(label="CO₂ density and thermodynamic properties",
               source="Span & Wagner (1996)", T_K=(216.592, 1100.0), P_MPa=800.0),
    "w":  dict(label="Water density and thermodynamic properties",
               source="Wagner & Pruß (2002), IAPWS-95", T_K=(273.16, 1273.0), P_MPa=1000.0),
}


def _brine_range_notes(salt, conc, T_C, P_MPa):
    """Laliberte (2009) fit-range notes, split by property:
    -> {"density": [...], "viscosity": [...]} of dict(outside, estimated_by)."""
    p = _LAL[salt]
    w = _brine_state(salt, conc, T_C, _rho_water_cp(T_C, P_MPa))["w"]
    out = {"density": [], "viscosity": []}
    for prop, (key, wkey) in (("density", ("rho_T", "rho_wmax")),
                              ("viscosity", ("mu_T", "mu_wmax"))):
        lo, hi = p[key]
        if T_C < lo or T_C > hi:
            edge = lo if T_C < lo else hi
            out[prop].append(dict(
                outside=f"T = {T_C:.0f} °C is {'below' if T_C < lo else 'above'} the "
                        f"{salt} {prop} fit range ({lo:g}–{hi:g} °C)",
                estimated_by=f"salt effect (solution/pure-water ratio) held at its "
                             f"{edge:g} °C value and applied to CoolProp pure-water "
                             f"{prop} at the actual T and P"))
        if w > p[wkey]:
            out[prop].append(dict(
                outside=f"{w*100:.1f} wt% {salt} is above the fitted maximum "
                        f"({p[wkey]*100:.0f} wt%)",
                estimated_by="Laliberté model evaluated beyond its fitted composition "
                             "range (extrapolated)"))
    return out


def range_flags(T_C, P_MPa, S_g_L=0.0, ki_mol_L=0.0, water_liquid=True):
    """
    Every quantity whose (T, P, composition) lies outside the range of the
    published source behind it, and HOW the app then estimates it.

    -> list of dict(applies_to, quantity, source, outside, estimated_by), with
    applies_to in {"nw", "w_density", "w_viscosity", "ift"} (where the UI puts
    the red flag). Empty when everything is inside its source's range.
    """
    flags = []

    def add(applies_to, quantity, source, notes):
        for n in notes:
            flags.append(dict(applies_to=applies_to, quantity=quantity,
                              source=source, **n))

    T_K = T_C + 273.15
    lim = _CP_LIMITS["nw"]
    if not (lim["T_K"][0] <= T_K <= lim["T_K"][1]) or P_MPa > lim["P_MPa"]:
        add("nw", lim["label"], lim["source"], [dict(
            outside=f"T = {T_C:.0f} °C, P = {P_MPa:.1f} MPa is outside the equation "
                    f"of state's stated range ({lim['T_K'][0]-273.15:.1f}–"
                    f"{lim['T_K'][1]-273.15:.0f} °C, up to {lim['P_MPa']:.0f} MPa)",
            estimated_by="CoolProp extrapolates the equation of state")])
    if not water_liquid:
        return flags

    lim = _CP_LIMITS["w"]
    if not (lim["T_K"][0] <= T_K <= lim["T_K"][1]) or P_MPa > lim["P_MPa"]:
        add("w_density", lim["label"], lim["source"], [dict(
            outside=f"T = {T_C:.0f} °C, P = {P_MPa:.1f} MPa is outside the equation "
                    f"of state's stated range ({lim['T_K'][0]-273.15:.2f}–"
                    f"{lim['T_K'][1]-273.15:.0f} °C, up to {lim['P_MPa']:.0f} MPa)",
            estimated_by="CoolProp extrapolates the equation of state")])

    salt, conc = (("KI", ki_mol_L) if ki_mol_L > 0
                  else ("NaCl", S_g_L) if S_g_L > 0 else (None, 0.0))
    if salt:
        bn = _brine_range_notes(salt, conc, T_C, P_MPa)
        add("w_density", f"{salt} brine density", "Laliberté (2009)", bn["density"])
        add("w_viscosity", f"{salt} brine viscosity", "Laliberté (2009)", bn["viscosity"])

    # CO2-water / brine IFT: validity range of the Mutailipu et al. (2024) model
    # (273-473 K, up to 70 MPa, up to 5 mol/kg salt).
    add("ift", "CO₂–" + ("brine" if salt else "water") + " interfacial tension",
        "Mutailipu et al. (2024)", _ift_notes(T_C, P_MPa, S_g_L, ki_mol_L))
    return flags


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
SENS_DP = 0.0005   # MPa, the differencing half-step (0.5 kPa = 1/20 of the 10 kPa
                   # reporting unit, the same ratio as SENS_DT to its unit, 1 K)
SENS_P_SCALE = 0.01   # %/MPa -> %/10 kPa
NEAR_CRITICAL_KP = 5.0   # flag when the NW fluid's kappa_T*P >= this (see sensitivity_block)

# Step size matters near the critical point, where the derivative genuinely
# diverges: CO2 density at 32 °C/7.5 MPa reads -63 %/K at a 0.5 K step but
# converges to -32.0 %/K below 0.01 K. Do not coarsen these.
# The same test for pressure (checked 2026-10-27, same state): S_P of CO2 density
# reads +3.48 %/10 kPa at a 50 kPa half-step, +2.48 at 20 kPa, +2.370 at 5 kPa
# and converges to +2.364 below 1 kPa. The half-step was 5 kPa (0.3% off) until
# 2026-10-27, when it was set to 0.5 kPa = 1/20 of the reporting unit, the same
# ratio as the 0.05 K step to its 1 K unit (0.9% off at this state). Away from the
# critical point both steps are converged to 4+ figures. The half-steps are
# independent of the reporting units (%/K, %/10 kPa).

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

    IFT honesty: the IFT derivatives are withheld (None, `clamped` set, with a
    `reason`) when the finite-difference stencil (a) leaves the model's validity
    range, where the value is an extrapolation, (b) straddles the CO2-H2O
    three-phase line below 304.63 K, where the IFT genuinely jumps, or (c)
    straddles the x1 = x3 branch point (at every temperature), where the
    correlation's slope is discontinuous (see ift_branch_between). In all cases
    a difference quotient is an artefact, not a sensitivity.
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

    # IFT, and whether its derivatives are real (any of the stencil points
    # outside the measured data makes them an artefact of the clamp)
    sig_f = lambda T, P: _ift(nw_id, T, P, S_g_L, ki_mol_L=ki_mol_L)
    sigT, sigP = _rel_sens(sig_f, T_C, P_MPa)
    sigma_out = any(ift_out_of_range(T_C + dT, P_MPa + dP, S_g_L, ki_mol_L)
                    for dT in (-SENS_DT, 0.0, SENS_DT) for dP in (-SENS_DP, 0.0, SENS_DP))
    if not _water_is_liquid(T_C + 273.15, P_MPa * 1e6):
        # water is steam: no IFT at all (see _ift), so nothing to flag -- a plain dash
        sigma_out, sigma_kink, sigT, sigP = False, None, None, None
    else:
        sigma_kink = None if sigma_out else ift_branch_between(T_C, P_MPa, SENS_DT, SENS_DP)
    sigma_clamped = sigma_out or bool(sigma_kink)
    sigma_reason = "out_of_range" if sigma_out else sigma_kink
    if sigma_clamped:
        sigT = sigP = None

    # Near the critical point (and along the Widom line above it) the nonwetting
    # fluid's derivatives genuinely diverge. Measure of closeness: the isothermal
    # compressibility relative to an ideal gas, kappa_T * P (= 1 for an ideal gas,
    # 2-4 for a dense supercritical fluid, 15+ within a few K and a few hundred kPa
    # of the CO2 critical point). Chosen from a 121-state map (T 25-80 C, P 5-20 MPa)
    # of CO2 density, compressibility and viscosity: at kappa_T*P >= 5 it flags
    # 7 states and every state whose current differencing steps are >1% off the
    # converged derivative is among them.
    kP = None
    try:
        kap = _nw_props(nw_id, T_C + 273.15, P_MPa * 1e6)["compressibility_1_GPa"]
        kP = _safe(kap * 1e-3 * P_MPa) if kap is not None else None   # 1/GPa -> 1/MPa
    except Exception:
        pass
    near_critical = {"flag": bool(kP is not None and kP >= NEAR_CRITICAL_KP),
                     "kappa_T_P": kP, "threshold": NEAR_CRITICAL_KP}

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
        "ift":  {"S_T": sigT, "S_P": sigP, "clamped": bool(sigma_clamped),
                 "reason": sigma_reason,
                 "estimate": bool(ki_mol_L > 0)},
        "drho": {"S_T": dT_, "S_P": dP_},
        "near_critical": near_critical,
        "derived": derived,
        "note_fixed_Q": True,
    }


# ── Field CCS sites on the CO2 phase diagram ─────────────────────────────────
#
# ALL conditions here come from ONE compilation, Imre, Ramboz, Deiters & Kraska
# (2015), Environ. Earth Sci. 73, 4373-4384 (doi:10.1007/s12665-014-3716-5),
# which in turn quotes the original references recorded in each entry. A site is
# listed only if that paper reports it. (Until 2026-10-21 this table mixed values
# from six different sources, four of them second-hand; those sites were dropped.)
#
# Sites are RANGES, not points: each carries one or more "regions", a box
# T_min..T_max x P_min..P_max in degC and MPa. A region with equal min and max
# on an axis is a line, and on both axes a point. They are drawn as boxes.
#
# These are the P-T conditions Imre et al. report for each site. For Nagaoka
# they are the WELL HEAD and WELL BOTTOM conditions, not a reservoir range;
# Imre et al. state that the storage P-T conditions there are not available.
#
# To add a site, append a dict with the same keys (and at least one region).
# "label_at" is the [T, P] the name label points to and "label_px" its [dx, dy]
# offset in pixels (dy positive is DOWN). They are hand-set because the sites
# overlap (Nagaoka's well head sits inside Sleipner's box), so automatic
# placement collides.
CCS_SITES = [
    {
        "key": "sleipner", "name": "Sleipner", "location": "North Sea",
        "formation": "Utsira Formation",
        "regions": [
            {"label": "Storage conditions in the Utsira Fm",
             "T_min": 25.0, "T_max": 42.0, "P_min": 8.0, "P_max": 11.0,
             "label_at": [25.0, 11.0], "label_px": [-25, -30]},
        ],
        "reference": "Imre et al. (2015), Environ. Earth Sci. 73, 4373–4384",
        "original": "Baklid et al. (1996); Bickle et al. (2007); Arts et al. "
                    "(2008); Ringrose & Eiken (2011)",
        "note": "Imre et al. note these conditions lie close to the Widom region, "
                "and that ~2 mol% CH4 in the injected stream could shift the "
                "compressibility maximum by ~5 K.",
    },
    {
        "key": "snohvit", "name": "Snøhvit", "location": "Barents Sea",
        "formation": "Sandy reservoir, ~2.6 km below sea level",
        "regions": [
            {"label": "Initial (2008, ~29 MPa) to 2011 (~39 MPa)",
             "T_min": 98.0, "T_max": 100.0, "P_min": 29.0, "P_max": 39.0,
             "label_at": [100.0, 34.0], "label_px": [40, 0]},
        ],
        "reference": "Imre et al. (2015), Environ. Earth Sci. 73, 4373–4384",
        "original": "Hansen et al. (2011); Hansen et al. (2013)",
        "note": "Initial conditions ~29 MPa and 98–100 °C in 2008; reservoir "
                "pressure rose to ~39 MPa by 2011 under near-continuous "
                "injection because of limited permeability. Drawn as one box "
                "spanning that pressure rise.",
    },
    {
        "key": "nagaoka", "name": "Nagaoka", "location": "Japan",
        "formation": "—",
        "regions": [
            {"label": "Well head",
             "T_min": 32.0, "T_max": 32.0, "P_min": 7.0, "P_max": 11.0,
             "label_at": [32.0, 11.0], "label_px": [50, -40]},
            {"label": "Well bottom",
             "T_min": 48.0, "T_max": 48.0, "P_min": 19.0, "P_max": 19.0,
             "label_at": [48.0, 19.0], "label_px": [40, 0]},
        ],
        "reference": "Imre et al. (2015), Environ. Earth Sci. 73, 4373–4384",
        "original": "Kikuta et al. (2005)",
        "note": "Well-head (7–11 MPa, 32 °C) and well-bottom (19 MPa, 48 °C) "
                "conditions, not a reservoir range. Imre et al. state that "
                "details of the storage P–T conditions and their evolution "
                "are lacking.",
    },
    {
        "key": "ketzin", "name": "Ketzin", "location": "Germany",
        "formation": "—",
        "regions": [
            {"label": "Initial reservoir conditions",
             "T_min": 33.5, "T_max": 34.0, "P_min": 6.1, "P_max": 6.2,
             "label_at": [33.75, 6.15], "label_px": [55, 35]},
        ],
        "reference": "Imre et al. (2015), Environ. Earth Sci. 73, 4373–4384",
        "original": "Kempka & Kühn (2013); Loizzo et al. (2013)",
        "note": "The box spans the two sources' initial conditions. Kempka & Kühn: "
                "62 bar at 639.5 m in well Ktzi 201/2007, temperature taken as "
                "constant at 34 °C through the reservoir. Loizzo et al.: 33.5 °C "
                "and 6.1 MPa at 630 m. Loizzo et al. also report the Ktzi 200 "
                "wellhead pressure rising and stabilising near 5.1 MPa after CO₂ "
                "arrived in July 2008; no wellhead temperature is given, so that "
                "is not plotted. Pressure is below the CO₂ critical pressure.",
    },
]

# Sites that Imre et al. report but whose values are not yet entered. They are
# listed in the UI so the gap is visible, and nothing is estimated. (Empty
# since Ketzin was entered 2026-10-21.)
CCS_SITES_PENDING = []


def _ccs_state(T_C, P_MPa, Tc_C, Pc_MPa):
    """(CO2 density [kg/m3] or None, regime label, supercritical?) at one state.

    The regime comes from explicit comparison with T_c and P_c, not from
    CoolProp's phase index, which calls T > T_c, P < P_c "supercritical" even
    though the density there is gas-like.
    """
    try:
        rho = float(CP.PropsSI("D", "T", T_C + 273.15, "P", P_MPa * 1e6, "CO2"))
    except Exception:
        rho = None
    if T_C >= Tc_C:
        sc = P_MPa >= Pc_MPa
        regime = "Supercritical" if sc else "Gas-like (above T<sub>c</sub>, below P<sub>c</sub>)"
    else:
        sc = False
        try:
            Psat = CP.PropsSI("P", "T", T_C + 273.15, "Q", 0, "CO2") / 1e6
            regime = "Liquid" if P_MPa >= Psat else "Vapour"
        except Exception:
            regime = "—"
    return rho, regime, bool(sc)


def co2_phase_diagram(T_min_C=-20.0, T_max_C=140.0,
                      P_min_MPa=0.5, P_max_MPa=40.0,
                      iso_densities=(200, 400, 600, 700, 800, 900)):
    """
    CO2 phase diagram in (T, P) with field CCS sites placed on it as
    P-T ranges (boxes), per Imre et al. (2015).

    Returns the vapour-pressure (saturation) curve, the critical and triple
    points, exact iso-density contours, and each site annotated with the CO2
    density range and regimes CoolProp gives over its reported P-T range.

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

    # Place the sites and evaluate CO2 over each region. A region's density is
    # the min-max over a grid spanning the box; its regime is every label met.
    sites = []
    for s in CCS_SITES:
        entry = dict(s)
        regions = []
        for r in s["regions"]:
            Ts = np.linspace(r["T_min"], r["T_max"], 9)
            Ps = np.linspace(r["P_min"], r["P_max"], 9)
            rhos, labels, scs = [], [], []
            for T_ in Ts:
                for P_ in Ps:
                    rho, reg, sc = _ccs_state(float(T_), float(P_), Tc_C, Pc_MPa)
                    if rho is not None:
                        rhos.append(rho)
                    if reg not in labels:
                        labels.append(reg)
                    scs.append(sc)
            rr = dict(r)
            rr["T_mid"] = 0.5 * (r["T_min"] + r["T_max"])
            rr["P_mid"] = 0.5 * (r["P_min"] + r["P_max"])
            rr["is_point"] = bool(r["T_min"] == r["T_max"] and r["P_min"] == r["P_max"])
            rr["rho_min"] = float(min(rhos)) if rhos else None
            rr["rho_max"] = float(max(rhos)) if rhos else None
            rr["regimes"] = labels
            rr["all_supercritical"] = bool(all(scs))
            regions.append(rr)
        entry["regions"] = regions
        all_rho = [v for r in regions for v in (r["rho_min"], r["rho_max"]) if v is not None]
        entry["rho_min"] = float(min(all_rho)) if all_rho else None
        entry["rho_max"] = float(max(all_rho)) if all_rho else None
        regs = []
        for r in regions:
            for g in r["regimes"]:
                if g not in regs:
                    regs.append(g)
        entry["regimes"] = regs
        # Fully supercritical only if EVERY sampled state in EVERY region is
        entry["supercritical"] = bool(all(r["all_supercritical"] for r in regions))
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
    # Solubility removed from the response 2026-10-08 (not needed for this
    # tool). The whole validated solubility path (_solubility, Duan & Sun,
    # Duan 1992 EOS, Henry+KK) is retained below; restoring it is one
    # _solubility() call here plus a "solubility" key in `interaction` and
    # the three UI rows.
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
        "ift_out_of_range": ift_out_of_range(T_C, P_MPa, S, ki_mol_L=ki),
        "ift_is_estimate":  bool(ki > 0),   # KI: no CO2/KI IFT data; see _ift()
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
    # diffusivity); the wetting phase in the flow calculation is the brine,
    # whose viscosity differs once salt is present.
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
        "range_flags": range_flags(T_C, P_MPa, S, ki_mol_L=ki,
                                   water_liquid=not w_out.get("water_not_liquid", False)),
    }
