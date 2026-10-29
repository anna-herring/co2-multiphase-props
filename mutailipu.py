"""
CO2-water / CO2-brine interfacial tension and mutual solubility
===============================================================
Mutailipu, M.; Song, Y.; Yao, Q.; Liu, Y.; Trusler, J.P.M. (2024)
"Solubility and interfacial tension models for CO2-brine systems under CO2
geological storage conditions", Fuel 357, 129712, doi:10.1016/j.fuel.2023.129712.

Equation and table numbers below are the paper's.

  * Solubility (Sec. 2): gamma-phi approach. Aqueous phase = modified Margules
    activity model (Eqs 20-26, Table 8); non-aqueous phase = Peng-Robinson EoS
    with the Panagiotopoulos-Reid combining rule (Eqs 17-19, A-1..A-7).
    K1, K2 from the polynomial forms of Fernandez-Prini / IAPWS-95 /
    McBride-Wright et al. (2015) (Eqs 13-16, Table 4).
  * IFT (Sec. 3): gamma = gamma_b - dgamma(tau, x1), with x1 the CO2 mole
    fraction in the aqueous phase from the solubility model (Eqs 30-42,
    Table 11).

Stated validity (Sec. 4): 273-473 K; p <= 70 MPa for the IFT (100 MPa for the
solubility); total salt molality up to 5 mol/kg; cations Na+, K+, Mg2+, Ca2+
with Cl-, SO4(2-), HCO3-.

Three departures from the printed text, each found by testing against the
paper's own data (details in METHODS_DRAFT.md section 4):
  1. Eq A-4 reads "0.7464 + 1.54226 w - 0.26992 w" in the PDF text; the standard
     Peng-Robinson kappa 0.37464 + 1.54226 w - 0.26992 w^2 is used.
  2. Eq 38: b6 multiplies only the first term of the bracket (see _dgamma).
  3. For brines the IFT uses the SALT-FREE x1 (see ift()); the literal salted x1
     over-predicts measured brine IFTs by ~9 mN/m. CONFIRMED by the paper's
     Supplementary Information (mmc1.docx): its Tables A.21, A.27, A.28 list, for
     brine rows, an x(CO2) equal to the salt-free value and gamma_mod computed
     from it.
Validation: against measurements, Pereira (2016) CO2-water, 78 points, rms
1.0 mN/m; Pereira (2017) CO2-NaCl(aq), 42 points, rms 0.9 mN/m. Against the
authors' OWN calculated values in the SI (Tables A.9-A.28): 921 CO2-water points
x1 rms 1e-5 and IFT rms 0.036 mN/m (max 0.24); 453 brine points x1 rms 0.000
and IFT rms 0.03-0.04 mN/m -- i.e. this module reproduces their model to the
rounding of the published tables.
"""
from __future__ import annotations

import math

import numpy as np

try:                                       # psat of water for the Poynting term
    import CoolProp.CoolProp as _CP
except Exception:                          # pragma: no cover
    _CP = None

R = 8.314462618            # J/(mol K)
T_UCEP = 304.63            # K
M2 = 0.018015              # kg/mol, water
TC_WATER = 647.096         # K

# ── Table 4: polynomial coefficients in tau, i = 0..4 ─────────────────────────
_A = (5.2677, 7.1625, -16.3072, 17.4946, -9.0997)         # ln(H12/MPa)
_B = (-5.3804, 17.3049, -19.6272, 17.4627, -8.4245)       # ln(phi2sat p2sat/MPa)
_C = (18.1040, 1.6946, 7.3717, -6.9088, 7.5670)           # V2sat / (cm3/mol)
_D = (34.3287, 13.7101, 30.4981, 0.8212, -2.3406)         # V1inf / (cm3/mol)

# ── Table 8 ───────────────────────────────────────────────────────────────────
_MARG = (-1.23, -8.85, 17.91)                              # A0, A1, A2 (Eq 22)
_K12 = (0.1070, -0.2447)                                   # k12,0 k12,1
_K21 = (0.2606, -0.0284)
# delta_i,0 and delta_i,1: 1=Na+ 2=K+ 3=Mg2+ 4=Ca2+ 5=SO4 6=HCO3 (Eq 26)
_DELTA = {1: (1.0310, -0.2489), 2: (0.7149, 0.2475), 3: (1.9327, 0.1267),
          4: (2.0390, -0.2093), 5: (0.1688, -0.4291), 6: (0.0899, -0.7102)}

# ── Table 11 ──────────────────────────────────────────────────────────────────
_IFT = dict(b1=1546.4, b2=-1439.2, b3=4.0159, b4=-63.118, b5=4414244.0,
            b6=-430.7, b7=-2250.9, b8=-27889.0, b9=-966.9, b10=3169.0,
            b11=124.5, b12=14.0, b13=1.608, b14=1.072)

# Peng-Robinson pure-component constants (text of Sec. 2 / standard values)
_PR = {1: dict(Tc=304.13, pc=7.377e6, w=0.2239),           # CO2
       2: dict(Tc=647.096, pc=22.064e6, w=0.3449)}         # H2O

# Stated range of validity (Sec. 4)
T_MIN_K, T_MAX_K = 273.15, 473.15
P_MAX_IFT_MPa, P_MAX_SOL_MPa = 70.0, 100.0
M_SALT_MAX = 5.0


def _poly(c, tau):
    return sum(ci * tau ** i for i, ci in enumerate(c))


def tau_of(T_K):
    return T_K / T_UCEP - 1.0


def p_three_phase_MPa(T_K):
    """Eq 33 (Wendland et al.): CO2-H2O three-phase pressure."""
    t = T_K
    return math.exp(3.99263 - 1335.97 / t + 0.007865 * t)


def water_surface_tension_mN_m(T_K):
    """Eq 31 (IAPWS R1-76)."""
    x = 1.0 - T_K / TC_WATER
    return 235.8 * x ** 1.256 * (1.0 - 0.625 * x)


def _psat_water_MPa(T_K):
    """Saturation pressure of water (IAPWS-95 via CoolProp), for the Poynting term."""
    return _CP.PropsSI("P", "T", T_K, "Q", 0, "Water") / 1e6


# ── Brine description ─────────────────────────────────────────────────────────

def ions_from_salt(salt: str, molality: float) -> dict:
    """Ion molalities [mol/kg water] for a single salt of the given molality.
    KI is mapped to KCl (iodide is not in the model's anion set) -- an ESTIMATE."""
    m = float(molality)
    salt = salt.upper()
    if salt == "NACL":
        return dict(Na=m, Cl=m)
    if salt in ("KCL", "KI"):
        return dict(K=m, Cl=m)
    if salt == "CACL2":
        return dict(Ca=m, Cl=2 * m)
    if salt == "MGCL2":
        return dict(Mg=m, Cl=2 * m)
    raise ValueError(f"unsupported salt {salt}")


def _m_plus(ions: dict) -> float:
    """Eq 42: charge-weighted cation molality."""
    return (ions.get("Na", 0.0) + ions.get("K", 0.0)
            + 2.0 * ions.get("Mg", 0.0) + 2.0 * ions.get("Ca", 0.0))


# ── Peng-Robinson with the Panagiotopoulos-Reid combining rule ────────────────

def _pr_pure(T_K):
    out = {}
    for i, p in _PR.items():
        w = p["w"]
        kap = 0.37464 + 1.54226 * w - 0.26992 * w * w
        alpha = (1.0 + kap * (1.0 - math.sqrt(T_K / p["Tc"]))) ** 2
        out[i] = (0.45724 * (R * p["Tc"]) ** 2 / p["pc"] * alpha,
                  0.07780 * R * p["Tc"] / p["pc"])
    return out                                   # {i: (a_i, b_i)}  SI


def _ln_phi_roots(T_K, p_Pa, y1, tau):
    """All real PR roots for the non-aqueous phase -> list of (Z, ln phi1, ln phi2)."""
    y = {1: y1, 2: 1.0 - y1}
    ab = _pr_pure(T_K)
    a_i = {i: ab[i][0] for i in (1, 2)}
    b_i = {i: ab[i][1] for i in (1, 2)}
    k = {(1, 2): _K12[0] + _K12[1] * tau, (2, 1): _K21[0] + _K21[1] * tau}
    c12 = math.sqrt(a_i[1] * a_i[2])

    def a_ij(i, j):                              # Eq 17
        if i == j:
            return a_i[i]
        return c12 * (1.0 - k[(i, j)] + (k[(i, j)] - k[(j, i)]) * y[i])

    a = sum(y[i] * y[j] * a_ij(i, j) for i in (1, 2) for j in (1, 2))     # A-6
    b = y[1] * b_i[1] + y[2] * b_i[2]                                      # A-7
    A = a * p_Pa / (R * T_K) ** 2
    B = b * p_Pa / (R * T_K)

    # (1/n) d(n^2 a)/d n_k, analytic for the composition-dependent a_ij.
    # d_ij = c_ij (k_ij - k_ji);  G = sum_ij d_ij y_i^2 y_j
    d = {(i, j): (0.0 if i == j else c12 * (k[(i, j)] - k[(j, i)]))
         for i in (1, 2) for j in (1, 2)}
    G = sum(d[(i, j)] * y[i] ** 2 * y[j] for i in (1, 2) for j in (1, 2))

    def dna(kk):
        s = 0.0
        for j in (1, 2):
            if j != kk:
                s += y[j] * c12 * (1.0 - k[(kk, j)])           # a_kj (1-kkj part)
            else:
                s += y[j] * a_i[kk]
        for i in (1, 2):
            if i != kk:
                s += y[i] * c12 * (1.0 - k[(i, kk)])           # a_ik (1-kik part)
            else:
                s += y[i] * a_i[kk]
        s += 2.0 * y[kk] * sum(d[(kk, j)] * y[j] for j in (1, 2))
        s += sum(d[(i, kk)] * y[i] ** 2 for i in (1, 2))
        s -= G
        return s

    # cubic A-5
    coeff = [1.0, -(1.0 - B), A - 3.0 * B * B - 2.0 * B, -(A * B - B * B - B ** 3)]
    roots = [r.real for r in np.roots(coeff) if abs(r.imag) < 1e-9 and r.real > B]
    out = []
    s2 = math.sqrt(2.0)
    for Z in sorted(roots):
        lnphi = {}
        for kk in (1, 2):
            Bk = b_i[kk] * p_Pa / (R * T_K)
            lnphi[kk] = (Bk / B * (Z - 1.0) - math.log(Z - B)
                         + A / (2.0 * s2 * B) * (Bk / B - dna(kk) / a)
                         * math.log((Z + (1.0 + s2) * B) / (Z + (1.0 - s2) * B)))
        out.append((Z, lnphi[1], lnphi[2]))
    return out


def _pick_root(roots, y1, phase):
    """phase: 'vapor' (largest Z), 'liquid' (smallest Z) or 'auto' (min Gibbs)."""
    if len(roots) == 1 or phase == "any":
        return roots[0] if len(roots) == 1 else min(
            roots, key=lambda r: y1 * r[1] + (1 - y1) * r[2])
    if phase == "vapor":
        return roots[-1]
    if phase == "liquid":
        return roots[0]
    return min(roots, key=lambda r: y1 * r[1] + (1 - y1) * r[2])


# ── Activity model ────────────────────────────────────────────────────────────

def _ln_gamma(x1, tau, ions):
    A = _MARG[0] + _MARG[1] * tau + _MARG[2] * tau * tau          # Eq 22
    x2 = 1.0 - x1
    ms = sum(ions.values())
    lam = 0.06754 * (tau + 1) + 0.00353 / (tau + 1) + 0.02854 / (tau + 1) ** 2   # 24
    xi = 0.00396 * (tau + 1) - 0.06605 / (tau + 1) + 0.05667 / (tau + 1) ** 2    # 25
    dl = {i: _DELTA[i][0] + _DELTA[i][1] * tau for i in _DELTA}                    # 26
    mNa, mK = ions.get("Na", 0.0), ions.get("K", 0.0)
    mMg, mCa = ions.get("Mg", 0.0), ions.get("Ca", 0.0)
    mCl, mSO4, mHCO3 = ions.get("Cl", 0.0), ions.get("SO4", 0.0), ions.get("HCO3", 0.0)
    lng1p = (math.log(1.0 + ms * M2)
             + 2.0 * lam * (dl[1] * mNa + dl[2] * mK + dl[3] * mMg + dl[4] * mCa)
             + xi * mCl * (mNa + mK + mMg + mCa)
             + dl[5] * mSO4 + dl[6] * mHCO3)                                       # 23
    ln_g1 = 2.0 * A * x1 * x2 * x2 + lng1p                                         # 20
    ln_g2 = A * (1.0 - 2.0 * x2) * x1 * x1                                         # 21
    return ln_g1, ln_g2


# ── Mutual solubility ─────────────────────────────────────────────────────────

def solubility(T_K, p_MPa, ions=None, phase="auto", tol=1e-11, maxit=400):
    """
    Mutual solubility of CO2 and H2O (+ salt in the aqueous phase only).

    -> dict(x1, y2, m1, converged)   x1: CO2 mole fraction in the aqueous phase
       (Eq 12 convention, ions counted in the total), y2: H2O mole fraction in
       the non-aqueous phase, m1: CO2 molality [mol/kg].
    """
    ions = ions or {}
    tau = tau_of(T_K)
    ms = sum(ions.values())
    p_Pa = p_MPa * 1e6
    psat = _psat_water_MPa(T_K)
    K1 = math.exp(_poly(_A, tau)) * math.exp(
        _poly(_D, tau) * (p_MPa - psat) / (R * T_K))                               # 6
    K2 = math.exp(_poly(_B, tau)) * math.exp(
        _poly(_C, tau) * (p_MPa - psat) / (R * T_K))                               # 7

    y2, x1 = 0.01, 0.01
    ok = False
    for it in range(maxit):
        roots = _ln_phi_roots(T_K, p_Pa, 1.0 - y2, tau)
        if not roots:
            break
        Z, lp1, lp2 = _pick_root(roots, 1.0 - y2, phase)
        ln_g1, ln_g2 = _ln_gamma(x1, tau, ions)
        Aq = math.exp(ln_g2) * K2 / (math.exp(lp2) * p_MPa)                        # 10
        Bq = math.exp(lp1) * p_MPa / (math.exp(ln_g1) * K1)                        # 11
        y2n = (1.0 - Bq) / ((1.0 / Aq - Bq) * (1.0 + ms * M2) + Bq * ms * M2)      # 8
        x1n = Bq * (1.0 - y2n)                                                      # 9
        y2n = min(max(y2n, 1e-9), 0.999)
        x1n = min(max(x1n, 1e-12), 0.5)
        if abs(y2n - y2) < tol and abs(x1n - x1) < tol:
            y2, x1, ok = y2n, x1n, True
            break
        y2 += 0.6 * (y2n - y2)                                                      # damped
        x1 += 0.6 * (x1n - x1)
    m1 = x1 * (ms + 1.0 / M2) / (1.0 - x1)                                         # 12
    return dict(x1=x1, y2=y2, m1=m1, converged=ok, Z=Z if roots else None)


# ── Interfacial tension ───────────────────────────────────────────────────────

def _damping(tau):                                   # Eq 39
    return 1.0 if tau <= 0 else math.exp(-_IFT["b12"] * tau)


def _dgamma(tau, x1, x3):
    b = _IFT
    F = _damping(tau)
    g1 = (b["b1"] + b["b2"] * tau * math.tanh(b["b3"] * tau + b["b4"] * x1
                                              + b["b5"] * x1 ** 4)) * x1          # 34
    if x1 <= x3:                                                                    # 35
        g2 = b["b6"] * (1.0 + (b["b7"] + b["b8"] * tau) * x1 ** 2) * x1 * F         # 37
        return g1 + g2
    g3 = b["b11"] * abs(tau) ** 1.25 if tau <= 0 else 0.0                           # 40
    # Eq 38. As typeset, b6 appears to bracket both terms; fitting the paper's own
    # IFT data shows it multiplies ONLY the first term (the slope term
    # (b9 + b10 tau) ~ -880..-1030 is already in mN/m per unit x1). With b6 on
    # both terms the dense branch is off by up to 1400 mN/m.
    g4 = (b["b6"] * (1.0 + (b["b7"] + b["b8"] * tau) * x3 ** 2) * x3
          + (b["b9"] + b["b10"] * tau) * (x1 - x3)) * F                              # 38
    return g1 + g3 + g4                                                             # 36


def ift(T_K, p_MPa, ions=None, salt_in_x1=False):
    """
    CO2 / water-or-brine interfacial tension [mN/m] (Eqs 30-42).

    gamma = gamma_b - dgamma(tau, x1), gamma_b = gamma_w + (b13 + b14 tau) m+.

    x1 is the CO2 mole fraction from the SALT-FREE solubility at (T, p), not the
    salted one. The paper's text says x1 is "related to ion molalities", but
    (a) with the salted x1 the model over-predicts the 42 CO2 + NaCl(aq)
    measurements of Pereira et al. (2017) by +9.0 mN/m on average (rms 10.1),
    because salting-out shrinks x1 and with it dgamma; with the salt-free x1 they
    are reproduced to mean -0.03, rms 0.89 mN/m; and (b) the paper's own
    Supplementary Information tabulates, for brine rows, the salt-free x(CO2)
    and the gamma_mod computed from it (reproduced here to rms 0.03-0.04 mN/m
    on 453 brine points). The brine effect enters only through gamma_b (Eq 41).
    salt_in_x1=True gives the salted reading, which the SI contradicts.

    -> dict(ift, x1, x3, p3, region, gamma_b, converged)
    """
    ions = ions or {}
    tau = tau_of(T_K)
    gw = water_surface_tension_mN_m(T_K)
    gb = gw + (_IFT["b13"] + _IFT["b14"] * tau) * _m_plus(ions)                    # 41
    p3 = p_three_phase_MPa(T_K)
    sol_ions = ions if salt_in_x1 else {}

    if tau < 0:                         # sub-critical: vapour below p3, liquid above
        phase = "vapor" if p_MPa < p3 else "liquid"
        # x3: solubility at the three-phase pressure, same phase as the request
        sol3 = solubility(T_K, p3, sol_ions, phase=phase)
    else:
        phase = "auto"
        sol3 = solubility(T_K, p3, sol_ions, phase="auto")
    sol = solubility(T_K, p_MPa, sol_ions, phase=phase)
    x1, x3 = sol["x1"], sol3["x1"]
    gam = gb - _dgamma(tau, x1, x3)
    return dict(ift=gam, x1=x1, x3=x3, p3=p3,
                region="gas-liquid" if x1 <= x3 else "liquid-liquid/dense",
                gamma_b=gb, converged=sol["converged"] and sol3["converged"])


# ── Validity ──────────────────────────────────────────────────────────────────

def validity_notes(T_K, p_MPa, ions=None):
    """List of dict(outside=..., estimated_by=...) for conditions outside the
    paper's stated range. Empty when inside."""
    ions = ions or {}
    notes = []
    if T_K < T_MIN_K - 1e-9 or T_K > T_MAX_K + 1e-9:
        edge = T_MIN_K if T_K < T_MIN_K else T_MAX_K
        notes.append(dict(
            outside=f"T = {T_K-273.15:.0f} °C is outside the model's range "
                    f"({T_MIN_K-273.15:.0f}–{T_MAX_K-273.15:.0f} °C)",
            estimated_by=f"the correlation is evaluated beyond its fitted range "
                         f"(extrapolated from {edge-273.15:.0f} °C); not recommended"))
    if p_MPa > P_MAX_IFT_MPa:
        notes.append(dict(
            outside=f"P = {p_MPa:.0f} MPa is above the IFT model's limit "
                    f"({P_MAX_IFT_MPa:.0f} MPa)",
            estimated_by="the correlation is evaluated beyond its fitted range "
                         "(extrapolated); not recommended"))
    ms_salt = sum(ions.values()) / 2.0 if ions else 0.0     # ~ salt molality (1:1)
    if ms_salt > M_SALT_MAX:
        notes.append(dict(
            outside=f"salt molality {ms_salt:.1f} mol/kg is above the model's "
                    f"limit ({M_SALT_MAX:g} mol/kg)",
            estimated_by="the salting-out and surface-tension terms are extrapolated"))
    return notes
