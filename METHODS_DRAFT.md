# Methods — Fluid properties and displacement regime characterization

*Draft methods section for the Multiphase Flow Fluid Properties calculator
(https://multiphase-flow-props.onrender.com/). Describes the `co2-only` branch
as of 2026-10-29. Bracketed notes [like this] flag decisions or checks for the
author; delete them before submission.*

---

## 1. Overview

Fluid properties, interfacial properties, and displacement regime indicators
were computed with an open-source calculator (FastAPI/Python) that evaluates
the CO₂–brine fluid pair at user-specified temperature and pressure
(10–200 °C, 0.1–40 MPa) and, given core-flood parameters (flow rate, core
diameter, porosity, characteristic pore length, contact angle), returns the
capillary number, Bond number, and viscosity ratio together with the
corresponding positions on the Lenormand et al. (1988) drainage regime diagram
and a capillary–viscous–gravity force-balance diagram. [Add citation/DOI or
repository URL for the tool itself if it will be released with the paper.]

## 2. Pure-fluid thermodynamic properties

All pure-fluid properties (density ρ, viscosity μ, speed of sound c,
isothermal compressibility κ_T, isobaric and isochoric heat capacities c_p and
c_v, thermal expansivity, thermal conductivity, Joule–Thomson coefficient, and
compressibility factor Z) were obtained from the reference Helmholtz-energy
equations of state implemented in CoolProp (Bell et al., 2014): Span & Wagner
(1996) for CO₂ and IAPWS-95 (Wagner & Pruß, 2002) for water, with the
associated reference correlations for viscosity and thermal conductivity
[CoolProp uses Fenghour et al. (1998) for CO₂ viscosity and Huber et al.
(2009) for water viscosity — cite if the journal wants transport-property
provenance]. Derived quantities follow exact thermodynamic identities:

- isentropic compressibility κ_s = κ_T / γ, where γ = c_p/c_v
- speed of sound c = √( γ / (ρ κ_T) )
- Joule–Thomson coefficient μ_JT = (∂T/∂P)_H

Each state point is evaluated through a single equation-of-state call so that
all reported properties correspond to the same thermodynamic state.

## 3. Wetting-phase (water and brine) properties

Only the **density and viscosity** of the wetting phase are corrected for
salinity. Every other wetting-phase thermodynamic property — speed of sound,
isothermal compressibility, thermal expansivity, thermal conductivity, the
Joule–Thomson coefficient, and the compressibility factor — is reported at its
**pure-water** value from CoolProp, for freshwater and brines alike, and is
labelled as such in the interface. These properties are therefore not brine
properties. For scale, at 25 °C the isothermal compressibility of 1 mol kg⁻¹
NaCl is about 10% below that of water and its speed of sound about 4% (≈60
m s⁻¹) above it; over 5–80 °C these differences span −8% to −13% and +3% to
+5%, and are roughly half as large at 0.5 mol kg⁻¹ (derived from the tables of
Apelblat, 2017). [The source does not state a pressure; atmospheric is
assumed, and the size of the effect at higher pressure is not established
here. The speed of sound is derived from the tabulated density, compressibility,
expansivity and heat capacity, not tabulated directly.]

**Density and viscosity (NaCl and KI).** Laliberté (2009), one model for
both salts. With w the solute mass fraction and w_w = 1 − w, the solution
density follows from the water density and the solute's apparent density,

    1/ρ_m = w_w/ρ_w + w/ρ_app,
    ρ_app = [c₀ w + c₁] exp[10⁻⁶ (t + c₄)²] / [w + c₂ + c₃ t],

and the viscosity from a mass-fraction-weighted geometric mean of the water
and solute viscosities,

    η_m = η_w^w_w · η_s^w,
    η_s = exp[(V₁ w^V₂ + V₃)/(V₄ t + 1)] / (V₅ w^V₆ + 1),

with t in °C. The fitted constants c₀–c₄ and V₁–V₆ are those tabulated in the
Supporting Information of Laliberté (2009). The fits are at roughly
atmospheric pressure, so the model is used as a solution-to-water *ratio*,
evaluated against Laliberté's own water, and multiplied onto CoolProp's water
density and viscosity at the actual (T, P). The freshwater limit is therefore
exactly CoolProp, and the pressure dependence of water is retained; the
pressure dependence of the salt effect itself is not modelled. The input
concentration (g L⁻¹ NaCl, mol L⁻¹ KI) fixes the solute mass per volume of
solution, so w = c/ρ_m is solved by fixed-point iteration. Ranges of the fit
(Laliberté, 2009, Tables 53 and 84): NaCl density 0–140 °C and viscosity
5–154 °C, w ≤ 0.264 (about 6 mol kg⁻¹); KI density 0–100 °C and viscosity
5–95 °C, w ≤ 0.627. Outside these the temperature dependence of the salt
effect is held at the edge of the fitted range (never extrapolated) and the
interface marks the value "extrapolated".

The calculator limits salt concentration to 2 mol L⁻¹ for both salts, so the composition range stays inside the fitted mass-fraction range (2 mol L⁻¹ is w ≈ 0.11 for NaCl and 0.27 for KI) and, for NaCl, close to the highest molality at which the IFT increment was measured (1.98 mol kg⁻¹; 2 mol L⁻¹ NaCl is 2.1–2.25 mol kg⁻¹). For KI, 2 mol L⁻¹ is 2.2–2.4 mol kg⁻¹, slightly beyond that molality, so the KI IFT estimate there carries the out-of-range flag.

[Accuracy, checked here: against the Apelblat (2017) supplementary tables
(Tables S2, S12; 5–80 °C, ≤1 mol kg⁻¹) the density agrees to within 0.08%
(NaCl) and 0.12% (KI), and the viscosity to within 2.8% in both cases (the largest NaCl
viscosity deviation is at 5 °C). Against CoolProp's independent Melinder NaCl-brine correlation
(5–40 °C, 0.5–5 mol kg⁻¹) the density agrees to 0.07% and the viscosity to 2.3%.
These replace formulas that were +0.3% to +3.9% high in density and up to 41%
low in viscosity at 5 mol kg⁻¹ (Rowe & Chou/Batzle & Wang density; Mao & Duan
viscosity; CRC density and a Jones–Dole viscosity for KI). The Apelblat and
Melinder values are independent of Laliberté's own data compilation only in
part — all draw on the same literature — so the agreement is a consistency
check rather than an independent validation. The Laliberté coefficients were
read from the `chemicals` Python package's copy of the Supporting Information
table and cross-checked against the T and w ranges printed in the paper.]

The brine correlations are liquid-phase correlations and are applied only
where water is liquid at (T, P); at states where water is vapour the
wetting-phase properties (and all downstream dimensionless numbers) are
withheld rather than extrapolated.

**Dissolved CO₂ is not included in the wetting-phase density.** The densities
above are for CO₂-free water or brine. In an equilibrated system the aqueous
phase contains dissolved CO₂, which raises its density. An estimate from the
solubility model of Mutailipu et al. (2024) and its partial molar volume of CO₂
at infinite dilution gives an increase of about 1.4% in pure water at 25 °C and
10 MPa, 1.0% at 50 °C, 0.5% at 100 °C and 0.3% at 150 °C, and about two thirds
of those values in 1 mol L⁻¹ NaCl. Because the effect of the salt on density is
larger than that of the dissolved CO₂, and to keep the density a property of
the brine alone, the CO₂ contribution is left out, and an error of up to about
1–2% in the aqueous density (largest in cold, high-pressure fresh water) is
expected where the phases are equilibrated. The effect is largest on the Bond
number, which depends on the density difference between the phases: where that
difference is small, as at 25 °C and 10 MPa, a 1.4% shift in the aqueous density
changes Bo by roughly 9%. [The partial molar volume used is evaluated at the
vapour pressure of water and carries no pressure dependence, so the estimate
may be slightly low at high pressure; it has not been compared with measured
densities of CO₂-saturated water.]

## 4. Interfacial tension

The CO₂–water and CO₂–brine interfacial tension (IFT) is calculated with the
correlation of Mutailipu et al. (2024), valid from 273 to 473 K, to 70 MPa and
for brines up to 5 mol kg⁻¹ of Na⁺, K⁺, Mg²⁺ and Ca²⁺ with Cl⁻, SO₄²⁻ and
HCO₃⁻. The IFT is written as

    γ = γ_b − Δγ(τ, x₁),   τ = T/304.63 K − 1,

where γ_b is the surface tension of the brine in the absence of CO₂ and
x₁ is the mole fraction of CO₂ dissolved in the aqueous phase. For pure water
γ_b is the IAPWS surface tension of water. For a brine,

    γ_b − γ_w = (1.608 + 1.072 τ) m⁺  [mN m⁻¹],

with m⁺ the charge-weighted cation molality (Σ zᵢmᵢ, mol kg⁻¹). The
dimensionless difference Δγ is a piecewise function of τ and x₁ with fourteen
fitted parameters (their Table 11), with separate forms for the gas–liquid
region (x₁ ≤ x₃) and the liquid–liquid or dense region (x₁ > x₃), where x₃ is
the dissolved CO₂ mole fraction on the three-phase line. Below the upper
critical end point (31.5 °C) the IFT therefore has a real, step-like change of
up to a few mN m⁻¹ across the three-phase pressure (about 5 MPa at 15 °C);
derivatives of the IFT are not reported across it.

**Dissolved CO₂.** x₁ comes from the mutual-solubility model of the same paper,
a γ–φ calculation in which the aqueous phase is described by a modified
Margules activity model and the CO₂-rich phase by the Peng–Robinson equation
of state with the Panagiotopoulos–Reid combining rule. Henry's constant
(Fernández-Prini et al.), the saturated-water fugacity and molar volume
(IAPWS-95) and the partial molar volume of CO₂ at infinite dilution
(McBride-Wright et al.) enter through the paper's polynomial fits in τ. The
paper's polynomials reproduce the underlying correlations in the implementation
to 0.2% for the saturated-water fugacity times pressure (275–473 K), 0.02% for
the saturated molar volume, and 0.3% for Henry's constant. The calculated CO₂
solubilities agree with the Duan et al. (2006) model to within 3% at 5–10 MPa
from 323 to 423 K (pure water and 1–2 mol kg⁻¹ NaCl) and to within ~5% at
20 MPa above 373 K. At lower temperatures and higher pressures the Duan et al.
values are higher (by 16% at 323 K and 20 MPa, and by ~30% at 283–298 K and
20 MPa); that is the liquid-CO₂ region. This comparison does not establish
which model is closer to experiment there, and the present application uses the
solubility only through the IFT.

**Two corrections to the printed equations.** (i) In their Eq. 38, as typeset,
the coefficient b₆ appears to multiply both terms of the bracket. Fitted to the
paper's own CO₂–water data, the model reproduces the measurements only if b₆
multiplies the first term alone; with b₆ on both terms the dense-region IFT is
wrong by up to 1400 mN m⁻¹. The form used is
Δγ₄ = [b₆{1 + (b₇ + b₈τ)x₃²}x₃ + (b₉ + b₁₀τ)(x₁ − x₃)]F(τ). (ii) Eq. A-4 prints
"0.7464 + 1.54226ω − 0.26992ω", which is the standard Peng–Robinson
0.37464 + 1.54226ω − 0.26992ω².

**Brine.** For a brine, x₁ is evaluated from the *salt-free* solubility at the
same T and p, so the effect of the salt on the IFT enters through γ_b alone.
This is not obvious from the text of the paper, which says x₁ is "related to
ion molalities", and it was established two ways. (i) With the salted x₁
(salting-out lowers x₁ by ~20% at 1 mol kg⁻¹), the model over-predicts the 42
CO₂ + NaCl(aq) IFTs of Pereira et al. (2017) by +9.0 mN m⁻¹ on average
(rms 10.1); with the salt-free x₁ the same data are reproduced to a mean
difference of −0.03 and an rms of 0.89 mN m⁻¹. (ii) The Supplementary
Information of the paper tabulates, for every brine data point (Tables A.21,
A.27, A.28), the CO₂ mole fraction and the IFT calculated by the authors; the
tabulated mole fractions equal the salt-free values (for example 0.02594 at
298.6 K, 10.31 MPa and 0.98 mol kg⁻¹ NaCl, against about 0.020 salted), and the
implementation here reproduces their x₁ and IFT for all 453 brine points
(IFT rms 0.03–0.04 mN m⁻¹, the rounding of the tables).

**Validation against the authors' own calculations.** The 921 CO₂–water points tabulated in the Supplementary Information (Tables A.9–A.20) are reproduced to an IFT rms of 0.036 mN m⁻¹ (maximum 0.24) and an x₁ rms of 1×10⁻⁵, which also confirms the two corrections to the printed equations above.

**Validation against measurements.** The model reproduces the 78 CO₂–water
IFTs of Pereira et al. (2016, their Table 6; 25–196 °C, to 69 MPa) with a mean
difference of +0.2 mN m⁻¹, an rms of 1.0 mN m⁻¹ (AARD 2.8%) and a maximum of
3.6 mN m⁻¹, and the 42 NaCl measurements as above. The paper reports an overall
rms of 0.65 mN m⁻¹ (AARD 1.8%) for CO₂–water over 363 primary data points and
states that nearly all points fall within ±2 mN m⁻¹. Compared with the direct
interpolation of the measured data used previously, the model differs by
0.0–0.9 mN m⁻¹ on average and by at most 2.0 mN m⁻¹ per isotherm between 30 and
196 °C, which is the same size as the interpolation's own estimated error
between isotherms (0.9–1.8 mN m⁻¹ rms).

**KI (estimate).** The model has no iodide, and no CO₂/water–KI interfacial
tension measurements are available. KI is therefore treated as a 1:1 potassium
salt at its own molality — its brine term depends only on the charge-weighted
cation molality, 1 mol kg⁻¹ for a 1:1 salt, the same as for NaCl — and the
result is labelled an estimate in the interface. Whether iodide behaves like
chloride at the CO₂/water interface has not been measured. [The previous
estimate (the measured NaCl increment of Pereira et al. at the same molality)
and the still earlier −0.8 mN m⁻¹ per mol L⁻¹ from air/water data have been
removed.]

**Range and flags.** Outside 273–473 K, above 70 MPa, or above 5 mol kg⁻¹ the
correlation is extrapolated, the state is flagged in the interface and the
IFT derivatives are withheld. The application's own window (10–200 °C, up to
40 MPa, up to 2 mol L⁻¹) lies inside the validity range.

## 5. Dimensionless numbers and displacement regimes

For a core-flood at volumetric rate Q through a cylindrical core of diameter
d and porosity φ, the interstitial velocity is

    v = Q / (A φ),   A = π (d/2)².

The displacement is characterized by the contact-angle-corrected capillary
number, Bond number, and viscosity ratio:

    Ca = v μ_inj / (σ cos θ),
    Bo = Δρ g L² / (σ cos θ),
    M  = μ_NW / μ_W,

where μ_inj is the viscosity of the injected phase (μ_NW for drainage, μ_W
for imbibition; Ca is reported for both), σ the interfacial tension, θ the
receding contact angle, Δρ = |ρ_W − ρ_NW|, and L the characteristic pore
length. [State in the paper how L was chosen — e.g. mean pore diameter or the
critical percolation diameter l_c from the network analysis.] The calculator
also inverts Ca for the flow rate, Q = Ca σ cos θ A φ / μ_inj, to design
experiments at a target capillary number.

Drainage displacements are located on the Ca–M regime diagram of Lenormand
et al. (1988), with the phase boundaries of Lenormand et al. (1988) and the
modified boundaries of Zhang et al. (2011); these boundaries are approximate
and depend on pore geometry and wettability. The relative magnitude of
capillary, viscous, and gravitational forces is summarized on a force-balance
diagram (Polak et al., 2011) with coordinates Ca/Bo (viscous/gravity) and 1/Bo
(capillary/gravity), on which both injection directions are plotted.

## 6. In-situ CO₂ storage conditions

[Include only if the CCS-site panel is used in the paper.] Pressure–temperature
conditions for four field CO₂ storage projects are taken from a single
compilation, Imre et al. (2015), who in turn cite the original sources:
Sleipner, Utsira Formation, North Sea, 25–42 °C and 8–11 MPa (Baklid et al.,
1996; Bickle et al., 2007; Arts et al., 2008; Ringrose & Eiken, 2011); Snøhvit,
Barents Sea, 98–100 °C with pressure rising from about 29 MPa (2008) to about
39 MPa (2011) (Hansen et al., 2011, 2013); Nagaoka, Japan, at the well head
(7–11 MPa, 32 °C) and well bottom (19 MPa, 48 °C) (Kikuta et al., 2005); and
Ketzin, Germany, initial reservoir conditions of 33.5–34 °C and 6.1–6.2 MPa
(Kempka & Kühn, 2013, 62 bar and a constant 34 °C; Loizzo et al., 2013,
33.5 °C and 6.1 MPa). Each site is plotted as a box spanning its reported
ranges on the CO₂ phase diagram, and the in-situ CO₂ density range is evaluated
from the Span & Wagner EOS over that box. Nagaoka's values are well-head and
well-bottom conditions, not a reservoir range, and Imre et al. note that the
storage P–T conditions there are not available. Phase regime labels are
assigned from explicit comparison with T_c and P_c rather than from the EOS
phase index, since a state with T > T_c but P < P_c (the Ketzin reservoir and
the low-pressure end of the Nagaoka well head) has gas-like density and is not
supercritical in the storage-relevant sense. [Sites not in Imre et al. were
removed on 2026-10-21 so that every value rests on one source; the earlier
values for Quest, In Salah, Frio and the Illinois Basin were from other
references. Imre et al. also give two natural CO₂ analogues, Montmiral and
Mihályi-Répcelak, which are not plotted.]

---

## Reference list (to format in journal style)

- Apelblat, A. (2017). Effect of temperature on compressibility properties of 0.1, 0.5 and 1.0 molal solutions of alkali metal halides. Part 1. Aqueous solutions of sodium chloride, sodium bromide, sodium iodide, potassium chloride, potassium bromide, potassium iodide, rubidium chloride and rubidium iodide in the 278.15 K to 353.15 K temperature range. *J. Mol. Liq.*, 242, 1263–1284. doi:10.1016/j.molliq.2017.07.110 [used here to validate brine density and viscosity, supplementary Tables S2 and S12].
- Arts, R., Chadwick, A., Eiken, O., Thibeau, S., & Nooner, S. (2008). Ten years' experience of monitoring CO₂ injection in the Utsira Sand at Sleipner, offshore Norway. *First Break*, 26, 65–72.
- Baklid, A., Korbøl, R., & Owren, G. (1996). Sleipner Vest CO₂ disposal, CO₂ injection into a shallow underground aquifer. Society of Petroleum Engineers, SPE paper 36600.
- Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014). Pure and pseudo-pure fluid thermophysical property evaluation and the open-source thermophysical property library CoolProp. *Ind. Eng. Chem. Res.*, 53(6), 2498–2508.
- Bickle, M., Chadwick, A., Huppert, H. E., Hallworth, M., & Lyle, S. (2007). Modelling carbon dioxide accumulation at Sleipner: implications for underground carbon storage. *Earth Planet. Sci. Lett.*, 255, 164–176.
- Duan, Z., Sun, R., Zhu, C., & Chou, I. M. (2006). An improved model for the calculation of CO₂ solubility in aqueous solutions containing Na⁺, K⁺, Ca²⁺, Mg²⁺, Cl⁻, and SO₄²⁻. *Marine Chemistry*, 98, 131.
- Fenghour, A., Wakeham, W. A., & Vesovic, V. (1998). The viscosity of carbon dioxide. *J. Phys. Chem. Ref. Data*, 27, 31–39.
- Fernández-Prini, R., Alvarez, J. L., & Harvey, A. H. (2003). Henry's constants and vapor–liquid distribution constants for gaseous solutes in H₂O and D₂O at high temperatures. *J. Phys. Chem. Ref. Data*, 32(2), 903–916.
- Hansen, O., Eiken, O., Østmo, S., Johansen, R. I., & Smith, A. (2011). Monitoring CO₂ injection into a fluvial brine-filled sandstone formation at the Snøhvit field, Barents Sea. Society of Exploration Geophysicists, Expanded Abstract SEG-2011-4092.
- Hansen, O., Gilding, D., Nazarian, B., Osdal, B., Ringrose, P., Kristoffersen, J.-B., Eiken, O., & Hansen, H. (2013). Snøhvit: the history of injecting and storing 1 Mt CO₂ in the fluvial Tubåen Fm. *Energy Procedia*, 37, 3565–3573. [Imre et al. print the pages as 356–3573.]
- Huber, M. L., Perkins, R. A., Laesecke, A., Friend, D. G., Sengers, J. V., Assael, M. J., Metaxa, I. N., Vogel, E., Mareš, R., & Miyagawa, K. (2009). New international formulation for the viscosity of H₂O. *J. Phys. Chem. Ref. Data*, 38, 101.
- Imre, A. R., Ramboz, C., Deiters, U. K., & Kraska, T. (2015). Anomalous fluid properties of carbon dioxide in the supercritical region: application to geological CO₂ storage and related hazards. *Environ. Earth Sci.*, 73, 4373–4384. doi:10.1007/s12665-014-3716-5
- Kempka, T., & Kühn, M. (2013). Numerical simulations of CO₂ arrival times and reservoir pressure coincide with observations from the Ketzin pilot site, Germany. *Environ. Earth Sci.*, 70, 3675–3685.
- Kikuta, K., Hongo, S., Tanase, D., & Ohsumi, T. (2005). Field test of CO₂ injection in Nagaoka, Japan. In *Proceedings of the 7th International Conference on Greenhouse Gas Control Technologies*, 1367–1372.
- Laliberté, M. (2009). A model for calculating the heat capacity of aqueous solutions, with updated density and viscosity data. *J. Chem. Eng. Data*, 54(6), 1725–1760. doi:10.1021/je8008123.
- Lenormand, R., Touboul, E., & Zarcone, C. (1988). Numerical models and experiments on immiscible displacements in porous media. *J. Fluid Mech.*, 189, 165–187.
- Loizzo, M., Henninges, J., Zimmer, M., & Liebscher, A. (2013). Multi-phase equilibrium in a CO₂-filled observation well at the Ketzin pilot site. *Energy Procedia*, 37, 3621–3629.
- McBride-Wright, M., Maitland, G. C., & Trusler, J. P. M. (2015). Viscosity and density of aqueous solutions of carbon dioxide at temperatures from (274 to 449) K and at pressures up to 100 MPa. *J. Chem. Eng. Data*, 60(1), 171–180.
- Mutailipu, M., Song, Y., Yao, Q., Liu, Y., & Trusler, J. P. M. (2024). Solubility and interfacial tension models for CO₂–brine systems under CO₂ geological storage conditions. *Fuel*, 357, 129712.
- Panagiotopoulos, A. Z., & Reid, R. C. (1986). New mixing rules for cubic equations of state for highly polar, asymmetric systems. *ACS Symp. Ser.*, 300, 571–582.
- Peng, D.-Y., & Robinson, D. B. (1976). A new two-constant equation of state. *Ind. Eng. Chem. Fundam.*, 15(1), 59–64.
- Pereira, L. M. C., Chapoy, A., Burgass, R., & Tohidi, B. (2017). Interfacial tension of CO₂ + brine systems: Experiments and predictive modelling. *Adv. Water Resour.*, 103, 64–75.
- Pereira, L. M. C., Chapoy, A., Burgass, R., Oliveira, M. B., Coutinho, J. A. P., & Tohidi, B. (2016). Study of the impact of high temperatures and pressures on the equilibrium densities and interfacial tension of the carbon dioxide/water system. *J. Chem. Thermodyn.*, 93, 404–415.
- Polak, S., Cinar, Y., Holt, T., & Torsæter, O. (2011). An experimental investigation of the balance between capillary, viscous, and gravitational forces during CO₂ injection into saline aquifers. *Energy Procedia*, 4, 4395–4402. doi:10.1016/j.egypro.2011.02.392
- Ringrose, P., & Eiken, O. (2011). Sleipner and Snøhvit projects. CSLF Interactive Workshop, Saudi Arabia, 1–2 March 2011. Carbon Sequestration Leadership Forum.
- Span, R., & Wagner, W. (1996). A new equation of state for carbon dioxide covering the fluid region from the triple-point temperature to 1100 K at pressures up to 800 MPa. *J. Phys. Chem. Ref. Data*, 25(6), 1509–1596.
- Wagner, W., & Pruß, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance for general and scientific use. *J. Phys. Chem. Ref. Data*, 31(2), 387–535.
- Zhang, C., Oostrom, M., Wietsma, T. W., Grate, J. W., & Warner, M. G. (2011). Influence of viscous and capillary forces on immiscible fluid displacement: pore-scale experimental study in a water-wet micromodel. *Energy Fuels*, 25(8), 3493–3505.
