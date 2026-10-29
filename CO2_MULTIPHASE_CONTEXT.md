# CO₂ Multiphase Flow Properties: context for contributors and AI assistants

This file describes the Multiphase Flow Fluid Properties calculator for CO₂–water and CO₂–brine systems. Give it to a Claude project, or any AI assistant, to bring it up to speed before making changes.

- **Live app (Render):** https://co2-multiphase-props.onrender.com
- **Front end (GitHub Pages):** https://anna-herring.github.io/co2-multiphase-props/
- **Repository:** https://github.com/anna-herring/co2-multiphase-props (branch `main`)

## 1. What the app computes

For a CO₂ non-wetting phase displacing a wetting phase (freshwater, NaCl brine or KI brine) in a porous core, at a user-set temperature and pressure:

- **Pure-fluid properties** of CO₂ and of the wetting phase: density, viscosity, speed of sound, isothermal compressibility, heat capacities, thermal expansivity, conductivity, Joule–Thomson coefficient, compressibility factor.
- **Interfacial tension (IFT)** between CO₂ and the wetting phase, including the effect of dissolved salt.
- **Dimensionless numbers** for core flooding: capillary number Ca, Bond number Bo, viscosity ratio M, from the flow rate, core diameter, porosity, characteristic length and contact angle. It can also invert Ca for a target flow rate.
- **Displacement regimes:** positions on the Lenormand et al. (1988) capillary-number/viscosity-ratio diagram (with Zhang et al. 2011 boundaries), and on a capillary–viscous–gravity force-balance diagram (Polak et al. 2011).
- **Sensitivity table:** log-derivatives of density, viscosity, speed of sound, compressibility, IFT and Δρ with respect to temperature and pressure, and the decomposition of Ca and Bo into their input sensitivities.
- **Property surfaces:** 3-D plots of any property over the app's window (10–200 °C, 0.1–20 MPa).
- **Field CCS sites:** P–T boxes for four storage sites on the CO₂ phase diagram (see §4).

The app is CO₂-only. Hydrogen and nitrogen were removed (2026-09-29); their dormant tables remain in `calculator.py`.

## 2. Architecture

- **`calculator.py`:** all physics. Entry points: `calculate()`, `surface_grid()`, `co2_phase_diagram()`, `sensitivity_block()`, `range_flags()`.
- **`mutailipu.py`:** the Mutailipu et al. (2024) CO₂–brine solubility and IFT model, self-contained.
- **`main.py`:** FastAPI app. Endpoints: `GET /api/fluids`, `POST /api/calculate`, `POST /api/surface`, `GET /api/surface-fluids`, `GET /api/ccs-sites`. Serves `static/` for local use and allows CORS from the GitHub Pages origin.
- **`static/index.html`:** the whole front end (HTML, CSS, JavaScript, Plotly and Chart.js). The footer logos are `doe_logo.png`, `utk_logo.png` and `osu_logo.png`.
- **`docs/`:** a copy of the front end for GitHub Pages. `docs/` must match `static/` (see §6).
- **`METHODS_DRAFT.md`:** methods text for a publication. Its bracketed notes are author working notes.
- **`requirements.txt`, `Procfile`:** Render deployment. Dependencies: fastapi, uvicorn, CoolProp, scipy, numpy, pydantic.

Every state point is evaluated through one CoolProp `AbstractState` update, so all properties at a state are consistent. Use method calls such as `AS.speed_sound()` rather than `PropsSI` output keys. Several `PropsSI` keys fail silently or raise errors, for example `"speed_sound"` and `"fugacity"`.

## 3. Models and sources

**CO₂ and water pure-fluid properties**
- Span & Wagner (1996) for CO₂, and IAPWS-95 (Wagner & Pruß 2002) for water, both through CoolProp (Bell et al. 2014).
- CO₂ viscosity: Fenghour, Wakeham & Vesovic (1998). Water viscosity: Huber et al. (2009).
- Derived quantities use exact identities: c = √(γ/(ρκ_T)), κ_s = κ_T/γ, γ = c_p/c_v.

**Wetting-phase density and viscosity (brines)**
- Laliberté (2009) apparent-density and viscosity model for NaCl and KI, applied as a solution-to-water ratio multiplied onto CoolProp water at the actual T and P.
- Salt concentration is capped at 2 mol L⁻¹ for both salts. Outside the fitted temperature range the value is held at the range edge and flagged.
- The salt effect does not depend on pressure. This is a known limitation (see §5).

**Wetting-phase thermodynamic properties (speed of sound, compressibility, expansivity, conductivity, Joule–Thomson, Z)**
- These are pure-water values for all wetting phases, and the UI labels them that way. The brine corrections built earlier were dropped on 2026-10-09.

**CO₂–water and CO₂–brine interfacial tension**
- Mutailipu, Song, Yao, Liu & Trusler (2024), *Fuel* 357, 129712. Valid 273–473 K, up to 70 MPa, and brines up to 5 mol kg⁻¹.
- γ = γ_b − Δγ(τ, x₁), with τ = T/304.63 K − 1 and x₁ the dissolved CO₂ mole fraction.
- Three points differ from the printed paper and were confirmed against the authors' Supplementary Information: Eq. 38's b₆ multiplies only the first bracket term; the Peng–Robinson κ uses 0.37464 + 1.54226ω − 0.26992ω²; and for brines, x₁ is the salt-free solubility.
- **Potassium iodide:** the model has no iodide, and no CO₂/KI IFT data exist. KI is treated as a 1:1 potassium salt at its own molality and is always labelled "estimate".
- Below 31.5 °C, the IFT has a real step across the three-phase pressure. Sensitivities are withheld across it.

**CO₂ density in the aqueous phase**
- Dissolved CO₂ is not included in the wetting-phase density. The expected error is 1–2%, and it is largest for Bond number at low Δρ.

**Field CCS sites**
- Four sites from Imre, Ramboz, Deiters & Kraska (2015), *Environ. Earth Sci.* 73, 4373–4384: Sleipner, Snøhvit, Nagaoka and Ketzin. Each is plotted as a box over the reported ranges.
- Nagaoka values are well-head and well-bottom conditions, not a reservoir range.
- Ketzin: Kempka & Kühn (2013) and Loizzo et al. (2013) give the initial reservoir conditions. Kempka & Kühn's value is unpublished data.
- Regimes are assigned by comparing with T_c and P_c explicitly, not from CoolProp's phase label.

**Sensitivities**
- Relative log-derivatives by central differences: S_T in %/K and S_P in %/10 kPa.
- Step sizes are 0.05 K and 0.5 kPa, which converge near the critical point to within about 1%.
- Derivatives are withheld across the IFT branch points and the three-phase line, and flagged near the critical point (κ_T·P ≥ 5).

## 4. Validation (what has been checked)

- **CO₂–water IFT:** against Pereira et al. (2016), 78 points at 25–196 °C up to 69 MPa: mean +0.2 mN/m, rms 1.0 mN/m, max 3.6 mN/m.
- **CO₂–NaCl IFT:** against Pereira et al. (2017), 42 points: mean 0.0, rms 0.9 mN/m.
- **Against the Mutailipu supplementary tables:** 921 CO₂–water points with IFT rms 0.036 mN/m, and 453 brine points with rms 0.03–0.04 mN/m.
- **Laliberté brine density and viscosity:** against Apelblat (2017) supplementary tables (5–80 °C, ≤1 mol kg⁻¹): density within 0.12%, viscosity within 2.8%. Against the Melinder correlation in CoolProp (5–40 °C): density within 0.07%.
- **Brine density at pressure (NaCl):** against Rowe & Chou (1970), the app reads slightly high at high pressure: max +0.30%, rms 0.09%, up to 34.3 MPa.
- **CO₂ fugacity:** checked against a finite difference of the residual Helmholtz energy (exact to 6 decimal places).

Earlier versions had errors, including a fugacity call that silently fell back to ideal gas, and a brine compressibility that was too weak. Both are fixed.

## 5. Known limitations

- **Pressure dependence of the brine salt effect** is not modelled. The app reads slightly high at high pressure and high salinity, and the error is larger at 20 MPa and above than at lower pressure.
- **KI** has no measured IFT and no pressure-dependence check. Treat KI results as estimates.
- **Brine thermodynamic properties** other than density and viscosity are pure-water values.
- **Dissolved CO₂ in the aqueous density** is left out (see §3).
- **Near the critical point (about 31 °C, 7.4 MPa for CO₂)**, sensitivity derivatives genuinely diverge and depend on the step size. Read them as magnitudes.
- **Interfacial tension below 31.5 °C** has a physical step, and derivatives are not reported across it.
- **Render free tier** sleeps after inactivity, so the first calculation can take about a minute to respond.
- **The GitHub Pages site depends on Render.** The front end loads from Pages, but every calculation goes to the Render API, so the site fails if Render is down.

## 6. Editing and deploying

1. Edit `static/index.html` or the Python files.
2. Run `./sync_pages.sh` to copy the front end and images into `docs/`. Commit both.
3. Commit and push to `main`. Render redeploys the backend automatically from `main`.
4. If the Render service URL changes, update `RENDER_API` near the top of `static/index.html`, then run `sync_pages.sh`.
5. Local run: `uvicorn main:app --port 8001` from the `multiphase_props` folder, then open http://localhost:8001. Locally the front end calls the API on the same origin.

If you add a new site to `CCS_SITES` in `calculator.py`, give it a source, and plot it only if both temperature and pressure are sourced.

## 7. References

- Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014). Pure and pseudo-pure fluid thermophysical property evaluation and the open-source thermophysical property library CoolProp. *Ind. Eng. Chem. Res.*, 53(6), 2498–2508.
- Duan, Z., & Sun, R. (2003). An improved model calculating CO₂ solubility in pure water and aqueous NaCl solutions from 273 to 533 K and from 0 to 2000 bar. *Chem. Geol.*, 193, 257.
- Duan, Z., Sun, R., Zhu, C., & Chou, I. M. (2006). An improved model for the calculation of CO₂ solubility in aqueous solutions containing Na⁺, K⁺, Ca²⁺, Mg²⁺, Cl⁻, and SO₄²⁻. *Marine Chemistry*, 98, 131.
- Fenghour, A., Wakeham, W. A., & Vesovic, V. (1998). The viscosity of carbon dioxide. *J. Phys. Chem. Ref. Data*, 27, 31–39.
- Huber, M. L., Perkins, R. A., Laesecke, A., Friend, D. G., Sengers, J. V., Assael, M. J., Metaxa, I. N., Vogel, E., Mareš, R., & Miyagawa, K. (2009). New international formulation for the viscosity of H₂O. *J. Phys. Chem. Ref. Data*, 38, 101.
- Imre, A. R., Ramboz, C., Deiters, U. K., & Kraska, T. (2015). Anomalous fluid properties of carbon dioxide in the supercritical region: application to geological CO₂ storage and related hazards. *Environ. Earth Sci.*, 73, 4373–4384.
- Kempka, T., & Kühn, M. (2013). Numerical simulations of CO₂ arrival times and reservoir pressure coincide with observations from the Ketzin pilot site, Germany. *Environ. Earth Sci.*, 70, 3675–3685.
- Laliberté, M. (2009). A model for calculating the heat capacity of aqueous solutions, with updated density and viscosity data. *J. Chem. Eng. Data*, 54(6), 1725–1760.
- Lenormand, R., Touboul, E., & Zarcone, C. (1988). Numerical models and experiments on immiscible displacements in porous media. *J. Fluid Mech.*, 189, 165–187.
- Loizzo, M., Henninges, J., Zimmer, M., & Liebscher, A. (2013). Multi-phase equilibrium in a CO₂-filled observation well at the Ketzin pilot site. *Energy Procedia*, 37, 3621–3629.
- McBride-Wright, M., Maitland, G. C., & Trusler, J. P. M. (2015). Viscosity and density of aqueous solutions of carbon dioxide at temperatures from (274 to 449) K and at pressures up to 100 MPa. *J. Chem. Eng. Data*, 60(1), 171–180.
- Mutailipu, M., Song, Y., Yao, Q., Liu, Y., & Trusler, J. P. M. (2024). Solubility and interfacial tension models for CO₂–brine systems under CO₂ geological storage conditions. *Fuel*, 357, 129712.
- Pereira, L. M. C., Chapoy, A., Burgass, R., & Tohidi, B. (2017). Interfacial tension of CO₂ + brine systems: experiments and predictive modelling. *Adv. Water Resour.*, 103, 64–75.
- Pereira, L. M. C., Chapoy, A., Burgass, R., Oliveira, M. B., Coutinho, J. A. P., & Tohidi, B. (2016). Study of the impact of high temperatures and pressures on the equilibrium densities and interfacial tension of the carbon dioxide/water system. *J. Chem. Thermodyn.*, 93, 404–415.
- Polak, S., Cinar, Y., Holt, T., & Torsæter, O. (2011). An experimental investigation of the balance between capillary, viscous, and gravitational forces during CO₂ injection into saline aquifers. *Energy Procedia*, 4, 4395–4402.
- Span, R., & Wagner, W. (1996). A new equation of state for carbon dioxide covering the fluid region from the triple-point temperature to 1100 K at pressures up to 800 MPa. *J. Phys. Chem. Ref. Data*, 25(6), 1509–1596.
- Wagner, W., & Pruß, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance for general and scientific use. *J. Phys. Chem. Ref. Data*, 31(2), 387–535.
- Zhang, C., Oostrom, M., Wietsma, T. W., Grate, J. W., & Warner, M. G. (2011). Influence of viscous and capillary forces on immiscible fluid displacement: pore-scale experimental study in a water-wet micromodel. *Energy Fuels*, 25(8), 3493–3505.
- Apelblat, A. (2017). Effect of temperature on compressibility properties of 0.1, 0.5 and 1.0 molal solutions of alkali metal halides. Part 1. *J. Mol. Liq.*, 242, 1263–1284.
- Rowe, A. M., & Chou, J. C. S. (1970). Pressure–volume–temperature–concentration relation of aqueous NaCl solutions. *J. Chem. Eng. Data*, 15, 61–66.
