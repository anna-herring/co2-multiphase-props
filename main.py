import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import calculator

app = FastAPI(title="Multiphase Flow Fluid Properties")

# The static front end is also published on GitHub Pages (docs/), which calls
# this API from another origin. Allow that origin, plus local dev servers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anna-herring.github.io",
                   "http://localhost:8001", "http://127.0.0.1:8001"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def ngrok_skip(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

class CalcRequest(BaseModel):
    T_C:           float = Field(50.0, ge=10.0, le=200.0)
    # Upper limit raised 20 -> 40 MPa so every CCS site on the phase diagram
    # is reachable by clicking it (Snohvit sits at 33 MPa). The IFT model
    # (Mutailipu 2024) is valid to 70 MPa and 473 K, so this whole window is
    # inside it; the response still carries interaction.ift_out_of_range.
    P_MPa:         float = Field(10.0, ge=0.0,  le=40.0)
    nw_fluid:      str   = "CO2"
    w_fluid:       str   = "water"
    # Salt caps = 2 mol/L (NaCl 116.88 g/L; 116.9 leaves room for float noise).
    # Chosen so the brine density/viscosity (Laliberte 2009) and IFT
    # (Mutailipu 2024, up to 5 mol/kg) correlations are used without
    # extrapolation in concentration.
    salinity_g_L:  Optional[float] = Field(None, ge=0.0, le=116.9)
    Q_ml_hr:       Optional[float] = Field(None, ge=0.0)
    Ca_nw_target:  Optional[float] = Field(None, ge=0.0)
    Ca_w_target:   Optional[float] = Field(None, ge=0.0)
    ki_mol_L:      Optional[float] = Field(None, ge=0.0, le=2.0)
    core_diam_mm:  float = Field(10.0, ge=0.1, le=100.0)
    porosity:      float = Field(0.22, ge=0.01, le=0.99)
    char_len_um:   float = Field(40.0, ge=0.1, le=10000.0)
    theta_deg:     float = Field(10.0, ge=0.0, lt=90.0)

class SurfaceRequest(BaseModel):
    fluid:     str   = "CO2"
    T_min_C:   float = Field(10.0,  ge=-250.0, le=500.0)
    T_max_C:   float = Field(200.0, ge=-250.0, le=500.0)
    P_min_MPa: float = Field(0.1,   ge=0.0,    le=100.0)
    P_max_MPa: float = Field(20.0,  ge=0.0,    le=100.0)
    n_T:       int   = Field(70, ge=10, le=140)
    n_P:       int   = Field(70, ge=10, le=140)


@app.post("/api/surface")
def surface(req: SurfaceRequest):
    if req.fluid not in calculator.SURFACE_FLUIDS:
        raise HTTPException(400, f"Unknown surface fluid: {req.fluid}")
    if req.T_max_C <= req.T_min_C:
        raise HTTPException(400, "T_max_C must be greater than T_min_C")
    if req.P_max_MPa <= req.P_min_MPa:
        raise HTTPException(400, "P_max_MPa must be greater than P_min_MPa")
    return calculator.surface_grid(req.fluid, req.T_min_C, req.T_max_C,
                                   req.P_min_MPa, req.P_max_MPa,
                                   req.n_T, req.n_P)


@app.get("/api/ccs-sites")
def ccs_sites():
    """CO2 phase diagram with field CCS sites placed on it."""
    return calculator.co2_phase_diagram()


@app.get("/api/surface-fluids")
def surface_fluids():
    return {k: {"label": v["label"], "color": v["color"]}
            for k, v in calculator.SURFACE_FLUIDS.items()}


@app.get("/api/fluids")
def list_fluids():
    return {
        "nw": {k: {"label": v["label"], "color": v["color"]}
               for k, v in calculator.NW_FLUIDS.items()},
        "wetting": {k: {"label": v["label"], "default_S": v["default_S"]}
                    for k, v in calculator.W_FLUIDS.items()},
    }

@app.post("/api/calculate")
def calculate(req: CalcRequest):
    if req.nw_fluid not in calculator.NW_FLUIDS:
        raise HTTPException(400, f"Unknown NW fluid: {req.nw_fluid}")
    if req.w_fluid not in calculator.W_FLUIDS:
        raise HTTPException(400, f"Unknown W fluid: {req.w_fluid}")
    return calculator.calculate(req.T_C, req.P_MPa, req.nw_fluid,
                                req.w_fluid, req.salinity_g_L,
                                req.Q_ml_hr, req.core_diam_mm,
                                req.porosity, req.char_len_um,
                                req.theta_deg, req.Ca_nw_target,
                                req.ki_mol_L, req.Ca_w_target)

_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=_static, html=True), name="static")
