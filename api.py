"""
cvae_api.py  -  FastAPI backend for the cVAE Synthetic Patient Generator
==========================================================================
Run with:
    uvicorn cvae_api:app --reload --port 8000

Requirements:
    pip install fastapi uvicorn torch numpy pandas scikit-learn anthropic

Expects these pickle/weight files in SAVE_DIR:
    - scalers_and_cols_swd.pkl   (meta: x_dim, c_dim, latent_dim, COND_COLS, FEATURE_COLS, LOG_TRANSFORM_COLS)
    - cvae_weights_swd.pt        (model state dict)

Also expects the scaler objects inside the pickle:
    meta['x_scaler'], meta['c_scaler']
"""

import os
import json
import pickle
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import anthropic

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# -- Config ---------------------------------------------------------------------
SAVE_DIR = os.environ.get("CVAE_SAVE_DIR", "model_artifacts")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Session store (in-memory) -------------------------------------------------
# { session_id: { "messages": [...], "created_at": str, "updated_at": str, "label": str } }
SESSIONS: dict = {}

# -- Model definition (matches training code exactly) -----------------------

class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, hidden_dims, latent_dim):
        super().__init__()
        in_dim = x_dim + c_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.z   = nn.Linear(in_dim, latent_dim)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=-1)
        return self.z(self.net(xc))


class Decoder(nn.Module):
    def __init__(self, latent_dim, c_dim, hidden_dims, x_dim):
        super().__init__()
        in_dim = latent_dim + c_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


class cVAE(nn.Module):
    def __init__(self, x_dim, c_dim, hidden_dims=(128, 64), latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(x_dim, c_dim, hidden_dims, latent_dim)
        self.decoder    = Decoder(latent_dim, c_dim, hidden_dims[::-1], x_dim)

    def forward(self, x, c):
        z     = self.encoder(x, c)
        x_hat = self.decoder(z, c)
        return x_hat, z

    @torch.no_grad()
    def generate(self, c, n_samples=1):
        self.eval()
        c = c.to(DEVICE)
        if n_samples > 1:
            c = c.repeat_interleave(n_samples, dim=0)
        z = torch.randn(c.size(0), self.latent_dim, device=DEVICE)
        return self.decoder(z, c)


# -- Load model & scalers ------------------------------------------------------

def load_model():
    with open(f"{SAVE_DIR}/scalers_and_cols_swd.pkl", "rb") as f:
        meta = pickle.load(f)

    model = cVAE(
        x_dim=meta["x_dim"],
        c_dim=meta["c_dim"],
        hidden_dims=(128, 64),
        latent_dim=meta["latent_dim"],
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(f"{SAVE_DIR}/cvae_weights_swd.pt", map_location=DEVICE)
    )
    model.eval()
    return model, meta


model, meta = load_model()
x_scaler         = meta["x_scaler"]
c_scaler         = meta["c_scaler"]
COND_COLS        = meta["cond_cols"]
FEATURE_COLS     = meta["feature_cols"]
LOG_TRANSFORM_COLS = meta.get("log_transform_cols", [
    "glucose", "creatinine", "lactate", "pt", "ptt", "bilirubin", "ast"
])

# -- Helpers -------------------------------------------------------------------

VALID_CAREUNITS  = ["CCU", "CSRU", "MICU", "SICU", "TSICU"]
VALID_LOS        = ["<2d", "2-7d", "7-14d", ">14d"]
VALID_CCS        = [
    "infectious", "neoplasm", "endocrine_metabolic", "blood", "mental",
    "nervous", "circulatory", "respiratory", "digestive", "genitourinary",
    "pregnancy", "skin", "musculoskeletal", "congenital", "perinatal",
    "symptoms", "injury", "external", "other",
]
VALID_ELIX = [
    "chf", "arrhythmia", "valvular_disease", "pulm_circulation", "pvd",
    "hypertension", "paralysis", "other_neuro", "chronic_pulm",
    "diabetes_uncomp", "diabetes_comp", "hypothyroidism", "renal_failure",
    "liver_disease", "pud", "aids", "lymphoma", "metastatic_ca",
    "solid_tumor", "rheumatoid", "coagulopathy", "obesity", "weight_loss",
    "fluid_elec", "blood_loss", "deficiency_anemia", "alcohol_abuse",
    "drug_abuse", "psychoses", "depression", "septicemia",
]


def build_condition_vector(age, gender_male, careunit, los_bucket, ccs_group, elix_flags=None):
    row = {col: 0.0 for col in COND_COLS}
    row["age"]        = float(age)
    row["gender_bin"] = float(gender_male)
    for key, prefix in [(careunit, "icu_"), (los_bucket, "los_"), (ccs_group, "ccs_")]:
        col = f"{prefix}{key}"
        if col in row:
            row[col] = 1.0
    if elix_flags:
        for k, v in elix_flags.items():
            col = f"elix_{k}"
            if col in row:
                row[col] = float(v)
    c_raw    = np.array([[row[col] for col in COND_COLS]], dtype=np.float32)
    c_scaled = c_scaler.transform(c_raw)
    return torch.tensor(c_scaled, dtype=torch.float32)


def inverse_transform_labs(scaled_array):
    raw    = x_scaler.inverse_transform(scaled_array)
    result = pd.DataFrame(raw, columns=FEATURE_COLS)
    for col in LOG_TRANSFORM_COLS:
        if col in result.columns:
            result[col] = np.expm1(result[col])
    return result


def generate_patients(age, gender_male, careunit, los_bucket, ccs_group,
                      elix_flags=None, n_samples=5):
    c = build_condition_vector(age, gender_male, careunit, los_bucket, ccs_group, elix_flags)
    synth_scaled = model.generate(c, n_samples=n_samples).cpu().numpy()
    df = inverse_transform_labs(synth_scaled)
    # round to 3 decimal places for clean output
    return df.round(3).to_dict(orient="records")


# -- FastAPI app ---------------------------------------------------------------

app = FastAPI(title="cVAE Synthetic Patient API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Tool definitions for Claude -----------------------------------------------

TOOLS = [
    {
        "name": "generate_synthetic_patients",
        "description": (
            "Generate synthetic ICU patient lab records using a trained cVAE model. "
            "Returns realistic first-day lab values (23 labs) conditioned on patient demographics "
            "and clinical context. Use this whenever the user asks for synthetic patient data, "
            "sample lab results, or test healthcare records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {
                    "type": "number",
                    "description": "Patient age in years (0-90)"
                },
                "gender_male": {
                    "type": "boolean",
                    "description": "True for male, False for female"
                },
                "careunit": {
                    "type": "string",
                    "enum": VALID_CAREUNITS,
                    "description": "ICU care unit"
                },
                "los_bucket": {
                    "type": "string",
                    "enum": VALID_LOS,
                    "description": "Expected length of stay bucket"
                },
                "ccs_group": {
                    "type": "string",
                    "enum": VALID_CCS,
                    "description": "Primary diagnosis CCS category"
                },
                "elix_flags": {
                    "type": "object",
                    "description": (
                        f"Optional Elixhauser comorbidity flags as {{name: 0 or 1}}. "
                        f"Valid keys: {', '.join(VALID_ELIX)}"
                    ),
                    "additionalProperties": {"type": "integer", "enum": [0, 1]}
                },
                "n_samples": {
                    "type": "integer",
                    "description": "Number of synthetic patient records to generate (1-50)",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5
                }
            },
            "required": ["age", "gender_male", "careunit", "los_bucket", "ccs_group"]
        }
    },
    {
        "name": "list_valid_options",
        "description": "Returns the valid enum values for careunit, los_bucket, ccs_group, and elix_flags. Call this when the user seems unsure about what values to use.",
        "input_schema": {"type": "object", "properties": {}}
    }
]


def process_tool_call(tool_name, tool_input):
    if tool_name == "generate_synthetic_patients":
        try:
            records = generate_patients(
                age          = tool_input["age"],
                gender_male  = tool_input["gender_male"],
                careunit     = tool_input["careunit"],
                los_bucket   = tool_input["los_bucket"],
                ccs_group    = tool_input["ccs_group"],
                elix_flags   = tool_input.get("elix_flags"),
                n_samples    = tool_input.get("n_samples", 5),
            )
            return json.dumps({
                "status": "success",
                "n_records": len(records),
                "patients": records
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    elif tool_name == "list_valid_options":
        return json.dumps({
            "careunits":  VALID_CAREUNITS,
            "los_buckets": VALID_LOS,
            "ccs_groups":  VALID_CCS,
            "elix_flags":  VALID_ELIX,
        })

    return json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})


# -- Chat endpoint -------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str                      # just the new user message
    session_id: Optional[str] = None  # omit to start a new session
    system: Optional[str] = None


SYSTEM_PROMPT = """You are a synthetic clinical data assistant powered by a cVAE (Conditional Variational Autoencoder) trained on MIMIC-III ICU data.

You can generate realistic synthetic first-day ICU lab records for any patient profile using the `generate_synthetic_patients` tool. The 23 labs cover:
- Core chemistry: glucose, sodium, potassium, creatinine, BUN, bicarbonate, chloride
- Calcium & electrolytes: calcium_total, free_calcium, phosphate
- Blood gas: pH, pCO2, pO2, lactate
- Hematology: hemoglobin, hematocrit, WBC, platelets
- Coagulation: PT, PTT
- Liver & nutrition: bilirubin, AST, albumin

When a user asks for synthetic patient data, use the tool immediately. If they are vague (e.g. "give me a sepsis patient"), make reasonable clinical assumptions and generate the data - then explain the choices made. Always present the output in a clean, readable table format.

For queries about available options (care units, diagnoses, comorbidities), use the `list_valid_options` tool.

Note: All data is synthetic and for research/testing only. Never claim it represents real patients."""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M UTC")


def _short_id(session_id: str) -> str:
    """Return last 8 chars of UUID for easy human reference."""
    return session_id[-8:].upper()


@app.post("/chat")
async def chat(req: ChatRequest):
    client = anthropic.Anthropic()
    system = req.system or SYSTEM_PROMPT

    # resolve or create session
    if req.session_id and req.session_id in SESSIONS:
        session = SESSIONS[req.session_id]
        session_id = req.session_id
    else:
        session_id = str(uuid.uuid4())
        session = {
            "messages":   [],
            "created_at": _now(),
            "updated_at": _now(),
            "label":      None,
        }
        SESSIONS[session_id] = session

    # auto-label from first user message (first 60 chars)
    if session["label"] is None:
        session["label"] = req.message[:60].strip()

    # append new user message to stored history
    messages = session["messages"]
    messages.append({"role": "user", "content": req.message})

    # agentic loop - handle tool use
    while True:
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 2048,
            system     = system,
            tools      = TOOLS,
            messages   = messages,
        )

        # serialize assistant content (Pydantic objects to plain dicts for storage)
        assistant_content = [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in response.content
        ]
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling {block.name} with inputs {block.input}")
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    # keep updated history and timestamp
    session["messages"]   = messages
    session["updated_at"] = _now()

    # extract final text
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    return {
        "response":   final_text,
        "session_id": session_id,
        "short_id":   _short_id(session_id),
    }


# -- Session management endpoints ----------------------------------------------

@app.get("/sessions")
async def list_sessions():
    """Return all active sessions as a summary list."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "short_id":   _short_id(sid),
                "label":      data["label"] or "(no label)",
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "turns":      sum(1 for m in data["messages"] if m["role"] == "user"),
            }
            for sid, data in SESSIONS.items()
        ]
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Restore a session by ID - returns full message history."""
    if session_id not in SESSIONS:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    data = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "short_id":   _short_id(session_id),
        "label":      data["label"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "turns":      sum(1 for m in data["messages"] if m["role"] == "user"),
        "messages":   data["messages"],
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in SESSIONS:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    del SESSIONS[session_id]
    return {"deleted": session_id}


@app.patch("/sessions/{session_id}/label")
async def rename_session(session_id: str, label: str):
    """Rename a session label."""
    if session_id not in SESSIONS:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    SESSIONS[session_id]["label"] = label
    return {"session_id": session_id, "label": label}


# -- Direct generation endpoint (no LLM) --------------------------------------

class GenerateRequest(BaseModel):
    age: float
    gender_male: bool
    careunit: str
    los_bucket: str
    ccs_group: str
    elix_flags: Optional[dict] = None
    n_samples: int = 5


@app.post("/generate")
async def generate(req: GenerateRequest):
    if req.careunit not in VALID_CAREUNITS:
        raise HTTPException(400, f"careunit must be one of {VALID_CAREUNITS}")
    if req.los_bucket not in VALID_LOS:
        raise HTTPException(400, f"los_bucket must be one of {VALID_LOS}")
    if req.ccs_group not in VALID_CCS:
        raise HTTPException(400, f"ccs_group must be one of {VALID_CCS}")

    records = generate_patients(
        age         = req.age,
        gender_male = req.gender_male,
        careunit    = req.careunit,
        los_bucket  = req.los_bucket,
        ccs_group   = req.ccs_group,
        elix_flags  = req.elix_flags,
        n_samples   = req.n_samples,
    )
    return {"n_records": len(records), "patients": records}


@app.get("/options")
async def options():
    return {
        "careunits":   VALID_CAREUNITS,
        "los_buckets": VALID_LOS,
        "ccs_groups":  VALID_CCS,
        "elix_flags":  VALID_ELIX,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}