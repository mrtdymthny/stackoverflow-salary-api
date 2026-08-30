"""
FastAPI application for Stack Overflow 2025 Developer Salary Prediction.
Serves predictions from a trained sklearn Pipeline (ColumnTransformer + GradientBoostingRegressor).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
from pathlib import Path
import logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent.parent / "stackoverflow_salary_model.pkl"

TOP_20_COUNTRIES = {
    "United States of America",
    "Germany",
    "India",
    "United Kingdom of Great Britain and Northern Ireland",
    "France",
    "Canada",
    "Ukraine",
    "Poland",
    "Netherlands",
    "Italy",
    "Brazil",
    "Australia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Israel",
    "Ireland",
    "Norway",
    "Denmark",
    "Japan",
}

TOP_10_DEV_TYPES = {
    "Developer, full-stack",
    "Developer, back-end",
    "Student",
    "Architect, software or solutions",
    "Developer, front-end",
    "Developer, desktop or enterprise applications",
    "Other (please specify):",
    "Developer, mobile",
    "Developer, embedded applications or devices",
    "Academic researcher",
    "Engineering manager",
}

ALL_FEATURES = [
    "Age", "EdLevel", "RemoteWork", "Country", "Industry", "OrgSize",
    "ICorPM", "DevType", "WorkExp", "YearsCode", "AISent", "AISelect",
    "AIThreat", "AIComplex", "ToolCountWork", "JobSat", "MainBranch",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stack Overflow 2025 Salary Predictor",
    description="Predict yearly developer salary (USD) using a trained Gradient Boosting model on Stack Overflow 2025 survey data.",
    version="1.0.0",
)

# Lazy-loaded model singleton
_model = None


def get_model():
    """Load and cache the model on first call."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            logger.error("Model file not found at %s", MODEL_PATH)
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Pydantic Schema
# ---------------------------------------------------------------------------

class SalaryPredictionInput(BaseModel):
    """Input schema for salary prediction with all 17 features."""

    Age: str = Field(
        ...,
        description="Age range",
        examples=["25-34 years old"],
    )
    EdLevel: str = Field(
        ...,
        description="Highest level of education",
        examples=["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
    )
    RemoteWork: str = Field(
        ...,
        description="Remote work arrangement",
        examples=["Remote"],
    )
    Country: str = Field(
        ...,
        description="Country of residence",
        examples=["United States of America"],
    )
    Industry: str = Field(
        ...,
        description="Industry sector",
        examples=["Software Development"],
    )
    OrgSize: str = Field(
        ...,
        description="Organization size",
        examples=["100 to 499 employees"],
    )
    ICorPM: str = Field(
        ...,
        description="Individual contributor or people manager",
        examples=["Individual contributor"],
    )
    DevType: str = Field(
        ...,
        description="Developer type / role",
        examples=["Developer, full-stack"],
    )
    WorkExp: float = Field(
        ...,
        ge=0,
        le=50,
        description="Years of professional work experience",
        examples=[5.0],
    )
    YearsCode: float = Field(
        ...,
        ge=0,
        le=50,
        description="Total years coding",
        examples=[8.0],
    )
    AISent: str = Field(
        ...,
        description="Sentiment toward AI tools",
        examples=["Favorable"],
    )
    AISelect: str = Field(
        ...,
        description="AI tool usage frequency",
        examples=["Yes, I use AI tools daily"],
    )
    AIThreat: str = Field(
        ...,
        description="Whether AI is seen as a threat to jobs",
        examples=["No"],
    )
    AIComplex: str = Field(
        ...,
        description="Perception of AI handling complex tasks",
        examples=["Good, but not great at handling complex tasks"],
    )
    ToolCountWork: float = Field(
        ...,
        ge=0,
        le=50,
        description="Number of tools used at work",
        examples=[10.0],
    )
    JobSat: float = Field(
        ...,
        ge=0,
        le=10,
        description="Job satisfaction score (0-10)",
        examples=[7.0],
    )
    MainBranch: str = Field(
        ...,
        description="Primary professional identity",
        examples=["I am a developer by profession"],
    )


class SalaryPredictionOutput(BaseModel):
    """Output schema for salary prediction."""
    predicted_salary_usd: float = Field(
        ..., description="Predicted yearly salary in USD"
    )
    input_features: dict = Field(
        ..., description="The features used for prediction (after mapping)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_country(country: str) -> str:
    """Map country to top-20 or 'Other'."""
    return country if country in TOP_20_COUNTRIES else "Other"


def map_devtype(devtype: str) -> str:
    """Map DevType to top-10 or 'Other'."""
    return devtype if devtype in TOP_10_DEV_TYPES else "Other"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
async def root():
    """Return project information."""
    return {
        "project": "Stack Overflow 2025 Developer Salary Predictor",
        "version": "1.0.0",
        "description": (
            "Predicts yearly developer salary (USD) using a Gradient Boosting Regressor "
            "trained on Stack Overflow 2025 survey data (17 features)."
        ),
        "model": "ColumnTransformer + GradientBoostingRegressor (sklearn Pipeline)",
        "endpoints": {
            "predict": "POST /predict  — submit features, get salary prediction",
            "health": "GET /health  — service health check",
        },
        "features": ALL_FEATURES,
    }


@app.get("/health", tags=["info"])
async def health():
    """Health check endpoint."""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")


@app.post("/predict", response_model=SalaryPredictionOutput, tags=["prediction"])
async def predict(data: SalaryPredictionInput):
    """
    Predict yearly salary in USD based on developer survey features.

    Country and DevType are automatically mapped to 'Other' if they are not
    in the top-20 / top-10 lists the model was trained on.
    """
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")

    # Apply Country / DevType mapping
    mapped_country = map_country(data.Country)
    mapped_devtype = map_devtype(data.DevType)

    # Build feature dict in model's expected column order
    features = {
        "Age": data.Age,
        "EdLevel": data.EdLevel,
        "RemoteWork": data.RemoteWork,
        "Country": mapped_country,
        "Industry": data.Industry,
        "OrgSize": data.OrgSize,
        "ICorPM": data.ICorPM,
        "DevType": mapped_devtype,
        "WorkExp": data.WorkExp,
        "YearsCode": data.YearsCode,
        "AISent": data.AISent,
        "AISelect": data.AISelect,
        "AIThreat": data.AIThreat,
        "AIComplex": data.AIComplex,
        "ToolCountWork": data.ToolCountWork,
        "JobSat": data.JobSat,
        "MainBranch": data.MainBranch,
    }

    # Create a single-row DataFrame in correct column order
    import pandas as pd
    df = pd.DataFrame([features], columns=ALL_FEATURES)

    try:
        prediction = model.predict(df)
        predicted_salary = round(float(prediction[0]), 2)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        )

    return SalaryPredictionOutput(
        predicted_salary_usd=predicted_salary,
        input_features=features,
    )
