"""
FastAPI application for Stack Overflow 2025 Developer Salary Prediction.

This API serves predictions from a trained scikit-learn Pipeline
(ColumnTransformer + GradientBoostingRegressor).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import logging
import joblib
import pandas as pd


# ============================================================
# Configuration
# ============================================================

# The model file must be in the same folder as main.py
MODEL_PATH = Path(__file__).resolve().parent / "Stackoverflow_salary_model.pkl"


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Allowed / mapped categories
# ============================================================

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
    "Age",
    "EdLevel",
    "RemoteWork",
    "Country",
    "Industry",
    "OrgSize",
    "ICorPM",
    "DevType",
    "WorkExp",
    "YearsCode",
    "AISent",
    "AISelect",
    "AIThreat",
    "AIComplex",
    "ToolCountWork",
    "JobSat",
    "MainBranch",
]


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Stack Overflow 2025 Salary Predictor",
    description=(
        "API for predicting yearly developer salary in USD "
        "using Stack Overflow 2025 survey data."
    ),
    version="1.0.0",
)


# ============================================================
# Model cache
# ============================================================

_model = None


def get_model():
    """
    Load the trained model once and keep it in memory.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            logger.error(
                "Model file not found: %s",
                MODEL_PATH
            )

            raise RuntimeError(
                f"Model file not found: {MODEL_PATH}"
            )

        logger.info(
            "Loading model from: %s",
            MODEL_PATH
        )

        _model = joblib.load(MODEL_PATH)

        logger.info("Model loaded successfully.")

    return _model


# ============================================================
# Request Schema
# ============================================================

class SalaryPredictionInput(BaseModel):

    Age: str = Field(
        ...,
        description="Age range",
        examples=["25-34 years old"],
    )

    EdLevel: str = Field(
        ...,
        description="Highest level of education",
        examples=[
            "Bachelor's degree (B.A., B.S., B.Eng., etc.)"
        ],
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
        examples=[
            "Good, but not great at handling complex tasks"
        ],
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
        description="Job satisfaction score",
        examples=[7.0],
    )

    MainBranch: str = Field(
        ...,
        description="Primary professional identity",
        examples=[
            "I am a developer by profession"
        ],
    )


# ============================================================
# Response Schema
# ============================================================

class SalaryPredictionOutput(BaseModel):

    predicted_salary_usd: float = Field(
        ...,
        description="Predicted yearly salary in USD"
    )

    input_features: dict = Field(
        ...,
        description="Features used for prediction"
    )


# ============================================================
# Helper Functions
# ============================================================

def map_country(country: str) -> str:
    """
    Keep top-20 countries.
    Map all other countries to 'Other'.
    """

    return (
        country
        if country in TOP_20_COUNTRIES
        else "Other"
    )


def map_devtype(devtype: str) -> str:
    """
    Keep supported developer types.
    Map all others to 'Other'.
    """

    return (
        devtype
        if devtype in TOP_10_DEV_TYPES
        else "Other"
    )


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/", tags=["Info"])
async def root():

    return {
        "project": "Stack Overflow 2025 Developer Salary Predictor",
        "version": "1.0.0",
        "status": "online",
        "model": (
            "ColumnTransformer + "
            "GradientBoostingRegressor"
        ),
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "prediction": "POST /predict",
        },
        "features": ALL_FEATURES,
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health", tags=["Info"])
async def health():

    try:

        model = get_model()

        return {
            "status": "healthy",
            "model_loaded": model is not None,
        }

    except Exception as exc:

        logger.exception(
            "Health check failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}",
        )


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=SalaryPredictionOutput,
    tags=["Prediction"],
)
async def predict(
    data: SalaryPredictionInput
):

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = get_model()

    except Exception as exc:

        logger.exception(
            "Model loading failed"
        )

        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}",
        )

    # --------------------------------------------------------
    # Map Country and Developer Type
    # --------------------------------------------------------

    mapped_country = map_country(
        data.Country
    )

    mapped_devtype = map_devtype(
        data.DevType
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        [features],
        columns=ALL_FEATURES
    )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(df)

        predicted_salary = round(
            float(prediction[0]),
            2
        )

    except Exception as exc:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return SalaryPredictionOutput(

        predicted_salary_usd=predicted_salary,

        input_features=features,
    )
