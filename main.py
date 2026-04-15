"""
FastAPI application for model serving
Exposes health check and prediction endpoints
"""

import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Configuration
MODEL_PATH = "model.pkl"

# Create FastAPI app
app = FastAPI(
    title="MLOps Model Server",
    description="Iris classifier prediction API",
    version="1.0.0"
)

# Load model at startup
model = None

def load_model():
    """Load trained model from disk"""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {MODEL_PATH}")

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    try:
        load_model()
    except Exception as e:
        print(f"✗ Error loading model: {e}")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        example = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: int
    confidence: float
    iris_species: str

class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str
    model_loaded: bool

# Iris species mapping
IRIS_SPECIES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prediction endpoint

    Takes iris flower measurements and returns the predicted species
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    # Prepare input features
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    # Normalize features (same as training)
    # Note: In production, use a fitted scaler saved with the model
    # For this exercise, we use hardcoded training set stats
    feature_means = np.array([5.843333, 3.054, 3.758667, 1.198667])
    feature_stds = np.array([0.816497, 0.432877, 1.758047, 0.763161])
    features_scaled = (features - feature_means) / feature_stds

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))
    species = IRIS_SPECIES[prediction]

    return PredictionResponse(
        prediction=int(prediction),
        confidence=confidence,
        iris_species=species
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
