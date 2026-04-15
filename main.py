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
    description="Wine classifier prediction API",
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
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    class Config:
        example = {
            "alcohol": 13.0,
            "malic_acid": 2.34,
            "ash": 2.36,
            "alcalinity_of_ash": 19.5,
            "magnesium": 99.7,
            "total_phenols": 2.30,
            "flavanoids": 2.03,
            "nonflavanoid_phenols": 0.36,
            "proanthocyanins": 1.59,
            "color_intensity": 5.06,
            "hue": 0.96,
            "od280_od315_of_diluted_wines": 2.61,
            "proline": 746.0
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: int
    confidence: float
    wine_class: str

class HealthResponse(BaseModel):
    """Response model for health endpoint"""
    status: str
    model_loaded: bool

# Wine class mapping
WINE_CLASSES = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2"
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

    Takes wine chemical measurements and returns the predicted wine class
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    # Prepare input features
    features = np.array([[
        request.alcohol,
        request.malic_acid,
        request.ash,
        request.alcalinity_of_ash,
        request.magnesium,
        request.total_phenols,
        request.flavanoids,
        request.nonflavanoid_phenols,
        request.proanthocyanins,
        request.color_intensity,
        request.hue,
        request.od280_od315_of_diluted_wines,
        request.proline
    ]])

    # Normalize features (same as training)
    feature_means = np.array([13.0006, 2.3363, 2.3665, 19.4949, 99.7416,
                               2.2951, 2.0293, 0.3619, 1.5909, 5.0581,
                               0.9574, 2.6117, 746.8933])
    feature_stds = np.array([0.8118, 1.1171, 0.2743, 3.3396, 14.2825,
                              0.6257, 0.9989, 0.1244, 0.5726, 2.3183,
                              0.2285, 0.7099, 314.9075])
    features_scaled = (features - feature_means) / feature_stds

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))
    wine_class = WINE_CLASSES[prediction]

    return PredictionResponse(
        prediction=int(prediction),
        confidence=confidence,
        wine_class=wine_class
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
