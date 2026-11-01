from pathlib import Path
import warnings
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "PBL_5_sem_phase_1_model.pkl"

NUMERIC_COLUMNS = [
    "Age", "Years at Company", "Monthly Income", "Number of Promotions",
    "Number of Dependents", "Company Tenure", "Distance from Home"
]
CATEGORICAL_COLUMNS = [
    "Gender", "Job Role", "Work-Life Balance", "Job Satisfaction",
    "Performance Rating", "Overtime", "Education Level", "Marital Status",
    "Job Level", "Company Size", "Remote Work", "Leadership Opportunities",
    "Innovation Opportunities", "Company Reputation", "Employee Recognition"
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

app = FastAPI(title="Employee Attrition Predictor", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class EmployeeInput(BaseModel):
    age: int = Field(ge=18, le=100, alias="Age")
    years_at_company: int = Field(ge=0, le=60, alias="Years at Company")
    monthly_income: float = Field(ge=0, le=1000000, alias="Monthly Income")
    number_of_promotions: int = Field(ge=0, le=30, alias="Number of Promotions")
    number_of_dependents: int = Field(ge=0, le=20, alias="Number of Dependents")
    company_tenure: int = Field(ge=0, le=100, alias="Company Tenure")
    distance_from_home: float = Field(ge=0, le=1000, alias="Distance from Home")
    gender: str = Field(alias="Gender")
    job_role: str = Field(alias="Job Role")
    work_life_balance: str = Field(alias="Work-Life Balance")
    job_satisfaction: str = Field(alias="Job Satisfaction")
    performance_rating: str = Field(alias="Performance Rating")
    overtime: str = Field(alias="Overtime")
    education_level: str = Field(alias="Education Level")
    marital_status: str = Field(alias="Marital Status")
    job_level: str = Field(alias="Job Level")
    company_size: str = Field(alias="Company Size")
    remote_work: str = Field(alias="Remote Work")
    leadership_opportunities: str = Field(alias="Leadership Opportunities")
    innovation_opportunities: str = Field(alias="Innovation Opportunities")
    company_reputation: str = Field(alias="Company Reputation")
    employee_recognition: str = Field(alias="Employee Recognition")

    model_config = {"populate_by_name": True, "extra": "forbid"}


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=160, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ])


def ensure_model():
    # The supplied .pkl is checked for fitted state. If it is unfitted,
    # create and save a complete preprocessing + classification pipeline.
    if MODEL_PATH.exists():
        try:
            candidate = joblib.load(MODEL_PATH)
            if hasattr(candidate, "classes_") and hasattr(candidate, "predict"):
                return candidate
        except Exception:
            pass
    if not DATA_PATH.exists():
        raise RuntimeError("Model is unfitted and train.csv is missing. Place train.csv beside main.py.")
    data = pd.read_csv(DATA_PATH)
    required = FEATURE_COLUMNS + ["Attrition"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"train.csv is missing columns: {missing}")
    data = data.dropna(subset=["Attrition"])
    model = build_pipeline()
    model.fit(data[FEATURE_COLUMNS], data["Attrition"].astype(str).str.strip())
    joblib.dump(model, MODEL_PATH)
    return model


try:
    model = ensure_model()
    MODEL_ERROR = None
except Exception as exc:
    model, MODEL_ERROR = None, str(exc)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "message": "Prediction API is ready." if model is not None else MODEL_ERROR,
    }


@app.post("/api/predict")
def predict(employee: EmployeeInput):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {MODEL_ERROR}")
    payload = employee.model_dump(by_alias=True)
    row = pd.DataFrame([{c: payload[c] for c in FEATURE_COLUMNS}])
    try:
        prediction = str(model.predict(row)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row)[0]
            classes = [str(c) for c in model.classes_]
            if "Left" in classes:
                probability = float(probs[classes.index("Left")]) * 100
            elif "Stayed" in classes:
                probability = (1 - float(probs[classes.index("Stayed")])) * 100
        left = prediction.lower() == "left"
        return {
            "prediction": prediction,
            "risk": "Higher attrition likelihood" if left else "Lower attrition likelihood",
            "attrition_probability": round(probability, 2) if probability is not None else None,
            "note": "This is a model estimate, not a certainty or a judgment about an individual."
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
