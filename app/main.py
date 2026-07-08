import os
import re
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from bs4 import BeautifulSoup
import contractions
import emoji
import dill
import warnings
from scipy.special import expit, softmax

warnings.filterwarnings('ignore')

# Preprocessing function
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Define the paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "saved_models")

print("Loading models (this might take a few seconds)...")
models_loaded = False
NUM_CLASSES = 7
try:
    ml_model = joblib.load(os.path.join(MODELS_DIR, "best_ml_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    dl_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_dl_model.keras"))
    tokenizer = joblib.load(os.path.join(MODELS_DIR, "tokenizer.pkl"))
    if os.path.exists(os.path.join(MODELS_DIR, "label_encoder.pkl")):
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    else:
        label_encoder = None

    # Load config for max sequence length
    if os.path.exists(os.path.join(MODELS_DIR, "dl_config.pkl")):
        dl_config = joblib.load(os.path.join(MODELS_DIR, "dl_config.pkl"))
        MAX_LEN = dl_config.get("max_sequence_length", 190)
    elif os.path.exists(os.path.join(MODELS_DIR, "config.pkl")):
        config = joblib.load(os.path.join(MODELS_DIR, "config.pkl"))
        MAX_LEN = config.get("max_sequence_length", 190)
    else:
        MAX_LEN = 190

    models_loaded = True
    print(f"Models loaded successfully. MAX_LEN={MAX_LEN}")
    if label_encoder is not None:
        print(f"Label Encoder classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
        NUM_CLASSES = len(label_encoder.classes_)
except Exception as e:
    print(f"Error loading models: {e}")
    MAX_LEN = 190

# 7-class fallback map matching the label encoder order
FALLBACK_CLASS_MAP = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality Disorder",
    5: "Stress",
    6: "Suicidal"
}

# -------------------------------------------------------------------
# Helper: get probabilities from any sklearn-style model
# (mirrors the notebook's get_model_probabilities)
# -------------------------------------------------------------------
def get_model_probabilities(model, X_matrix):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_matrix)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_matrix)
        if scores.ndim == 1:
            p1 = expit(scores)
            return np.column_stack([1 - p1, p1])
        return softmax(scores, axis=1)

    preds = model.predict(X_matrix)
    probs = np.zeros((len(preds), NUM_CLASSES))
    probs[np.arange(len(preds)), preds] = 1.0
    return probs

# -------------------------------------------------------------------
# Explainability helper functions (used by SHAP & LIME)
# -------------------------------------------------------------------
def transform_texts_for_ml(texts):
    """Clean texts and vectorize with TF-IDF."""
    cleaned = [clean_text(t) for t in texts]
    return vectorizer.transform(cleaned)

def predict_ml_proba(texts):
    """Predict probabilities with the ML model (for LIME / SHAP)."""
    X_matrix = transform_texts_for_ml(texts)
    return get_model_probabilities(ml_model, X_matrix)

def predict_dl_proba(texts):
    """Predict probabilities with the DL model (for LIME / SHAP)."""
    cleaned = [clean_text(t) for t in texts]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post", dtype="int32")
    return dl_model.predict(padded, batch_size=32, verbose=0)

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class PredictRequest(BaseModel):
    text: str
    model: str = "traditional" 

class ExplainRequest(BaseModel):
    text: str
    model: str = "traditional"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/predict")
async def predict(req: PredictRequest):
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are not loaded. Check server logs for details."}
        )
    cleaned = clean_text(req.text)
    results = []
    
    if req.model in ["deep", "both"]:
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post", dtype="int32")
        pred = dl_model.predict(padded)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][class_idx])
        
        if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
            prediction_label = label_encoder.inverse_transform([class_idx])[0]
        else:
            prediction_label = FALLBACK_CLASS_MAP.get(class_idx, "Unknown")
            
        results.append({
            "model_name": "Deep Learning",
            "prediction": prediction_label, 
            "confidence": confidence
        })
        
    if req.model in ["traditional", "both"]:
        vec = vectorizer.transform([cleaned])
        confidence = None
        
        if hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(vec)
            class_idx = np.argmax(probs, axis=1)[0]
            confidence = float(probs[0][class_idx])
            pred_class = ml_model.classes_[class_idx]
            prediction_label = str(pred_class)
        elif hasattr(ml_model, "decision_function"):
            scores = ml_model.decision_function(vec)
            if len(scores.shape) == 1:
                scores = np.array([scores])
            probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            class_idx = np.argmax(probs, axis=1)[0]
            confidence = float(probs[0][class_idx])
            pred_class = ml_model.classes_[class_idx]
            prediction_label = str(pred_class)
        else:
            pred = ml_model.predict(vec)
            prediction_label = str(pred[0])

        # Decode numeric labels via label encoder
        if prediction_label.isdigit() and label_encoder is not None:
            prediction_label = label_encoder.inverse_transform([int(prediction_label)])[0]
        elif prediction_label.isdigit():
            prediction_label = FALLBACK_CLASS_MAP.get(int(prediction_label), "Unknown")
            
        results.append({
            "model_name": "Traditional ML",
            "prediction": prediction_label, 
            "confidence": confidence
        })
        
    return {"results": results}


# -------------------------------------------------------------------
# /explain — SHAP + LIME explainability endpoint
# -------------------------------------------------------------------
@app.post("/explain")
async def explain(req: ExplainRequest):
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are not loaded."}
        )

    import shap
    from lime.lime_text import LimeTextExplainer

    class_names = list(label_encoder.classes_) if label_encoder is not None else list(FALLBACK_CLASS_MAP.values())
    explanations = []

    # ---- Traditional ML ----
    if req.model in ["traditional", "both"]:
        try:
            # Predicted class
            probs = predict_ml_proba([req.text])
            pred_class_idx = int(np.argmax(probs[0]))
            pred_class_name = class_names[pred_class_idx]

            # LIME
            lime_explainer = LimeTextExplainer(class_names=class_names)
            lime_exp = lime_explainer.explain_instance(
                req.text,
                predict_ml_proba,
                num_features=15,
                top_labels=1,
            )
            top_label = lime_exp.available_labels()[0]
            lime_pairs = lime_exp.as_list(label=top_label)

            # SHAP
            masker = shap.maskers.Text(r"\W+")
            shap_explainer = shap.Explainer(
                predict_ml_proba,
                masker,
                output_names=class_names,
            )
            shap_values = shap_explainer(
                [req.text],
                max_evals=500,
                batch_size=16,
            )
            sv = shap_values[0, :, pred_class_idx]
            shap_pairs = list(zip(
                [str(v) for v in sv.data],
                [float(v) for v in sv.values],
            ))
            # Sort by absolute value descending, keep top 15
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            shap_pairs = shap_pairs[:15]

            explanations.append({
                "model_name": "Traditional ML",
                "predicted_class": pred_class_name,
                "lime": [[w, round(float(v), 4)] for w, v in lime_pairs],
                "shap": [[w, round(float(v), 4)] for w, v in shap_pairs],
            })
        except Exception as e:
            print(f"Error generating ML explanations: {e}")
            explanations.append({
                "model_name": "Traditional ML",
                "predicted_class": "Error",
                "lime": [],
                "shap": [],
                "error": str(e),
            })

    # ---- Deep Learning ----
    if req.model in ["deep", "both"]:
        try:
            # Predicted class
            probs = predict_dl_proba([req.text])
            pred_class_idx = int(np.argmax(probs[0]))
            pred_class_name = class_names[pred_class_idx]

            # LIME
            lime_explainer = LimeTextExplainer(class_names=class_names)
            lime_exp = lime_explainer.explain_instance(
                req.text,
                predict_dl_proba,
                num_features=15,
                top_labels=1,
            )
            top_label = lime_exp.available_labels()[0]
            lime_pairs = lime_exp.as_list(label=top_label)

            # SHAP
            masker = shap.maskers.Text(r"\W+")
            shap_explainer = shap.Explainer(
                predict_dl_proba,
                masker,
                output_names=class_names,
            )
            shap_values = shap_explainer(
                [req.text],
                max_evals=500,
                batch_size=16,
            )
            sv = shap_values[0, :, pred_class_idx]
            shap_pairs = list(zip(
                [str(v) for v in sv.data],
                [float(v) for v in sv.values],
            ))
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            shap_pairs = shap_pairs[:15]

            explanations.append({
                "model_name": "Deep Learning",
                "predicted_class": pred_class_name,
                "lime": [[w, round(float(v), 4)] for w, v in lime_pairs],
                "shap": [[w, round(float(v), 4)] for w, v in shap_pairs],
            })
        except Exception as e:
            print(f"Error generating DL explanations: {e}")
            explanations.append({
                "model_name": "Deep Learning",
                "predicted_class": "Error",
                "lime": [],
                "shap": [],
                "error": str(e),
            })

    return {"explanations": explanations}


if __name__ == "__main__":
    print("Starting Mental Health Detection Web App...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
