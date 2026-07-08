import os
import joblib
import tensorflow as tf

base_dir = r"C:\Users\Bijoy\Desktop\PROJECTS\Explainable_AI_for_Mental_Health_Detection_from_Social_Media_Post\saved_models"

print("Loading Traditional ML Model...")
try:
    ml_model = joblib.load(os.path.join(base_dir, "best_ml_model.pkl"))
    print("ML Model type:", type(ml_model))
    if hasattr(ml_model, 'classes_'):
        print("ML Model classes:", list(ml_model.classes_))
except Exception as e:
    print("Error loading ML model:", e)

print("\nLoading Deep Learning Model...")
try:
    dl_model = tf.keras.models.load_model(os.path.join(base_dir, "best_dl_model.keras"))
    print("DL Model summary:")
    dl_model.summary()
except Exception as e:
    print("Error loading DL model:", e)

print("\nLoading Vectorizer...")
try:
    vectorizer = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))
    print("Vectorizer type:", type(vectorizer))
except Exception as e:
    print("Error loading vectorizer:", e)

print("\nLoading Tokenizer...")
try:
    tokenizer = joblib.load(os.path.join(base_dir, "tokenizer.pkl"))
    print("Tokenizer type:", type(tokenizer))
except Exception as e:
    print("Error loading tokenizer:", e)

print("\nLoading Label Encoder...")
try:
    le = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
    print("Label Encoder classes:", list(le.classes_))
    print("Number of classes:", len(le.classes_))
except Exception as e:
    print("Error loading label encoder:", e)

print("\nLoading Configs...")
try:
    config = joblib.load(os.path.join(base_dir, "config.pkl"))
    print("ML Config:", config)
except Exception as e:
    print("Error loading config:", e)

try:
    dl_config = joblib.load(os.path.join(base_dir, "dl_config.pkl"))
    print("DL Config:", dl_config)
except Exception as e:
    print("Error loading dl_config:", e)
