from IPython import get_ipython
from IPython.display import display

!pip install pandas nltk matplotlib seaborn tensorflow

!pip install tensorflow

!pip install --upgrade tensorflow

!pip install shap

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')

# Step 1: Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path, low_memory=False)
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns}")

    data = data.iloc[:, [1, 3]]
    data.columns = ['text', 'label']
    data['label'] = data['label'].map({0: 'No Issues', 1: 'Mental Health Issues'})  # Binary encoding
    data = data.dropna(subset=['label'])
    print("First few rows of the data:")
    print(data.head())
    return data

# Step 2: Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+|[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Step 3: Prepare the data for deep learning models
def prepare_data(data, max_words=10000, max_len=100):
    data['text'] = data['text'].apply(preprocess_text)
    texts = data['text'].values
    labels = data['label'].map({'No Issues': 0, 'Mental Health Issues': 1}).values

    print(f"Texts shape: {texts.shape}")
    print(f"Labels shape: {labels.shape}")

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    print(f"Padded sequences shape: {padded_sequences.shape}")
    return padded_sequences, labels, tokenizer

# Step 4: Define Deep Learning Models
# CNN Model
def create_cnn_model(max_words):
    model = Sequential([
        Embedding(max_words, 128),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# LSTM Model
def create_lstm_model(max_words):
    model = Sequential([
        Embedding(max_words, 128),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Bidirectional GRU Model
def create_rnn_model(max_words):
    model = Sequential([
        Embedding(max_words, 128),
        Bidirectional(GRU(128, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Train and Evaluate Models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype('int32')


# Plot Accuracy and Loss for all models
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title(f'Training and Validation Accuracy - {model_name}') # Model-specific title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{model_name}_accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'Training and Validation Loss - {model_name}')  # Model-specific title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_name}_loss.png')
    plt.show()


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)


    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {roc_auc:.2f}")

    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "conf_matrix": conf_matrix,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }

# Step 6: Plot Confusion Matrix and ROC Curve
def plot_performance_metrics(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    model_names = list(results.keys())
    scores = {metric: [results[model_name][metric] for model_name in model_names] for metric in metrics}

    x = np.arange(len(model_names))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, scores[metric], width, label=metric_names[i])

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('Model_Performance_Comparison.png')
    plt.show()

def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Issues", "Mental Health Issues"], yticklabels=["No Issues", "Mental Health Issues"])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

def plot_roc_curves(results):
    plt.figure(figsize=(8, 6))
    for model_name, result in results.items():
        plt.plot(result['fpr'], result['tpr'], lw=2, label=f'{model_name} (AUC = {result["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('combined_roc_curve.png')
    plt.show()

# Model Explainability
def explain_models(results, X, y, tokenizer):
    best_model_name = max(results, key=lambda name: results[name]['accuracy'])  # Assuming 'accuracy' is the key for the model's accuracy
    model = results[best_model_name]['model']

    # Split the data inside explain_models to make X_test available
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Change here to access the element using .toarray() and indexing


    # Subset to explain
    subset = X_train[0:100] #.toarray() # Changed here to use slicing
    #subset = shap.sample(X_train, 100)
    try:
      model.predict(subset).shape[1]
    except IndexError:
      #explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output),subset)
      explainer = shap.KernelExplainer(model, subset)
    else:
      #explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), subset)
      explainer = shap.KernelExplainer(model, subset)


    #explainer = shap.DeepExplainer(model, subset)

    # Get SHAP values
    shap_values =explainer.shap_values(subset) #, check_additivity = False)

    shap.initjs()

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, subset)
    plt.savefig(f'{best_model_name}_SHAP_summary.png', dpi=300, bbox_inches = "tight")
    plt.show()

# Main Function
def main():
    file_path = 'Final_Cleaned.csv'
    data = load_data(file_path)

    if data is None:
        return

    max_words = 10000
    X, y, tokenizer = prepare_data(data, max_words)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        #"CNN": create_cnn_model(max_words),
        #"LSTM": create_lstm_model(max_words),
        "BiGRU": create_rnn_model(max_words)
    }

    results = {}
    for model_name, model_instance in models.items():
        results[model_name] = train_and_evaluate_model(model_instance, X_train, X_test, y_train, y_test, model_name)

    plot_performance_metrics(results)# Call the new function to plot performance metrics
    plot_roc_curves(results)# Call the new function to plot combined ROC curves

    for model_name, result in results.items():
        plot_confusion_matrix(result['conf_matrix'], model_name)

    best_model_name = max(results, key=lambda name: results[name]['accuracy'])
    print(f"The best model is: {best_model_name}")

    explain_models({best_model_name: results[best_model_name]}, X, y, tokenizer)

    print("Welcome to the Mental Health Detection System!")
    while True:
        text = input("Please enter a sentence (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        if not text.strip():
            print("Input cannot be empty. Please try again.")
            continue

        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        best_model = results[best_model_name]["model"]
        prediction_prob = best_model.predict(padded_sequence)
        prediction = "Mental Health Issues" if prediction_prob > 0.5 else "No Issues"
        print(f"Prediction: {prediction} (Confidence: {prediction_prob[0][0]:.2f})")

if __name__ == "__main__":
    main()