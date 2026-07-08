# 📂 Dataset

This project uses the **Sentiment Analysis for Mental Health** dataset published on Kaggle. The dataset consists of labeled textual statements collected from multiple public sources, including social media platforms such as Reddit and Twitter, for multi-class mental health classification. :contentReference[oaicite:0]{index=0}

---

## 📊 Dataset Information

| Attribute | Details |
|-----------|---------|
| **Dataset Name** | Sentiment Analysis for Mental Health |
| **Author** | Suchintika Sarkar |
| **Source** | Kaggle |
| **Task** | Multi-class Text Classification |
| **Language** | English |
| **File Format** | CSV |
| **Total Samples** | 53,043 |
| **Classes** | 7 Mental Health Categories |

---

## 🧠 Mental Health Categories

The dataset contains the following seven mental health classes:

- Anxiety
- Bipolar
- Depression
- Normal
- Personality Disorder
- Stress
- Suicidal

---

## 📁 Dataset Structure

| Column | Description |
|--------|-------------|
| `statement` | Text statement or social media post |
| `status` | Mental health category label |

---

## 📥 Download Dataset

The original dataset is **not included** in this repository due to Kaggle's licensing and distribution policies.

Download it directly from Kaggle:

👉 **Dataset Page**

[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

---

## 📂 Expected File Structure

After downloading, place the dataset as shown below:

```text
dataset/
└── Combined Data.csv
```

---

## 🚀 Using the Dataset

1. Download the dataset from Kaggle.
2. Extract the downloaded archive.
3. Copy **Combined Data.csv** into the project dataset directory.
4. Run the notebook or FastAPI application.

---

## 📄 Sample Dataset

A small sample dataset (`sample_Combined_Data.csv`) containing **2,000 rows** is included in this repository for quick testing and demonstration purposes.

---

## 📚 Citation

If you use this dataset in your research or project, please cite the original Kaggle dataset:

> Suchintika Sarkar. *Sentiment Analysis for Mental Health*. Kaggle. :contentReference[oaicite:2]{index=2}

---

## ⚠️ Disclaimer

This dataset is intended **solely for research and educational purposes**.

Predictions generated using this project **must not** be considered a substitute for professional mental health assessment, diagnosis, or treatment.
