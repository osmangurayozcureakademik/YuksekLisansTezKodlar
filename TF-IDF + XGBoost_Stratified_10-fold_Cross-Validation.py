import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Memory

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# Genel Ayarlar

DATA_PATH = "final_dataset.csv"
RANDOM_STATE = 7
CACHE_DIRECTORY = "/content/xgboost_evaluation_cache"


# Veri Setinin Yüklenmesi

df = pd.read_csv(DATA_PATH)

required_columns = {"Payload", "attack_type"}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Veri setinde eksik sütunlar bulunmaktadır: "
        f"{sorted(missing_columns)}"
    )

if df["Payload"].isna().any():
    raise ValueError(
        "Payload sütununda eksik değer bulunmaktadır."
    )

if df["attack_type"].isna().any():
    raise ValueError(
        "attack_type sütununda eksik değer bulunmaktadır."
    )

# X ve y'nin tanımlanması

X = df["Payload"].astype(str)
y = df["attack_type"].astype(str)

print("Toplam gözlem sayısı:", len(df))

print("\nSınıf dağılımı:")
print(y.value_counts())

# Sınıf etiketlerinin sayısal forma dönüştürülmesi

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = list(label_encoder.classes_)
encoded_labels = np.arange(len(class_names))

print("\nSınıf eşleştirmeleri:")

for encoded_label, class_name in enumerate(class_names):
    print(f"{encoded_label}: {class_name}")

# TF-IDF önbelleğinin tanımlanması

memory = Memory(
    location=CACHE_DIRECTORY,
    verbose=0
)


# TF-IDF + XGBoost Pipeline

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
                max_features=10000,
                sublinear_tf=True,
                dtype=np.float32
            )
        ),

        (
            "xgb",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                booster="gbtree",
                tree_method="hist",

                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                n_estimators=200,

                n_jobs=1,

                random_state=RANDOM_STATE,
                verbosity=0
            )
        )
    ],

    memory=memory
)

# Tabakalı 10 Katlı Çapraz Doğrulama

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Performans ölçütlerinin tanımlanması

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted"
}


# Çapraz doğrulama sonuçlarının elde edilmesi

print(
    "\n Tabakalı 10-katlı çapraz doğrulama başlatılıyor"
)

cv_results = cross_validate(
    estimator=model,
    X=X,
    y=y_encoded,
    cv=cv,
    scoring=scoring,

    # Katlar paralel çalıştırılır.
    n_jobs=-1,
    pre_dispatch="n_jobs",

    return_train_score=False,
    error_score="raise"
)

# Katlara ait ortalama ve standart sapma değerleri

print(
    "\n Tabakalı 10 Katlı Çapraz Doğrulama Sonuçları:"
)

cv_summary_rows = []

for metric_name in scoring:
    metric_scores = cv_results[f"test_{metric_name}"]

    mean_score = metric_scores.mean()
    std_score = metric_scores.std()

    print(
        f"{metric_name}: "
        f"{mean_score:.4f} ± {std_score:.4f}"
    )

    cv_summary_rows.append({
        "metric": metric_name,
        "mean": mean_score,
        "standard_deviation": std_score
    })


# Çapraz Doğrulama Özetinin Kaydedilmesi

cv_summary_df = pd.DataFrame(cv_summary_rows)

cv_summary_df.to_csv(
    "xgboost_tfidf_multiclass_cv_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nÇapraz doğrulama sonuçları CSV olarak kaydedildi."
)

# Her Gözlem için Çapraz Doğrulama Dışı Tahminler

print(
    "\nÇapraz doğrulama dışı tahminler elde ediliyor..."
)

y_pred_encoded = cross_val_predict(
    estimator=model,
    X=X,
    y=y_encoded,
    cv=cv,

    n_jobs=-1,
    pre_dispatch="n_jobs",

    method="predict"
)


# Tahminlerin Sınıf Adlarına Dönüştürülmesi

y_pred_class = label_encoder.inverse_transform(
    y_pred_encoded
)

# Genel Değerlendirme Ölçütlerinin Hesaplanması

accuracy = accuracy_score(
    y_encoded,
    y_pred_encoded
)

macro_precision = precision_score(
    y_encoded,
    y_pred_encoded,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_encoded,
    y_pred_encoded,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_encoded,
    y_pred_encoded,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    y_encoded,
    y_pred_encoded,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_encoded,
    y_pred_encoded,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_encoded,
    y_pred_encoded,
    average="weighted",
    zero_division=0
)

# Genel değerlendirme sonuçlarının gösterilmesi

print("\nGenel Değerlendirme Sonuçları:")

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")


# Genel Ölçütlerin CSV Olarak Kaydedilmesi

general_metrics_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1"
    ],

    "score": [
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_precision,
        weighted_recall,
        weighted_f1
    ]
})

general_metrics_df.to_csv(
    "xgboost_tfidf_multiclass_general_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nGenel değerlendirme sonuçları CSV olarak kaydedildi."
)

# Sınıflandırma raporu

print("\nSınıflandırma Raporu:")

report_text = classification_report(
    y_encoded,
    y_pred_encoded,
    labels=encoded_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report_text)


# Sınıflandırma Raporunun CSV Olarak Kaydedilmesi

report_dict = classification_report(
    y_encoded,
    y_pred_encoded,
    labels=encoded_labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

classification_report_df = (
    pd.DataFrame(report_dict)
    .transpose()
)

classification_report_df.to_csv(
    "xgboost_tfidf_multiclass_classification_report.csv",
    encoding="utf-8-sig"
)

print(
    "\nSınıflandırma raporu CSV olarak kaydedildi."
)

# Çapraz Doğrulama Dışı Tahminlerin Kaydedilmesi

predictions_df = pd.DataFrame({
    "Payload": X,
    "actual_class": y,
    "predicted_class": y_pred_class,
    "actual_encoded": y_encoded,
    "predicted_encoded": y_pred_encoded,
    "correct_prediction": (
        y_encoded == y_pred_encoded
    ).astype(int)
})

predictions_df.to_csv(
    "xgboost_tfidf_multiclass_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nÇapraz doğrulama dışı tahminler CSV olarak kaydedildi."
)

# Karışıklık matrisi

cm = confusion_matrix(
    y_encoded,
    y_pred_encoded,
    labels=encoded_labels
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print("\nKarışıklık Matrisi:")
print(cm_df)


# Karışıklık Matrisinin CSV Olarak Kaydedilmesi

cm_df.to_csv(
    "xgboost_tfidf_multiclass_confusion_matrix.csv",
    encoding="utf-8-sig"
)

print(
    "\nKarışıklık matrisi CSV olarak kaydedildi."
)


# Karışıklık Matrisinin Görselleştirilmesi

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("XGBoost Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

plt.tight_layout()

plt.savefig(
    "xgboost_tfidf_multiclass_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    "\nKarışıklık matrisi görsel olarak kaydedildi."
)
