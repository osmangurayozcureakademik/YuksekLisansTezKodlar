import os
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

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# Genel ayarlar

RANDOM_STATE = 7

CACHE_DIRECTORY = (
    "/content/stacking_lsvm_xgboost_tfidf_cache"
)

print(
    "Kullanılabilir mantıksal CPU sayısı:",
    os.cpu_count()
)

# Veri setinin yüklenmesi

df = pd.read_csv(
    "final_dataset.csv"
)

# Veri seti kontrolleri

required_columns = {
    "Payload",
    "attack_type"
}

missing_columns = required_columns.difference(
    df.columns
)

if missing_columns:
    raise ValueError(
        "Veri setinde eksik sütunlar bulunmaktadır: "
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

print(
    "\nToplam gözlem sayısı:",
    len(df)
)

print(
    "\nSınıf dağılımı:"
)

print(
    y.value_counts()
)

# Sınıf etiketlerinin sayısal forma dönüştürülmesi

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

class_names = list(
    label_encoder.classes_
)

encoded_labels = np.arange(
    len(class_names)
)

print(
    "\nSınıf eşleştirmeleri:"
)

for encoded_label, class_name in enumerate(
    class_names
):
    print(
        f"{encoded_label}: "
        f"{class_name}"
    )


# Pipeline önbelleği

memory = Memory(
    location=CACHE_DIRECTORY,
    verbose=0
)

# Linear SVM temel sınıflandırıcısı

linear_svm_model = LinearSVC(
    C=1.0,
    random_state=RANDOM_STATE,
    max_iter=5000
)

# XGBoost temel sınıflandırıcısı

xgboost_model = XGBClassifier(
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

# Yığınlama Temel Sınıflandırıcılarının Tanımlanması

base_models = [
    (
        "linear_svm",
        linear_svm_model
    ),

    (
        "xgboost",
        xgboost_model
    )
]

# Yığınlama Mekanizması için İç Tabakalı 10 Katlı Çapraz Doğrulama

inner_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


# SE Modelinin Tanımlanması

meta_classifier = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
)


stacking_model = StackingClassifier(
    estimators=base_models,

    final_estimator=meta_classifier,

    # İç Tabakalı 10 Katlı Çapraz Doğrulama
    cv=inner_cv,

    stack_method="auto",

    passthrough=False,

    # İç İçe Paralellik Engellenir.
    n_jobs=1
)


# 12. TF-IDF + SE Pipeline

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
            "stacking",
            stacking_model
        )
    ],

    memory=memory
)

# Nihai Değerlendirme için Dış Tabakalı 10 Katlı Çapraz Doğrulama

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Performans Ölçütlerinin Tanımlanması

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted"
}


# 15. Tabakalı 10 Katlı Çapraz Doğrulama

print(
    "\nTF-IDF + Stacking Ensemble "
    "(Linear SVM + XGBoost) değerlendirmesi başlatılıyor"
)

print(
    "İç stacking işlemi: "
    "Tabakalı 10 Katlı Çapraz Doğrulama"
)

print(
    "Nihai Performans Değerlendirmesi: "
    "Tabakalı 10 Katlı Çapraz Doğrulama"
)


cv_results = cross_validate(
    estimator=model,
    X=X,
    y=y_encoded,
    cv=outer_cv,
    scoring=scoring,

    n_jobs=-1,
    pre_dispatch="n_jobs",

    return_train_score=False,
    error_score="raise"
)

# Kat Ortalamaları ve Standart Sapmalar

print(
    "\n Tabakalı 10 Katmanlı Çapraz Doğrulama Sonuçları:"
)

cv_summary_rows = []

for metric_name in scoring:

    metric_scores = cv_results[
        f"test_{metric_name}"
    ]

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


cv_summary_df = pd.DataFrame(
    cv_summary_rows
)

cv_summary_df.to_csv(
    "stacking_lsvm_xgboost_tfidf_multiclass_cv_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nÇapraz doğrulama sonuçları CSV olarak kaydedildi."
)

# Çapraz Doğrulama Dışı Tahminlerin Elde Edilmesi

print(
    "\nÇapraz doğrulama dışı tahminler elde ediliyor"
)

y_pred_encoded = cross_val_predict(
    estimator=model,
    X=X,
    y=y_encoded,
    cv=outer_cv,

    n_jobs=-1,
    pre_dispatch="n_jobs",

    method="predict"
)


# Sayısal Tahminlerin Sınıf Adlarına Dönüştürülmesi

y_pred_class = label_encoder.inverse_transform(
    y_pred_encoded
)

# Genel Performans Ölçütlerinin Hesaplanması

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

# Genel Değerlendirme Sonuçlarının Gösterilmesi

print(
    "\nGenel Değerlendirme Sonuçları:"
)

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision: "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall: "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1-score: "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted Precision: "
    f"{weighted_precision:.4f}"
)

print(
    f"Weighted Recall: "
    f"{weighted_recall:.4f}"
)

print(
    f"Weighted F1-score: "
    f"{weighted_f1:.4f}"
)


# Genel Performans Sonuçlarının Kaydedilmesi

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
    "stacking_lsvm_xgboost_tfidf_multiclass_general_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nGenel değerlendirme sonuçları "
    "CSV olarak kaydedildi."
)

# Sınıflandırma Raporu

print(
    "\nSınıflandırma Raporu:"
)

report_text = classification_report(
    y_encoded,
    y_pred_encoded,
    labels=encoded_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(
    report_text
)


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
    "stacking_lsvm_xgboost_tfidf_multiclass_"
    "classification_report.csv",
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
    "stacking_lsvm_xgboost_tfidf_multiclass_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nÇapraz doğrulama dışı tahminler "
    "CSV olarak kaydedildi."
)

# Karışıklık Matrisi

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

print(
    "\nKarışıklık Matrisi:"
)

print(
    cm_df
)


cm_df.to_csv(
    "stacking_lsvm_xgboost_tfidf_multiclass_"
    "confusion_matrix.csv",
    encoding="utf-8-sig"
)

print(
    "\nKarışıklık matrisi CSV olarak kaydedildi."
)

# Karışıklık Matrisinin Görselleştirilmesi

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Stacking Ensemble "
    "(Linear SVM + XGBoost) Karışıklık Matrisi"
)

plt.xlabel(
    "Tahmin Edilen Sınıf"
)

plt.ylabel(
    "Gerçek Sınıf"
)

plt.tight_layout()

plt.savefig(
    "stacking_lsvm_xgboost_tfidf_multiclass_"
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
print(
    "\nKarışıklık matrisi görsel olarak kaydedildi."
)
