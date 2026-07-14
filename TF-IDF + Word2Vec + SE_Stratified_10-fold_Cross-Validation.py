import os
import sys
import shutil
import importlib

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

from sklearn.pipeline import Pipeline, FeatureUnion
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


# 1. Genel Ayarlar

RANDOM_STATE = 7

LSVM_CACHE_DIRECTORY = (
    "/content/stacking_hybrid_lsvm_cache"
)

XGB_CACHE_DIRECTORY = (
    "/content/stacking_hybrid_xgboost_cache"
)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=5,
        random_state=7
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.random_state = random_state

    def tokenize(self, text):
        text = str(text).lower()
        return re.findall(r"[a-zA-Z0-9_]+|[^\s]", text)

    def fit(self, X, y=None):
        tokenized_texts = [
            self.tokenize(text)
            for text in X
        ]

        self.w2v_model_ = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.random_state
        )

        return self

    def transform(self, X):
        document_vectors = []

        for text in X:
            tokens = self.tokenize(text)

            token_vectors = [
                self.w2v_model_.wv[token]
                for token in tokens
                if token in self.w2v_model_.wv
            ]

            if token_vectors:
                document_vector = np.mean(
                    token_vectors,
                    axis=0
                ).astype(np.float32)
            else:
                document_vector = np.zeros(
                    self.vector_size,
                    dtype=np.float32
                )

            document_vectors.append(document_vector)

        return np.asarray(
            document_vectors,
            dtype=np.float32
        )


# Eski Önbelleklerin Temizlenmesi

shutil.rmtree(
    LSVM_CACHE_DIRECTORY,
    ignore_errors=True
)

shutil.rmtree(
    XGB_CACHE_DIRECTORY,
    ignore_errors=True
)

lsvm_memory = Memory(
    location=LSVM_CACHE_DIRECTORY,
    verbose=0
)

xgb_memory = Memory(
    location=XGB_CACHE_DIRECTORY,
    verbose=0
)


# Veri Setinin Yüklenmesi

df = pd.read_csv(
    "final_dataset.csv"
)


# Veri Seti Kontrolleri

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


# X ve y'nin Tanımlanması

X = df["Payload"].astype(str)
y = df["attack_type"].astype(str)

print(
    "\nToplam gözlem sayısı:",
    len(df)
)

print("\nSınıf dağılımı:")
print(y.value_counts())


# Sınıf Etiketlerinin Sayısal Forma Dönüştürülmesi

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = list(
    label_encoder.classes_
)

encoded_labels = np.arange(
    len(class_names)
)

print("\nSınıf eşleştirmeleri:")

for encoded_label, class_name in enumerate(
    class_names
):
    print(
        f"{encoded_label}: "
        f"{class_name}"
    )

# LSVM için Hibrit Özellik Çıkarımı

lsvm_features = FeatureUnion(
    transformer_list=[
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
            "word2vec",
            Word2VecVectorizer(
                vector_size=200,
                window=5,
                min_count=1,
                workers=1,
                sg=1,
                epochs=5,
                random_state=RANDOM_STATE
            )
        )
    ],

    n_jobs=1
)


# LSVM Temel Model Pipeline

linear_svm_pipeline = Pipeline(
    steps=[
        (
            "features",
            lsvm_features
        ),

        (
            "linear_svm",
            LinearSVC(
                C=1.0,
                random_state=RANDOM_STATE,
                max_iter=5000
            )
        )
    ],

    memory=lsvm_memory
)


# XGBoost için Hibrit Özellik Çıkarımı

xgboost_features = FeatureUnion(
    transformer_list=[
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
            "word2vec",
            Word2VecVectorizer(
                vector_size=200,
                window=3,
                min_count=1,
                workers=1,
                sg=1,
                epochs=5,
                random_state=RANDOM_STATE
            )
        )
    ],

    n_jobs=1
)

# XGBoost Temel Model Pipeline

xgboost_pipeline = Pipeline(
    steps=[
        (
            "features",
            xgboost_features
        ),

        (
            "xgboost",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                booster="gbtree",
                tree_method="hist",

                learning_rate=0.1,
                max_depth=3,
                min_child_weight=3,
                n_estimators=200,

                n_jobs=1,
                random_state=RANDOM_STATE,
                verbosity=0
            )
        )
    ],

    memory=xgb_memory
)

# Yığınlama Temel Modellerinin Tanımlanması

base_models = [
    (
        "linear_svm",
        linear_svm_pipeline
    ),

    (
        "xgboost",
        xgboost_pipeline
    )
]

# İç Tabakalı 10 Katlı Çapraz Doğrulama

inner_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

# LR Meta-Sınıflandırıcısı

meta_classifier = LogisticRegression(
    C=10.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=5000,
    random_state=RANDOM_STATE
)

# SE Modelinin Tanımlanması

stacking_model = StackingClassifier(
    estimators=base_models,

    final_estimator=meta_classifier,

    # İç Tabakalı 10 Katlı Çapraz Doğrulama
    cv=inner_cv,

    passthrough=False,

    n_jobs=1
)

# Dış Tabakalı 10 Katlı Çapraz Doğrulama

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

# Tabakalı 10 Katlı Çapraz Doğrulama

print(
    "\nTF-IDF + Word2Vec + Stacking Ensemble "
    "(Linear SVM + XGBoost) değerlendirmesi başlatılıyor..."
)

print(
    "İç stacking işlemi: "
    "Tabakalı 10-katlı çapraz doğrulama"
)

print(
    "Nihai performans değerlendirmesi: "
    "Tabakalı 10 Katlı Çapraz Doğrulama"
)


cv_results = cross_validate(
    estimator=stacking_model,
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
    "\n Tabakalı 10 Katlı Çapraz Doğrulama Sonuçları:"
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
    "stacking_lsvm_xgboost_tfidf_word2vec_"
    "multiclass_cv_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nÇapraz doğrulama sonuçları CSV olarak kaydedildi."
)

# Çapraz Doğrulama Dışı Tahminlerin Elde Edilmesi

print(
    "\nÇapraz doğrulama dışı tahminler elde ediliyor..."
)

y_pred_encoded = cross_val_predict(
    estimator=stacking_model,
    X=X,
    y=y_encoded,
    cv=outer_cv,

    n_jobs=-1,
    pre_dispatch="n_jobs",

    method="predict"
)


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

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")


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
    "stacking_lsvm_xgboost_tfidf_word2vec_"
    "multiclass_general_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nGenel değerlendirme sonuçları CSV olarak kaydedildi."
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

print(report_text)


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
    "stacking_lsvm_xgboost_tfidf_word2vec_"
    "multiclass_classification_report.csv",
    encoding="utf-8-sig"
)

print(
    "\nSınıflandırma raporu CSV olarak kaydedildi."
)

# Çapraz Doğrulama Dışı Tahminlerin Kaydedilmesi

predictions_df = pd.DataFrame({
    "row_id": np.arange(len(df)),
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
    "stacking_lsvm_xgboost_tfidf_word2vec_"
    "multiclass_predictions.csv",
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

    index=[
        f"Actual_{class_name}"
        for class_name in class_names
    ],

    columns=[
        f"Predicted_{class_name}"
        for class_name in class_names
    ]
)

print(
    "\nKarışıklık Matrisi:"
)

print(cm_df)


cm_df.to_csv(
    "stacking_lsvm_xgboost_tfidf_word2vec_"
    "multiclass_confusion_matrix.csv",
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
    "(Linear SVM + XGBoost) "
    "TF-IDF + Word2Vec Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(
    "stacking_lsvm_xgboost_tfidf_word2vec_"
    "multiclass_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    "\nKarışıklık matrisi görsel olarak kaydedildi."
)

# Önbelleklerin Temizlenmesi

lsvm_memory.clear(
    warn=False
)

xgb_memory.clear(
    warn=False
)

print(
    "\nÖnbellekler temizlendi."
)

print(
    "TF-IDF + Word2Vec + Stacking Ensemble "
    "(Linear SVM + XGBoost) "
    "performans değerlendirmesi tamamlandı."
)
