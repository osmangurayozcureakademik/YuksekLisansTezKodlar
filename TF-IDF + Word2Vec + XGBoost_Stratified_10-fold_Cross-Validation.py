import os
import re
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin


# Genel Ayarlar

CACHE_DIRECTORY = (
    "/content/xgboost_tfidf_word2vec_evaluation_cache"
)


# Pipeline Önbelleğinin Tanımlanması

memory = Memory(
    location=CACHE_DIRECTORY,
    verbose=0
)


# Word2Vec Dönüştürücü Sınıfı

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

        return re.findall(
            r"[a-zA-Z0-9_]+|[^\s]",
            text
        )

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

            if len(token_vectors) == 0:
                document_vector = np.zeros(
                    self.vector_size,
                    dtype=np.float32
                )
            else:
                document_vector = np.mean(
                    token_vectors,
                    axis=0
                ).astype(np.float32)

            document_vectors.append(document_vector)

        return np.asarray(
            document_vectors,
            dtype=np.float32
        )


# Veri Setinin Yüklenmesi

df = pd.read_csv("final_dataset.csv")


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
    "\nKullanılabilir mantıksal CPU sayısı:",
    os.cpu_count()
)

print(
    "Toplam gözlem sayısı:",
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


# TF-IDF + Word2Vec Hibrit Özellik Çıkarımı
#
# GridSearchCV Sonucunda Belirlenen Word2Vec Değerleri:
#
# vector_size = 200
# window = 3

features = FeatureUnion(
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

                # İç İçe Paralelliğin Önlenmesi ve
                # Tekrarlanabilirliğin Artırılması
                workers=1,

                # Skip-Gram
                sg=1,
                epochs=5,
                random_state=7
            )
        )
    ],

    # Paralellik Çapraz Doğrulama Seviyesinde Uygulanır.
    n_jobs=1
)


# TF-IDF + Word2Vec + XGBoost Pipeline
#
# GridSearchCV Sonucunda Belirlenen XGBoost Değerleri:
#
# learning_rate = 0.1
# max_depth = 3
# min_child_weight = 3
# n_estimators = 200

model = Pipeline(
    steps=[
        (
            "features",
            features
        ),

        (
            "xgb",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                booster="gbtree",
                tree_method="hist",

                learning_rate=0.1,
                max_depth=3,
                min_child_weight=3,
                n_estimators=200,

                # Paralellik Çapraz Doğrulama
                # Seviyesinde Uygulanır.
                n_jobs=1,

                random_state=7,
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
    random_state=7
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


# Çapraz Doğrulama Sonuçlarının Elde Edilmesi

print(
    "\nTabakalı 10-katlı çapraz doğrulama "
    "başlatılıyor..."
)

cv_results = cross_validate(
    estimator=model,
    X=X,
    y=y_encoded,
    cv=cv,
    scoring=scoring,

    # Farklı Katlar Paralel Çalıştırılır.
    n_jobs=-1,
    pre_dispatch="n_jobs",

    return_train_score=False,
    error_score="raise"
)


# Ortalama ve Standart Sapma Sonuçları

print(
    "\n10-Fold Stratified Cross-Validation Sonuçları:"
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
    "xgboost_tfidf_word2vec_multiclass_cv_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nÇapraz doğrulama sonuçları CSV olarak kaydedildi."
)


# Çapraz Doğrulama Dışı Tahminlerin Elde Edilmesi
#
# Her Örnek, Eğitiminde Kullanılmadığı Katın Modeli
# Tarafından Tahmin Edilmektedir.

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


# Genel Değerlendirme Sonuçları

print("\nGenel Değerlendirme Sonuçları:")

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
    "xgboost_tfidf_word2vec_multiclass_general_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nGenel değerlendirme sonuçları "
    "CSV olarak kaydedildi."
)


# Sınıflandırma Raporu

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
    "xgboost_tfidf_word2vec_multiclass_"
    "classification_report.csv",
    encoding="utf-8-sig"
)

print(
    "\nSınıflandırma raporu CSV olarak kaydedildi."
)


# Çapraz Doğrulama Dışı Tahminlerin Kaydedilmesi
#
# Bu Dosya McNemar Testi için Kullanılabilir.

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
    "xgboost_tfidf_word2vec_multiclass_predictions.csv",
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

print("\nKarışıklık Matrisi:")
print(cm_df)


# Karışıklık Matrisinin CSV Olarak Kaydedilmesi

cm_df.to_csv(
    "xgboost_tfidf_word2vec_multiclass_"
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
    "XGBoost TF-IDF + Word2Vec Karışıklık Matrisi"
)

plt.xlabel(
    "Tahmin Edilen Sınıf"
)

plt.ylabel(
    "Gerçek Sınıf"
)

plt.tight_layout()

plt.savefig(
    "xgboost_tfidf_word2vec_multiclass_"
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    "\nKarışıklık matrisi görsel olarak kaydedildi."
)


# Önbelleğin Temizlenmesi

memory.clear(
    warn=False
)

print(
    "\nÖnbellek temizlendi."
)

print(
    "TF-IDF + Word2Vec + XGBoost "
    "performans değerlendirmesi tamamlandı."
)
