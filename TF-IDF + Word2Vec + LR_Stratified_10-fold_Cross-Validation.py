import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Word2Vec Dönüştürücü Sınıfı

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=3, min_count=1, workers=4, sg=1, random_state=7):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.random_state = random_state

    # Metinlerin Tokenlara Ayrılması

    def tokenize(self, text):
        text = str(text).lower()
        return re.findall(r"[a-zA-Z0-9_]+|[^\s]", text)

    # Word2Vec Modelinin Eğitilmesi

    def fit(self, X, y=None):
        tokenized_texts = [self.tokenize(text) for text in X]

        self.w2v_model_ = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            seed=self.random_state
        )

        return self

    # Belge Vektörlerinin Oluşturulması

    def transform(self, X):
        vectors = []

        for text in X:
            tokens = self.tokenize(text)

            token_vectors = [
                self.w2v_model_.wv[token]
                for token in tokens
                if token in self.w2v_model_.wv
            ]

            if len(token_vectors) == 0:
                vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(np.mean(token_vectors, axis=0))

        return np.array(vectors)


# Veri Setinin Yüklenmesi

df = pd.read_csv("final_dataset.csv")


# X ve y'nin tanımlanması

X = df["Payload"]
y = df["attack_type"]


# TF-IDF + Word2Vec + LR Pipeline

model = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            max_features=10000,
            sublinear_tf=True
        )),
        ("word2vec", Word2VecVectorizer(
            vector_size=100,
            window=3,
            min_count=1,
            workers=4,
            sg=1,
            random_state=7
        ))
    ])),
    ("lr", LogisticRegression(
        C=10.0,
        penalty="l2",
        solver="saga",
        max_iter=1000,
        random_state=7,
        n_jobs=-1
    ))
])


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


# Sonuçların Elde Edilmesi

cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# Çapraz Doğrulama Sonuçlarının Yazdırılması

print("\n Çapraz Doğrulama Sonuçları:")

for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")


# Tahminler

y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    n_jobs=-1
)


# Değerlendirme Ölçütleri Sonuçları

print("\nGenel Değerlendirme Sonuçları:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Macro Precision: {precision_score(y, y_pred, average='macro'):.4f}")
print(f"Macro Recall: {recall_score(y, y_pred, average='macro'):.4f}")
print(f"Macro F1-score: {f1_score(y, y_pred, average='macro'):.4f}")
print(f"Weighted Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
print(f"Weighted Recall: {recall_score(y, y_pred, average='weighted'):.4f}")
print(f"Weighted F1-score: {f1_score(y, y_pred, average='weighted'):.4f}")


# Sınıf Etiketleri

labels = ["benign", "cmdi", "sqli", "xss"]


# Sınıflandırma Raporu

print("\nSınıflandırma Raporu:")

print(classification_report(
    y,
    y_pred,
    labels=labels,
    digits=4
))


# Karışıklık matrisi

cm = confusion_matrix(
    y,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nKarışıklık Matrisi:")
print(cm_df)


# Karışıklık Matrisinin CSV Olarak Kaydedilmesi

cm_df.to_csv(
    "lr_tfidf_word2vec_multiclass_confusion_matrix.csv",
    encoding="utf-8-sig"
)

print("\nKarışıklık matrisi CSV olarak kaydedildi.")


# Karışıklık Matrisinin Görselleştirilmesi

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Logistic Regression TF-IDF + Word2Vec Karışıklık Matrisi"
)

plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

plt.tight_layout()

plt.savefig(
    "lr_tfidf_word2vec_multiclass_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nKarışıklık matrisi görsel olarak kaydedildi.")
