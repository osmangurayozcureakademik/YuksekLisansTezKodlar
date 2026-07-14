import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
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

# Veri Setinin Yüklenmesi
df = pd.read_csv("final_dataset.csv")

# X ve y'nin tanımlanması
X = df["Payload"]
y = df["attack_type"]

# TF-IDF + LR Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=10000,
        sublinear_tf=True
    )),
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

print("\n Tabakalı 10 Katmanlı Çapraz Doğrulama Sonuçları: ")
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

print("\n Genel Değerlendirme Sonuçları: ")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Macro Precision: {precision_score(y, y_pred, average='macro'):.4f}")
print(f"Macro Recall: {recall_score(y, y_pred, average='macro'):.4f}")
print(f"Macro F1-score: {f1_score(y, y_pred, average='macro'):.4f}")
print(f"Weighted Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
print(f"Weighted Recall: {recall_score(y, y_pred, average='weighted'):.4f}")
print(f"Weighted F1-score: {f1_score(y, y_pred, average='weighted'):.4f}")

# Karışıklık Matrisi

labels = ["benign", "cmdi", "sqli", "xss"]

# Sınıflandırma Raporu
print("\n Sınıflandırma Raporu: ")
print(classification_report(
    y,
    y_pred,
    labels=labels,
    digits=4
))

# Karışıklık Matrisi
cm = confusion_matrix(y, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n Karışıklık Matrisi:")
print(cm_df)

cm_df.to_csv(
    "lr_tfidf_multiclass_confusion_matrix.csv",
    encoding="utf-8-sig"
)

print("\n Karışıklık matrisi CSV olarak kaydedildi")

# Karışıklık Matrisinin Görselleştirilmesi

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("LR Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")

plt.tight_layout()

plt.savefig(
    "lr_tfidf_multiclass_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n Karışıklık matrisi görsel olarak kaydedildi")
