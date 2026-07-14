import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Veri setinin yüklenmesi

df = pd.read_csv("final_dataset.csv")

# X ve y'nin tanımlanması

X = df["Payload"]
y = df["attack_type"]

# Tabakalı 10 Katlı Çapraz Doğrulama

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=7
)

# TF-IDF + KNN Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=10000,
        sublinear_tf=True
    )),
    ("knn", KNeighborsClassifier())
])

# KNN parametreleri için GridSearchCV aralığı

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["cosine", "euclidean"]
}

# GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

# Modelin eğitilmesi

grid_search.fit(X, y)

# Sonuçlar

print("\n En İyi Parametreler:")
print(grid_search.best_params_)

print("\n En İyi Çapraz Doğrulama Doğruluğu:")
print(f"{grid_search.best_score_:.4f}")
