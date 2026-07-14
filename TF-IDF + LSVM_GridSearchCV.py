import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.svm import LinearSVC
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

# TF-IDF + LSVM Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=10000,
        sublinear_tf=True
    )),
    ("svm", LinearSVC(
        random_state=7,
        max_iter=5000
    ))
])

# C parametresi için GridSearch aralığı

param_grid = {
    "svm__C": [0.01, 0.1, 1.0, 10.0]
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
