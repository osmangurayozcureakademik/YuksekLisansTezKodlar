import pandas as pd
import numpy as np
import re

from gensim.models import Word2Vec

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Word2Vec Dönüştürücü Sınıfı
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, sg=1, random_state=7):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.random_state = random_state

    def tokenize(self, text):
        text = str(text).lower()
        return re.findall(r"[a-zA-Z0-9_]+|[^\s]", text)

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

# Tabakalı 10 Katlı Çapraz Doğrulama
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=7
)

# TF-IDF + Word2Vec Özellik Birleşimi
features = FeatureUnion([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=10000,
        sublinear_tf=True
    )),
    ("word2vec", Word2VecVectorizer(
        min_count=1,
        workers=4,
        sg=1,
        random_state=7
    ))
])

# TF-IDF + Word2Vec + LR Pipeline
model = Pipeline([
    ("features", features),
    ("lr", LogisticRegression(
        penalty="l2",
        solver="saga",
        max_iter=1000,
        random_state=7,
        n_jobs=-1
    ))
])

# LR ve Word2Vec Parametreleri için GridSearch Aralıkları
param_grid = {
    "features__word2vec__vector_size": [100, 200],
    "features__word2vec__window": [3, 5],
    "lr__C": [0.01, 0.1, 1.0, 10.0]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=3
)

# Modelin Eğitilmesi
grid_search.fit(X, y)

# Sonuçlar
print("\nEn İyi Parametreler:")
print(grid_search.best_params_)

print("\nEn İyi Çapraz Doğrulama Doğruluğu:")
print(f"{grid_search.best_score_:.4f}")

results = pd.DataFrame(grid_search.cv_results_)
results.to_csv("lr_tfidf_word2vec_gridsearch_results.csv", index=False)
