import os
import numpy as np
import pandas as pd

from joblib import Memory, hash as joblib_hash

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import (
    Pipeline,
    FeatureUnion
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

import re
import numpy as np

from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

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

# Eksik Sütun ve Değer Kontrolü

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
    "Kullanılabilir mantıksal CPU sayısı:",
    os.cpu_count()
)

print(
    "\nToplam gözlem sayısı:",
    len(df)
)

print("\nSınıf dağılımı:")
print(y.value_counts())



# XGBoost için Sınıf Etiketlerinin Sayısal Forma Dönüştürülmesi

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nSınıf eşleştirmeleri:")

for encoded_label, class_name in enumerate(
    label_encoder.classes_
):
    print(
        f"{encoded_label}: "
        f"{class_name}"
    )

# Tabakalı 10 Katlı Çapraz Doğrulama

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=7
)

# Pipeline önbelleği

memory = Memory(
    location="/content/xgboost_tfidf_word2vec_cache",
    verbose=0
)

# TF-IDF + Word2Vec hibrit özellik çıkarımı

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
                vector_size=100,
                window=5,
                min_count=1,

                # GridSearchCV dış seviyede paralel
                # çalıştığı için iç paralellik kapatılmıştır.
                workers=1,

                # Skip-Gram
                sg=1,

                # Önceki hibrit deneyle tutarlı değer
                epochs=5,

                random_state=7
            )
        )
    ],

    # Paralellik GridSearchCV seviyesinde uygulanmaktadır.
    n_jobs=1
)


# Önbellek ve Serileştirme Ön Kontrolü

feature_hash = joblib_hash(features)

print(
    "\nÖzellik çıkarma Pipeline hash testi başarılı:"
)

print(feature_hash)

# TF-IDF + Word2Vec + XGBoost Pipeline

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

                # Histogram tabanlı hızlı ağaç oluşturma
                tree_method="hist",

                # Paralellik GridSearchCV seviyesinde
                # uygulanmaktadır.
                n_jobs=1,

                random_state=7,
                verbosity=0
            )
        )
    ],

    memory=memory
)

# XGBoost ve Word2Vec Parametreleri için GridSearch Aralıkları

param_grid = {
    "features__word2vec__vector_size": [
        100,
        200
    ],

    "features__word2vec__window": [
        3,
        5
    ],

    "xgb__n_estimators": [
        100,
        200
    ],

    "xgb__learning_rate": [
        0.05,
        0.10
    ],

    "xgb__max_depth": [
        3,
        5
    ],

    "xgb__min_child_weight": [
        1,
        3
    ]
}

# GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",

    cv=cv,

    # Kombinasyonlar ve katlar dış seviyede paralel çalıştırılır.
    n_jobs=-1,

    # Aynı anda gereğinden fazla iş oluşturulmasını önler.
    pre_dispatch="n_jobs",

    refit=True,
    return_train_score=False,
    verbose=3,

    # Bir hata meydana gelirse sessizce puan atamak yerine gerçek hatayı gösterir.
    error_score="raise"
)

# Modelin eğitilmesi

print(
    "\nTF-IDF + Word2Vec + XGBoost "
    "GridSearchCV başlatılıyor..."
)

print(
    "64 hiperparametre kombinasyonu ve toplam "
    "640 çapraz doğrulama eğitimi gerçekleştirilecektir."
)

grid_search.fit(
    X,
    y_encoded
)

# En iyi hiperparametrelerin gösterilmesi

print("\n" + "=" * 70)

print(
    "EN İYİ TF-IDF + WORD2VEC + "
    "XGBOOST HİPERPARAMETRELERİ"
)

print("=" * 70)

for parameter_name, parameter_value in (
    grid_search.best_params_.items()
):
    print(
        f"{parameter_name}: "
        f"{parameter_value}"
    )


print(
    "\nEn İyi Ortalama Çapraz Doğrulama Doğruluğu:"
)

print(
    f"{grid_search.best_score_:.6f}"
)

# GridSearchCV Sonuçlarının Hazırlanması

results = pd.DataFrame(
    grid_search.cv_results_
)

selected_columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_fit_time",
    "std_fit_time",

    "param_features__word2vec__vector_size",
    "param_features__word2vec__window",

    "param_xgb__n_estimators",
    "param_xgb__learning_rate",
    "param_xgb__max_depth",
    "param_xgb__min_child_weight"
]

results_summary = (
    results[selected_columns]
    .sort_values(
        by=[
            "rank_test_score",
            "mean_test_score"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(drop=True)
)

# Sonuçların yazdırılması

print("\n" + "=" * 70)
print("BÜTÜN GRIDSEARCHCV SONUÇLARI")
print("=" * 70)

print(
    results_summary.to_string(
        index=False
    )
)

# Özet Sonuçların CSV Dosyasına Kaydedilmesi

results_summary.to_csv(
    "xgboost_tfidf_word2vec_gridsearch_results.csv",
    index=False,
    encoding="utf-8-sig"
)

# GridSearchCV Tarafından Oluşturulan Bütün Sonuçların Kaydedilmesi

results.to_csv(
    "xgboost_tfidf_word2vec_gridsearch_full_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nGridSearchCV sonuçları CSV olarak kaydedildi."
)

# En iyi Parametrelerin Ayrı Dosyaya Kaydedilmesi

best_parameters_df = pd.DataFrame(
    list(
        grid_search.best_params_.items()
    ),
    columns=[
        "parameter",
        "selected_value"
    ]
)

best_parameters_df[
    "best_mean_cv_accuracy"
] = grid_search.best_score_

best_parameters_df.to_csv(
    "xgboost_tfidf_word2vec_best_parameters.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "En iyi hiperparametreler ayrı CSV dosyasına "
    "kaydedildi."
)

# Sınıf Eşleştirmelerinin Kaydedilmesi

class_mapping = pd.DataFrame({
    "encoded_label": range(
        len(label_encoder.classes_)
    ),

    "class_name": label_encoder.classes_
})

class_mapping.to_csv(
    "xgboost_tfidf_word2vec_class_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "Sınıf eşleştirmeleri CSV olarak kaydedildi."
)

# En İyi Eğitilmiş Pipeline

best_model = grid_search.best_estimator_

print(
    "\nEn iyi TF-IDF + Word2Vec + XGBoost Pipeline:"
)

print(best_model)
