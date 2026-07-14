import os
import numpy as np
import pandas as pd

from joblib import Memory

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# Genel Ayarlar

DATA_PATH = "final_dataset.csv"
RANDOM_STATE = 7

# Google Colab üzerinde geçici önbellek klasörü
CACHE_DIRECTORY = "/content/xgboost_tfidf_cache"

# Kullanılabilir işlemci sayısı
CPU_COUNT = os.cpu_count() or 1

print("Kullanılabilir mantıksal CPU sayısı:", CPU_COUNT)

# Veri setinin yüklenmesi

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

print("\nToplam gözlem sayısı:", len(df))

print("\nSınıf dağılımı:")
print(y.value_counts())



# Sınıf Etiketlerinin Sayısal Forma Dönüştürülmesi

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nSınıf eşleştirmeleri:")

for encoded_label, class_name in enumerate(
    label_encoder.classes_
):
    print(f"{encoded_label}: {class_name}")


# Tabakalı 10 Katlı Çapraz Doğrulama

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

# TF-IDF dönüşümleri için önbellek
#
# TF-IDF yine her çapraz doğrulama katının yalnızca eğitim
# bölümü üzerinde öğrenilir. Önbellek yalnızca aynı dönüşümün
# parametre kombinasyonları için tekrar hesaplanmasını önler.

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

                # Histogram tabanlı hızlı ağaç oluşturma yöntemi
                tree_method="hist",

                # Paralellik GridSearchCV seviyesinde uygulanır.
                # Böylece iç içe paralellik engellenir.
                n_jobs=1,

                random_state=RANDOM_STATE,
                verbosity=0
            )
        )
    ],

    # TF-IDF dönüşümlerini önbelleğe alır.
    memory=memory
)

# XGBoost Hiperparametre Arama Uzayı
param_grid = {
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

    # Farklı kombinasyonlar ve katlar paralel çalıştırılır.
    n_jobs=-1,

    # Aynı anda gereğinden fazla işlem oluşturulmasını önler.
    pre_dispatch="n_jobs",

    refit=True,
    return_train_score=False,
    verbose=3,

    # Hata oluşursa deney sessizce devam etmez.
    error_score="raise"
)

# Modelin eğitilmesi

grid_search.fit(X, y_encoded)

# En iyi hiperparametrelerin gösterilmesi

print("\n" + "=" * 60)
print("EN İYİ XGBOOST HİPERPARAMETRELERİ")
print("=" * 60)

for parameter_name, parameter_value in (
    grid_search.best_params_.items()
):
    clean_name = parameter_name.replace("xgb__", "")

    print(f"{clean_name}: {parameter_value}")


print("\nEn İyi Ortalama Çapraz Doğrulama Doğruluğu:")
print(f"{grid_search.best_score_:.6f}")


# Bütün GridSearchCV Sonuçlarının Hazırlanması

results = pd.DataFrame(
    grid_search.cv_results_
)

selected_columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_fit_time",
    "std_fit_time",
    "param_xgb__n_estimators",
    "param_xgb__learning_rate",
    "param_xgb__max_depth",
    "param_xgb__min_child_weight"
]

results = (
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

print("\n" + "=" * 60)
print("BÜTÜN GRIDSEARCHCV SONUÇLARI")
print("=" * 60)

print(results.to_string(index=False))

# Sonuçların CSV Dosyasına Kaydedilmesi

results.to_csv(
    "xgboost_gridsearch_results.csv",
    index=False
)

print(
    "\nGridSearchCV sonuçları "
    "'xgboost_gridsearch_results.csv' "
    "dosyasına kaydedildi."
)

# Sınıf Eşleştirmelerinin Kaydedilmesi

class_mapping = pd.DataFrame({
    "encoded_label": range(len(label_encoder.classes_)),
    "class_name": label_encoder.classes_
})

class_mapping.to_csv(
    "xgboost_class_mapping.csv",
    index=False
)

print(
    "Sınıf eşleştirmeleri "
    "'xgboost_class_mapping.csv' "
    "dosyasına kaydedildi."
)

# En İyi Eğitilmiş Model

best_model = grid_search.best_estimator_

print("\nEn iyi TF-IDF + XGBoost Pipeline:")
print(best_model)
