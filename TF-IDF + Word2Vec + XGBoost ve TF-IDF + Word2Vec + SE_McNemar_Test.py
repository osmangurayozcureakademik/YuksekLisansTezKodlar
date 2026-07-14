import numpy as np
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar

# Tahmin Dosyalarının Yüklenmesi

XGBOOST_FILE = (
    "xgboost_tfidf_word2vec_multiclass_predictions.csv"
)

STACKING_FILE = (
    "stacking_lsvm_xgboost_tfidf_word2vec_multiclass_predictions.csv"
)

xgb_df = pd.read_csv(XGBOOST_FILE)
stacking_df = pd.read_csv(STACKING_FILE)

# Gerekli Sütunların Kontrol Edilmesi

required_columns = {
    "actual_class",
    "predicted_class"
}

for model_name, dataframe in [
    ("XGBoost", xgb_df),
    ("Stacking Ensemble", stacking_df)
]:
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{model_name} dosyasında eksik sütunlar var: "
            f"{sorted(missing_columns)}"
        )


# Aynı Gözlemlerin Eşleştirilmesi

if (
    "row_id" in xgb_df.columns
    and "row_id" in stacking_df.columns
):

    if xgb_df["row_id"].duplicated().any():
        raise ValueError(
            "XGBoost dosyasında yinelenen row_id var."
        )

    if stacking_df["row_id"].duplicated().any():
        raise ValueError(
            "Stacking dosyasında yinelenen row_id var."
        )

    xgb_selected = xgb_df[
        [
            "row_id",
            "actual_class",
            "predicted_class"
        ]
    ].rename(
        columns={
            "actual_class": "actual_class_xgb",
            "predicted_class": "predicted_class_xgb"
        }
    )

    stacking_selected = stacking_df[
        [
            "row_id",
            "actual_class",
            "predicted_class"
        ]
    ].rename(
        columns={
            "actual_class": "actual_class_stacking",
            "predicted_class": "predicted_class_stacking"
        }
    )

    comparison_df = xgb_selected.merge(
        stacking_selected,
        on="row_id",
        how="inner",
        validate="one_to_one"
    )

    if len(comparison_df) != len(xgb_df):
        raise ValueError(
            "Dosyalardaki row_id değerleri tam olarak eşleşmiyor."
        )

    comparison_df = comparison_df.sort_values(
        "row_id"
    ).reset_index(drop=True)

else:

    if len(xgb_df) != len(stacking_df):
        raise ValueError(
            "Dosyalardaki gözlem sayıları aynı değil."
        )

    # Payload sütunu dosyalarda (varsa) satır sıraları kontrol edilir

    if (
        "Payload" in xgb_df.columns
        and "Payload" in stacking_df.columns
    ):
        if not np.array_equal(
            xgb_df["Payload"].astype(str).to_numpy(),
            stacking_df["Payload"].astype(str).to_numpy()
        ):
            raise ValueError(
                "Dosyalardaki Payload sıraları aynı değil."
            )

    comparison_df = pd.DataFrame({
        "row_id": np.arange(len(xgb_df)),

        "actual_class_xgb": (
            xgb_df["actual_class"]
            .astype(str)
            .to_numpy()
        ),

        "predicted_class_xgb": (
            xgb_df["predicted_class"]
            .astype(str)
            .to_numpy()
        ),

        "actual_class_stacking": (
            stacking_df["actual_class"]
            .astype(str)
            .to_numpy()
        ),

        "predicted_class_stacking": (
            stacking_df["predicted_class"]
            .astype(str)
            .to_numpy()
        )
    })

# Gerçek Sınıfların Aynı Olduğunun Doğrulanması

comparison_df["actual_class_xgb"] = (
    comparison_df["actual_class_xgb"]
    .astype(str)
    .str.strip()
)

comparison_df["actual_class_stacking"] = (
    comparison_df["actual_class_stacking"]
    .astype(str)
    .str.strip()
)

comparison_df["predicted_class_xgb"] = (
    comparison_df["predicted_class_xgb"]
    .astype(str)
    .str.strip()
)

comparison_df["predicted_class_stacking"] = (
    comparison_df["predicted_class_stacking"]
    .astype(str)
    .str.strip()
)

if not np.array_equal(
    comparison_df["actual_class_xgb"].to_numpy(),
    comparison_df["actual_class_stacking"].to_numpy()
):
    raise ValueError(
        "İki dosyadaki gerçek sınıf etiketleri eşleşmiyor."
    )

print(
    "Dosyalar başarıyla eşleştirildi."
)

print(
    "Karşılaştırılan toplam gözlem sayısı:",
    len(comparison_df)
)

# Her Modelin Doğru/Yanlış Tahmin Durumunun Hesaplanması

xgb_correct = (
    comparison_df["actual_class_xgb"]
    == comparison_df["predicted_class_xgb"]
).to_numpy()

stacking_correct = (
    comparison_df["actual_class_stacking"]
    == comparison_df["predicted_class_stacking"]
).to_numpy()


# Modellerin Doğruluk Değerlerinin Gösterilmesi

xgb_accuracy = xgb_correct.mean()
stacking_accuracy = stacking_correct.mean()

accuracy_difference = (
    stacking_accuracy - xgb_accuracy
)

print("\nModel Doğrulukları:")

print(
    "TF-IDF + Word2Vec + XGBoost:",
    f"{xgb_accuracy:.6f}"
)

print(
    "TF-IDF + Word2Vec + Stacking Ensemble:",
    f"{stacking_accuracy:.6f}"
)

print(
    "Stacking Ensemble doğruluk farkı:",
    f"{accuracy_difference:.6f}"
)

print(
    "Yüzde puan farkı:",
    f"{accuracy_difference * 100:.4f}"
)

# McNemar 2×2 tablosunun oluşturulması

both_correct = int(
    np.sum(
        xgb_correct
        & stacking_correct
    )
)

xgb_correct_stacking_wrong = int(
    np.sum(
        xgb_correct
        & ~stacking_correct
    )
)

xgb_wrong_stacking_correct = int(
    np.sum(
        ~xgb_correct
        & stacking_correct
    )
)

both_wrong = int(
    np.sum(
        ~xgb_correct
        & ~stacking_correct
    )
)

table = np.array([
    [
        both_correct,
        xgb_correct_stacking_wrong
    ],
    [
        xgb_wrong_stacking_correct,
        both_wrong
    ]
])

table_df = pd.DataFrame(
    table,
    index=[
        "XGBoost doğru",
        "XGBoost yanlış"
    ],
    columns=[
        "Stacking doğru",
        "Stacking yanlış"
    ]
)

print("\nMcNemar Karşılaştırma Tablosu:")
print(table_df)


# Uyuşmazlık Sayılarının Gösterilmesi

b = xgb_correct_stacking_wrong
c = xgb_wrong_stacking_correct

discordant_total = b + c

print("\nUyuşmazlık Sayıları:")

print(
    "XGBoost doğru, Stacking yanlış:",
    b
)

print(
    "XGBoost yanlış, Stacking doğru:",
    c
)

print(
    "Toplam uyuşmazlık:",
    discordant_total
)

# Exact McNemar testinin uygulanması

if discordant_total == 0:
    raise ValueError(
        "Modeller arasında hiçbir uyuşmazlık bulunmadığı için "
        "McNemar testi uygulanamaz."
    )

result = mcnemar(
    table,
    exact=True
)

# Test Sonucunun Gösterilmesi

alpha = 0.05

print("\nExact McNemar Test Sonucu:")

print(
    "Test istatistiği:",
    result.statistic
)

print(
    "p-değeri:",
    f"{result.pvalue:.12g}"
)


if result.pvalue < alpha:

    print(
        "\nSonuç: Modeller arasındaki fark istatistiksel "
        "olarak anlamlıdır (p < 0.05)."
    )

    if c > b:
        print(
            "TF-IDF + Word2Vec + Stacking Ensemble, "
            "TF-IDF + Word2Vec + XGBoost modeline göre "
            "istatistiksel olarak anlamlı biçimde daha fazla "
            "doğru tahmin üretmiştir."
        )

    elif b > c:
        print(
            "TF-IDF + Word2Vec + XGBoost, "
            "Stacking Ensemble modeline göre istatistiksel "
            "olarak anlamlı biçimde daha fazla doğru tahmin "
            "üretmiştir."
        )

else:

    print(
        "\nSonuç: Modeller arasındaki fark istatistiksel "
        "olarak anlamlı değildir (p ≥ 0.05)."
    )

# McNemar tablosunun CSV olarak kaydedilmesi

table_df.to_csv(
    "mcnemar_tfidf_word2vec_xgboost_vs_"
    "stacking_contingency_table.csv",
    encoding="utf-8-sig"
)


# Test Özetinin CSV Olarak Kaydedilmesi

mcnemar_results_df = pd.DataFrame({
    "comparison": [
        "TF-IDF + Word2Vec + XGBoost vs "
        "TF-IDF + Word2Vec + Stacking Ensemble"
    ],

    "total_observations": [
        len(comparison_df)
    ],

    "xgboost_accuracy": [
        xgb_accuracy
    ],

    "stacking_accuracy": [
        stacking_accuracy
    ],

    "accuracy_difference": [
        accuracy_difference
    ],

    "both_correct": [
        both_correct
    ],

    "xgboost_correct_stacking_wrong": [
        b
    ],

    "xgboost_wrong_stacking_correct": [
        c
    ],

    "both_wrong": [
        both_wrong
    ],

    "discordant_total": [
        discordant_total
    ],

    "test_type": [
        "Exact McNemar"
    ],

    "test_statistic": [
        result.statistic
    ],

    "p_value": [
        result.pvalue
    ],

    "alpha": [
        alpha
    ],

    "statistically_significant": [
        result.pvalue < alpha
    ]
})

mcnemar_results_df.to_csv(
    "mcnemar_tfidf_word2vec_xgboost_vs_stacking.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nMcNemar karşılaştırma tablosu ve test sonucu "
    "CSV olarak kaydedildi."
)
