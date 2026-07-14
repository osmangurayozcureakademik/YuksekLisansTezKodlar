import numpy as np
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar

# Tahmin Dosyalarının Yüklenmesi

xgb_df = pd.read_csv(
    "xgboost_tfidf_multiclass_predictions.csv"
)

stacking_df = pd.read_csv(
    "stacking_lsvm_xgboost_tfidf_multiclass_predictions.csv"
)

# Dosya Yapısının Kontrol Edilmesi

required_columns = {
    "actual_class",
    "predicted_class"
}

for file_name, dataframe in [
    ("XGBoost", xgb_df),
    ("Stacking Ensemble", stacking_df)
]:
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{file_name} dosyasında eksik sütunlar var: "
            f"{sorted(missing_columns)}"
        )

# Aynı Gözlemlerin Karşılaştırıldığının Doğrulanması

if len(xgb_df) != len(stacking_df):
    raise ValueError(
        "Dosyalardaki gözlem sayıları aynı değil."
    )

if not np.array_equal(
    xgb_df["actual_class"].astype(str).to_numpy(),
    stacking_df["actual_class"].astype(str).to_numpy()
):
    raise ValueError(
        "Gerçek sınıfların sırası eşleşmiyor."
    )

# Payload İki dosyada da varsa sıralama ayrıca kontrol edilir.

if (
    "Payload" in xgb_df.columns
    and "Payload" in stacking_df.columns
):
    if not np.array_equal(
        xgb_df["Payload"].astype(str).to_numpy(),
        stacking_df["Payload"].astype(str).to_numpy()
    ):
        raise ValueError(
            "Payload sıraları eşleşmiyor."
        )

print(
    "Dosyalar aynı gözlemleri ve aynı sırayı içeriyor."
)

# Her model için doğru/yanlış durumlarının oluşturulması

xgb_correct = (
    xgb_df["actual_class"].astype(str)
    == xgb_df["predicted_class"].astype(str)
).to_numpy()

stacking_correct = (
    stacking_df["actual_class"].astype(str)
    == stacking_df["predicted_class"].astype(str)
).to_numpy()


# McNemar 2×2 tablosunun oluşturulması

both_correct = np.sum(
    xgb_correct & stacking_correct
)

xgb_correct_stacking_wrong = np.sum(
    xgb_correct & ~stacking_correct
)

xgb_wrong_stacking_correct = np.sum(
    ~xgb_correct & stacking_correct
)

both_wrong = np.sum(
    ~xgb_correct & ~stacking_correct
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

# Exact McNemar testinin uygulanması

result = mcnemar(
    table,
    exact=True
)

# Sonuçların gösterilmesi

print("\nUyuşmazlık Sayıları:")

print(
    "XGBoost doğru, Stacking yanlış:",
    xgb_correct_stacking_wrong
)

print(
    "XGBoost yanlış, Stacking doğru:",
    xgb_wrong_stacking_correct
)

print("\nMcNemar Test Sonucu:")
print(f"Test istatistiği: {result.statistic}")
print(f"p-değeri: {result.pvalue:.10f}")


alpha = 0.05

if result.pvalue < alpha:
    print(
        "\nSonuç: Modeller arasındaki performans farkı "
        "istatistiksel olarak anlamlıdır (p < 0.05)."
    )

    if (
        xgb_wrong_stacking_correct
        > xgb_correct_stacking_wrong
    ):
        print(
            "Stacking Ensemble, XGBoost'a göre "
            "istatistiksel olarak anlamlı biçimde "
            "daha fazla doğru tahmin üretmiştir."
        )
    else:
        print(
            "XGBoost, Stacking Ensemble'a göre "
            "istatistiksel olarak anlamlı biçimde "
            "daha fazla doğru tahmin üretmiştir."
        )

else:
    print(
        "\nSonuç: Modeller arasındaki performans farkı "
        "istatistiksel olarak anlamlı değildir "
        "(p ≥ 0.05)."
    )

# Sonuçların CSV olarak kaydedilmesi

mcnemar_results_df = pd.DataFrame({
    "comparison": [
        "TF-IDF + XGBoost vs TF-IDF + Stacking Ensemble"
    ],
    "both_correct": [
        both_correct
    ],
    "xgboost_correct_stacking_wrong": [
        xgb_correct_stacking_wrong
    ],
    "xgboost_wrong_stacking_correct": [
        xgb_wrong_stacking_correct
    ],
    "both_wrong": [
        both_wrong
    ],
    "test_type": [
        "Exact McNemar"
    ],
    "statistic": [
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
    "mcnemar_tfidf_xgboost_vs_stacking.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nMcNemar testi sonucu CSV olarak kaydedildi."
)
