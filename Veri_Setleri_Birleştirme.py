# Veri Ön İşlemesi Sonrası Veri Setlerinin Birleştirilmesi 



# Temizlenen XSS ve SQLi Veri Setleri ile Nihai CMDi Veri Setinin Birleştirilmesi


# Gerekli Kütüphanenin Yüklenmesi


import pandas as pd


# Veri Setlerinin Dosya Yollarının Tanımlanması


# Nihai CMDi Veri Setinin Dosya Yolunun Tanımlanması
cmdi_final_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Final Datasets/cmdi_final.csv"

# XSS Veri Setinin Dosya Yolunun Tanımlanması
xss_cleaned_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Temizlenmiş/XSS/xss_ieee_cleaned.csv"

# SQLi Veri Setinin Dosya Yolunun Tanımlanması
sqli_cleaned_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Temizlenmiş/SQLi/sqli_ieee_cleaned.csv"


# Veri Setlerinin Yüklenmesi

# Nihai CMDi Veri Setinin Yüklenmesi
cmdi_final_df = pd.read_csv(cmdi_final_path)

# XSS Veri Setinin Yüklenmesi
xss_cleaned_df = pd.read_csv(xss_cleaned_path)

# SQLi Veri Setinin Yüklenmesi
sqli_cleaned_df = pd.read_csv(sqli_cleaned_path)


# XSS ve SQLi Veri Setleri İçin Veri Dengeleme (Downsampling) İşlemleri

# Hedeflenen veri sayısının tanımlanması
target_count = 1581

# XSS Veri Seti için Downsampling Uygulanması

xss_ieee_benign_df = xss_cleaned_df[
    xss_cleaned_df["Label"] == 0
].sample(
    n=target_count,
    random_state=7
)

xss_ieee_attack_df = xss_cleaned_df[
    xss_cleaned_df["Label"] == 1
].sample(
    n=target_count,
    random_state=7
)

xss_balanced_df = pd.concat(
    [xss_ieee_benign_df, xss_ieee_attack_df],
    ignore_index=True
)

# Nihai XSS Veri Setinin Kaydedilmesi
xss_balanced_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Final Datasets/xss_final.csv",
    index=False
)

# SQLi veri seti için downsampling uygulanması

sqli_ieee_benign_df = sqli_cleaned_df[
    sqli_cleaned_df["Label"] == 0
].sample(
    n=target_count,
    random_state=7
)

sqli_ieee_attack_df = sqli_cleaned_df[
    sqli_cleaned_df["Label"] == 1
].sample(
    n=target_count,
    random_state=7
)

sqli_balanced_df = pd.concat(
    [sqli_ieee_benign_df, sqli_ieee_attack_df],
    ignore_index=True
)

# XSS Veri Setinin Kaydedilmesi
sqli_balanced_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Final Datasets/sqli_final.csv",
    index=False
)


print("XSS sınıf dağılımı:")
print(xss_balanced_df["Label"].value_counts())


print("SQLi sınıf dağılımı:")
print(sqli_balanced_df["Label"].value_counts())


# Veri Setlerine Saldırı Türü (Attack Type) Sütununun Eklenmesi ve Mapping (0: benign, 1: saldırı türü) Uygulanması


cmdi_final_df["attack_type"] = cmdi_final_df["Label"].map({
    0: "benign",
    1: "cmdi"
})

xss_balanced_df["attack_type"] = xss_balanced_df["Label"].map({
    0: "benign",
    1: "xss"
})

sqli_balanced_df["attack_type"] = sqli_balanced_df["Label"].map({
    0: "benign",
    1: "sqli"
})


# Veri Setlerinin Birleştirilmesi

final_dataset_df = pd.concat(
    [
        cmdi_final_df,
        xss_balanced_df,
        sqli_balanced_df
    ],
    ignore_index=True
)


# Birleştirilme Sonrası Sınıf Dağılımlarının Kontrol Edilmesi



print(final_dataset_df["Label"].value_counts())
print(final_dataset_df["attack_type"].value_counts())


# Nihai Veri Setinin Kaydedilmesi

final_dataset_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Final Datasets/final_dataset.csv",
    index=False
)




