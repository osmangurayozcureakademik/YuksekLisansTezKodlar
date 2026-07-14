# Gerekli Kütüphanelerin Yüklenmesi


import pandas as pd
import numpy as np


# Saldırı Veri Setlerinin Dosya Yollarının Tanımlanması


cmdi_kaggle_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/CMDi/command injection 2.csv"

sqli_ieee_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/SQLi/SQL_Injection_Detection_Dataset.csv"

xss_ieee_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/XSS/Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection.csv"


# Saldırı Veri Setlerinin Yüklenmesi ve Sütun Standardizasyonu


cmdi_kaggle_df = pd.read_csv(cmdi_kaggle_path)

cmdi_kaggle_df = cmdi_kaggle_df.rename(
    columns={"sentence": "Payload"}
)

cmdi_kaggle_df.head()


sqli_ieee_df = pd.read_csv(sqli_ieee_path)

sqli_ieee_df = sqli_ieee_df.drop(columns=["Unnamed: 2"])

sqli_ieee_df = sqli_ieee_df.rename(
    columns={"Query": "Payload"}
)

sqli_ieee_df.head()


xss_ieee_df = pd.read_csv(xss_ieee_path)

xss_ieee_df = xss_ieee_df.rename(
    columns={"Query": "Payload"}
)

xss_ieee_df.head()


# Saldırı Veri Setlerinin Genel İncelenmesi


print("CMDi Veri Seti Boyutu: ", cmdi_kaggle_df.shape)


print("CMDi Veri Seti Eksik Değerleri: ")

print(cmdi_kaggle_df.isnull().sum())


# CMDi Eksik Verisinin Temizlenmesi
cmdi_kaggle_df = cmdi_kaggle_df.dropna(subset=["Payload"])

print("CMDi Veri Seti Eksik Değerleri: ")

print(cmdi_kaggle_df.isnull().sum())


print("CMDi Veri Seti Sınıf Dağılımı: ")

print(cmdi_kaggle_df["Label"].value_counts())


print("CMDi Veri Seti Yinelenen Payload Sayısı: ")

print(cmdi_kaggle_df.duplicated(subset=["Payload"]).sum())


print("SQLi Veri Seti Boyutu: ", sqli_ieee_df.shape)


print("SQLi Veri Seti Eksik Değerleri: ")

print(sqli_ieee_df.isnull().sum())


print("SQLi Veri Seti Sınıf Dağılımı: ")

print(sqli_ieee_df["Label"].value_counts())


# Label Sütununun Sayısal Veri Tipine Dönüştürülmesi
sqli_ieee_df["Label"] = pd.to_numeric(
    sqli_ieee_df["Label"],
    errors="coerce"
)

# Geçersiz veya Eksik Etiket Değerlerinin Kaldırılması
sqli_ieee_df = sqli_ieee_df.dropna(subset=["Label"])

# Label Sütununun Tam Sayı Veri Tipine Dönüştürülmesi
sqli_ieee_df["Label"] = sqli_ieee_df["Label"].astype(int)

# Sadece Geçerli Sınıf Etiketlerinin (0 ve 1) Korunması
sqli_ieee_df = sqli_ieee_df[
    sqli_ieee_df["Label"].isin([0, 1])
]

# Temizlenmiş Veri Setinin Yeni Bir CSV Dosyası Olarak Kaydedilmesi
sqli_ieee_df.to_csv(
    "/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/SQLi/sqli_label_corrected.csv",
    index=False
)


# Label Sütunu Düzeltilen SQLi Veri Setinin Dosya Yolunun Tekrar Tanımlanması
sqli_ieee_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/SQLi/sqli_label_corrected.csv"

# Label Sütunu Düzeltilen SQLi Veri Setinin Tekrar Okunması
sqli_ieee_df = pd.read_csv(sqli_ieee_path)

print("Label Sütunu Düzeltilen SQLi Veri Seti Boyutu: ", sqli_ieee_df.shape)


print("Label Sütunu Düzeltilen SQLi Veri Seti Eksik Değerleri: ")

print(sqli_ieee_df.isnull().sum())


print("Label Sütunu Düzeltilen SQLi Veri Seti Sınıf Dağılımı: ")

print(sqli_ieee_df["Label"].value_counts())


print("XSS Veri Seti Boyutu: ", xss_ieee_df.shape)


print("XSS Veri Seti Eksik Değerleri: ")

print(xss_ieee_df.isnull().sum())


print("XSS Veri Seti Sınıf Dağılımı: ")

print(xss_ieee_df["Label"].value_counts())


# Yinelenen Verilerin Temizlenmesi
cmdi_kaggle_df = cmdi_kaggle_df.drop_duplicates(subset=["Payload"])
xss_ieee_df = xss_ieee_df.drop_duplicates(subset=["Payload"])
sqli_ieee_df = sqli_ieee_df.drop_duplicates(subset=["Payload"])


print("Yinelenen Veriler Temizlendikten Sonra CMDi Veri Seti Boyutu: ", cmdi_kaggle_df.shape)
print("Yinelenen Veriler Temizlendikten Sonra SQLi Veri Seti Boyutu: ", sqli_ieee_df.shape)
print("Yinelenen Veriler Temizlendikten Sonra XSS Veri Seti Boyutu: ", xss_ieee_df.shape)

print("Yinelenen Veriler Temizlendikten Sonra CMDi Veri Seti Sınıf Dağılımı: ")

print(cmdi_kaggle_df["Label"].value_counts())


print("Yinelenen Veriler Temizlendikten Sonra SQLi Veri Seti Sınıf Dağılımı: ")

print(sqli_ieee_df["Label"].value_counts())


print("Yinelenen Veriler Temizlendikten Sonra XSS Veri Seti Sınıf Dağılımı: ")

print(xss_ieee_df["Label"].value_counts())


# Temizlenmiş CMDi Veri Setinin Kaydedilmesi
cmdi_kaggle_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Temizlenmiş/CMDi/cmdi_kaggle_cleaned.csv",
    index=False
)

# Temizlenmiş SQLi Veri Setinin Kaydedilmesi
sqli_ieee_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Temizlenmiş/SQLi/sqli_ieee_cleaned.csv",
    index=False
)

# Temizlenmiş XSS Veri Setinin Kaydedilmesi
xss_ieee_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Temizlenmiş/XSS/xss_ieee_cleaned.csv",
    index=False
)


# CMDi Veri Seti Dengesizliğinin Giderilmesi İçin Ek Veri Seti Kullanımı

# IEEE DataPort Kaynaklı Command Injection Veri Setinin Ön İşlenmesi


# CMDi için Ek Veri Setinin Dosya Yolunun Tanımlanması
cmdi_ieee_path = r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/CMDi/CI v3 c4.2 pl1.csv"

# CMDi için Ek Veri Setinin Yüklenmesi
cmdi_ieee_df = pd.read_csv(cmdi_ieee_path)

# Yalnızca Payloads Sütununun Seçilmesi
cmdi_ieee_df = cmdi_ieee_df[["Payloads"]]

# Payloads Sütununda İlk Satırda Bulunan "[]" Değeri Geçersiz Olduğu için Kaldırılması
cmdi_ieee_df = cmdi_ieee_df[
    cmdi_ieee_df["Payloads"].astype(str).str.strip() != "[]"
]

# Payloads Sütunu Diğer Veri Setleri ile Uyumlu Olması için Payload Olarak Tekrar Adlandırılması
cmdi_ieee_df = cmdi_ieee_df.rename(
    columns={"Payloads": "Payload"}
)

# Payload Sütunundaki Eksik Değerlerin Kontrol Edilmesi
print(cmdi_ieee_df.isnull().sum())

# Payload Sütununda Eksik Değerler (varsa) Ortadan Kaldırılmaktadır
cmdi_ieee_df = cmdi_ieee_df.dropna(subset=["Payload"])

# Label Sütununun Ek Veri Setine Eklenmesi
# Saldırı Verisini Temsil Eden Veri Etiketinin 1 Olarak Atanması
cmdi_ieee_df["Label"] = 1

# Yinelenen Değerlerin Ortadan Kaldırılması
cmdi_ieee_df = cmdi_ieee_df.drop_duplicates(
    subset=["Payload"]
)


print("CMDi saldırısı ek veri seti boyutu: ", cmdi_ieee_df.shape)


# Sınıflar Arası Veri Dengesizliğini Gidermek için Rastgele 1103 Veri Seçilmesi
cmdi_ieee_sampled_df = cmdi_ieee_df.sample(
    n=1103,
    random_state=7
)

# Ek Veri Seti ile CMDi Veri Setinin Birleştirilmesi
cmdi_final_df = pd.concat(
    [cmdi_kaggle_df, cmdi_ieee_sampled_df],
    ignore_index=True
)

# Nihai CMDi Veri Setinin Kaydedilmesi
cmdi_final_df.to_csv(
    r"/Users/osmangurayozcure/Desktop/Akademik/Tez/Kaynaklar/Veri Setleri/Final Datasets/cmdi_final.csv",
    index=False
)

