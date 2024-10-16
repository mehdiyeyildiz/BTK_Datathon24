import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# dosyaları okutalım.
train = pd.read_csv("train.csv")
test = pd.read_csv("test_x.csv")

check_df(train)
#train verisinde değerlendirme puanı boş olan bir satır var, onu silelim.
train.drop(train[train["Degerlendirme Puani"].isnull()].index, axis=0, inplace=True)
train.head()
test.head()
train.shape
test.shape

#verileri birleştirelim
df = pd.concat([train, test])
df.columns = df.columns.str.replace(' ', '_').str.replace('?', '', regex=False).str.replace(',', '', regex=False).str.replace("'", '', regex=False)


# Keşifçi veri analizi

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
#cat_cols, num_cols, cat_but_car  = grab_col_names(df, cat_th=15)
# şu an veri tipine göre analizler ve eksik değer doldurma yapacağız o yüzden grab fonk kullanmıyoruz.
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in ['Degerlendirme_Puani', "id"]]
str_cols = [col for col in df.columns if df[col].dtypes == "O"]

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

for col in str_cols:
    cat_summary(df, col)


#büyük küçük harf farkı sorun oluşturuyor hepsini küçük yapalım
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

#boş değerleri doldurmadan önce bu bosluk anlam ifade edebileceğinden boş değerlere sıfır dolulara bir değer veren yeni değişkneler oluşturalım.

liste = ["Universite_Adi","Lise_Adi","Baska_Kurumdan_Aldigi_Burs_Miktari", "Girisimcilikle_Ilgili_Deneyiminizi_Aciklayabilir_misiniz","Daha_Önceden_Mezun_Olunduysa_Mezun_Olunan_Üniversite",
         "Uye_Oldugunuz_Kulubun_Ismi","Spor_Dalindaki_Rolunuz_Nedir","Hangi_STKnin_Uyesisiniz","Girisimcilikle_Ilgili_Deneyiminizi_Aciklayabilir_misiniz"]

def cat_but_car_summary(df, column_name):
    # Yeni sütunun adını oluştur
    yeni_sutun_adi = 'new_' + column_name

    # Yeni sütunu oluştur: Eğer değer boş ise 0 yap, değilse 1 yap
    df[yeni_sutun_adi] = df[column_name].apply(lambda x: 0 if pd.isna(x) or x in ["0", '-'] else 1)
    return df

for col in liste:
    cat_but_car_summary(df, col)

check_df(df)

# model geliştirirken farklı boşluk doldurma yöntemleri deneyebilirsin.
#boş değerleri veri tipine göre 0 ile dolduralım. ["Degerlendirme Puani", "id"] hariç
df['Dogum_Tarihi'].isnull().sum()
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in ['Degerlendirme_Puani', "id"]]
str_cols = [col for col in df.columns if df[col].dtypes == "O"]
df[num_cols] = df[num_cols].fillna(0)
df[str_cols] = df[str_cols].fillna("0")

###########################################

#veri tiplerini uygun hale dönüştürelim.

df.loc[df["Kardes_Sayisi"]=="kardeş sayısı 1 ek bilgi aile hk. anne vefat","Kardes_Sayisi"] = 1
df["Kardes_Sayisi"] = df["Kardes_Sayisi"].astype(int)
df["new_Baska_Kurumdan_Aldigi_Burs_Miktari"] = df["new_Baska_Kurumdan_Aldigi_Burs_Miktari"].astype(int)

#verideki yazım farklarından oluşan çeşitlilikleri düzeltelim
df.loc[(df["Spor_Dalindaki_Rolunuz_Nedir"] == "kaptan / li̇der")|(df["Spor_Dalindaki_Rolunuz_Nedir"] == "kaptan"), "Spor_Dalindaki_Rolunuz_Nedir"] = "lider/kaptan"
df.loc[(df["Spor_Dalindaki_Rolunuz_Nedir"] == "bireysel"), "Spor_Dalindaki_Rolunuz_Nedir"] = "bireysel spor"
df.loc[(df["Spor_Dalindaki_Rolunuz_Nedir"] == "di̇ğer"), "Spor_Dalindaki_Rolunuz_Nedir"] = "diğer"


df.loc[(df["Lise_Turu"] == "meslek"), "Lise_Turu"] = "meslek lisesi"
df.loc[(df["Lise_Turu"] == "özel")| (df["Lise_Turu"] == "özel lisesi"), "Lise_Turu"] = "özel lise"
df["Lise_Turu"].value_counts()

df[["Baba_Sektor", "Anne_Sektor", "Spor_Dalindaki_Rolunuz_Nedir"]] = df[["Baba_Sektor", "Anne_Sektor", "Spor_Dalindaki_Rolunuz_Nedir"]].replace("-", "0")

df.loc[(df["Baba_Egitim_Durumu"] == "üniversite mezunu")| (df["Baba_Egitim_Durumu"] == "üni̇versi̇te")| (df["Baba_Egitim_Durumu"] == "üniversite"), "Baba_Egitim_Durumu"] = "üniversite"
df.loc[(df["Baba_Egitim_Durumu"] == "i̇lkokul mezunu"), "Baba_Egitim_Durumu"] = "i̇lkokul"
df.loc[(df["Baba_Egitim_Durumu"] == "ortaokul mezunu"), "Baba_Egitim_Durumu"] = "ortaokul"
df.loc[(df["Baba_Egitim_Durumu"] == "lise mezunu")| (df["Baba_Egitim_Durumu"] == "lise")| (df["Baba_Egitim_Durumu"] == "li̇se"), "Baba_Egitim_Durumu"] = "lise"
df.loc[(df["Baba_Egitim_Durumu"] == "yüksek lisans / doktara")| (df["Baba_Egitim_Durumu"] == "yüksek li̇sans")| (df["Baba_Egitim_Durumu"] == "yüksek lisans")| (df["Baba_Egitim_Durumu"] == "doktora"), "Baba_Egitim_Durumu"] = "yüksek lisans / doktora"
df.loc[(df["Baba_Egitim_Durumu"] == "eği̇ti̇m yok")| (df["Baba_Egitim_Durumu"] == "eğitim yok"), "Baba_Egitim_Durumu"] = "eğitimi yok"

df.loc[(df["Anne_Egitim_Durumu"] == "i̇lkokul mezunu"), "Anne_Egitim_Durumu"] = "i̇lkokul"
df.loc[(df["Anne_Egitim_Durumu"] == "ortaokul mezunu"), "Anne_Egitim_Durumu"] = "ortaokul"
df.loc[(df["Anne_Egitim_Durumu"] == "lise mezunu")| (df["Anne_Egitim_Durumu"] == "li̇se")| (df["Anne_Egitim_Durumu"] == "lise"), "Anne_Egitim_Durumu"] = "lise"
df.loc[(df["Anne_Egitim_Durumu"] == "üniversite mezunu")| (df["Anne_Egitim_Durumu"] == "üni̇versi̇te")| (df["Anne_Egitim_Durumu"] == "üniversite"), "Anne_Egitim_Durumu"] = "üniversite"
df.loc[(df["Anne_Egitim_Durumu"] == "yüksek lisans / doktara")| (df["Anne_Egitim_Durumu"] == "yüksek lisans / doktora")| (df["Anne_Egitim_Durumu"] == "yüksek li̇sans")| (df["Anne_Egitim_Durumu"] == "doktora")| (df["Anne_Egitim_Durumu"] == "yüksek lisans"), "Anne_Egitim_Durumu"] = "yüksek lisans / doktora"
df.loc[(df["Anne_Egitim_Durumu"] == "eği̇ti̇m yok")| (df["Anne_Egitim_Durumu"] == "eğitim yok"), "Anne_Egitim_Durumu"] = "eğitimi yok"

df.loc[(df["Baba_Sektor"] == "di̇ğer"), "Baba_Sektor"] = "diğer"
df.loc[(df["Anne_Sektor"] == "di̇ğer"), "Anne_Sektor"] = "diğer"


df["Ingilizce_Seviyeniz"].value_counts()
df.loc[df["Ingilizce_Seviyeniz"] == "0", "İng_level"] = 0
df.loc[df["Ingilizce_Seviyeniz"] == "başlangıç", "İng_level"] = 1
df.loc[df["Ingilizce_Seviyeniz"] == "orta", "İng_level"] = 2
df.loc[df["Ingilizce_Seviyeniz"] == "i̇leri", "İng_level"] = 3
df["İng_level"] = df["İng_level"].astype(int)
df.drop(["Ingilizce_Seviyeniz"], axis=1,inplace=True)

df["Universite_Not_Ortalamasi"].value_counts()
df.loc[(df["Universite_Not_Ortalamasi"] == "1.00 - 2.50")|(df["Universite_Not_Ortalamasi"] == "0 - 1.79")|(df["Universite_Not_Ortalamasi"] == "2.00 - 2.50")|(df["Universite_Not_Ortalamasi"] == "1.80 - 2.49"), "Universite_Not_Ortalamasi"] = "2.50 ve altı"
df.loc[(df["Universite_Not_Ortalamasi"] == "3.50-3")|(df["Universite_Not_Ortalamasi"] == "3.00 - 3.49"), "Universite_Not_Ortalamasi"] = "3.00 - 3.50"
df.loc[(df["Universite_Not_Ortalamasi"] == "2.50 - 2.99")|(df["Universite_Not_Ortalamasi"] == "3.00-2.50")|(df["Universite_Not_Ortalamasi"] == "2.50 -3.00"), "Universite_Not_Ortalamasi"] = "2.50 - 3.00"
df.loc[(df["Universite_Not_Ortalamasi"] == "4.0-3.5")|(df["Universite_Not_Ortalamasi"] == "4-3.5"), "Universite_Not_Ortalamasi"] = "3.50 - 4.00"
df.loc[(df["Universite_Not_Ortalamasi"] == "not ortalaması yok"), "Universite_Not_Ortalamasi"] = "ortalama bulunmuyor"
df.loc[(df["Universite_Not_Ortalamasi"] == "3.50 - 4.00"), "Universite_Not_Ortalamasi"] = "3.00 - 4.00"

df["Lise_Mezuniyet_Notu"].value_counts()
df.loc[(df["Lise_Mezuniyet_Notu"] == "0"), "Lise_Mezuniyet_Notu"] = "not ortalaması yok"

df.loc[(df["Lise_Mezuniyet_Notu"] == "54-45")|(df["Lise_Mezuniyet_Notu"] == "25 - 50")|(df["Lise_Mezuniyet_Notu"] == "25 - 49")|
       (df["Lise_Mezuniyet_Notu"] == "44-0")|(df["Lise_Mezuniyet_Notu"] == "0 - 25")|(df["Lise_Mezuniyet_Notu"] == "0 - 24")|
       (df["Lise_Mezuniyet_Notu"] == "2.50 ve altı"), "Lise_Mezuniyet_Notu"] = "0 - 55"

df.loc[(df["Lise_Mezuniyet_Notu"] == "50 - 75")|(df["Lise_Mezuniyet_Notu"] == "50 - 74")|(df["Lise_Mezuniyet_Notu"] == "69-55")|
       (df["Lise_Mezuniyet_Notu"] == "3.00-2.50"), "Lise_Mezuniyet_Notu"] = "55 - 75"

df.loc[(df["Lise_Mezuniyet_Notu"] == "75 - 100")|(df["Lise_Mezuniyet_Notu"] == "84-70")|(df["Lise_Mezuniyet_Notu"] == "100-85")|
       (df["Lise_Mezuniyet_Notu"] == "4.00-3.50")|(df["Lise_Mezuniyet_Notu"] == "3.00 - 4.00")|(df["Lise_Mezuniyet_Notu"] == "3.50-3.00")|
       (df["Lise_Mezuniyet_Notu"] == "3.50-3"), "Lise_Mezuniyet_Notu"] = "75 - 100"


# kardeş sayısı ve ikameygah şehri değişkenlerindeki rare ifadeleri gruplayalım.
df.loc[(df["Ikametgah_Sehri"] == "------"), "Ikametgah_Sehri"] = "0"
df["Kardes_Sayisi"] = df["Kardes_Sayisi"].astype("O")

rare_list = ["Ikametgah_Sehri", "Kardes_Sayisi"]
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "Degerlendirme_Puani", rare_list)
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df[["Kardes_Sayisi","Degerlendirme_Puani"]], 0.01)
#rare_analyser(new_df, "Degerlendirme_Puani", ["Kardes_Sayisi"])
df["Kardes_Sayisi"] = new_df["Kardes_Sayisi"]

new_df = rare_encoder(df[["Ikametgah_Sehri","Degerlendirme_Puani"]], 0.005)
rare_analyser(new_df, "Degerlendirme_Puani", ["Ikametgah_Sehri"])
df["Ikametgah_Sehri"] = new_df["Ikametgah_Sehri"]

df.head()



#tip değişiklikleri oldu grupları yeniden oluşturalım
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in ['Degerlendirme_Puani', "id"]]
str_cols = [col for col in df.columns if df[col].dtypes == "O"]


#dönüşümlerden sonra cat_summary tekrar incele
for col in str_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col)



#hedef değişken analizi
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=11)
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, 'Degerlendirme_Puani', col)



# şu an bu fonksiyon için kullanılabilecek nümerik değişken yok
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, 'Degerlendirme_Puani', col)



df.head()

check_df(df)

#encoding ve standartlaştırma işlemi yapalım
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=11)
cat_cols = cat_cols + ["Ikametgah_Sehri"]

new_columns = num_cols + cat_cols
dff = df[new_columns]

num_cols = [col for col in num_cols if col not in ['Degerlendirme_Puani', 'id']]
num_cols = num_cols + ["İng_level"]

#ss = StandardScaler()
#dff[num_cols] = ss.fit_transform(dff[num_cols])

#minmaxscaler deneyelim
mms = MinMaxScaler()
dff[num_cols] = mms.fit_transform(dff[num_cols])

cat_cols = [col for col in cat_cols if col not in ["İng_level"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe
dff = one_hot_encoder(dff, cat_cols)
dff.head()
dff.shape
#(76173, 128),142

#model kurma aşaması
#voting_classifier kullan
train1 = dff[dff["Degerlendirme_Puani"].notnull()]
test1 = dff[dff["Degerlendirme_Puani"].isnull()]

X = train1.drop(["Degerlendirme_Puani", "id"], axis=1)
y = train1["Degerlendirme_Puani"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


models = [("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

#ilk sonuçlar
###
# RMSE: 640875465.4141 (LR)
# RMSE: 7.8726 (KNN)
# RMSE: 8.3281 (CART)
# RMSE: 6.098 (RF)
# RMSE: 6.332 (GBM)
# RMSE: 5.7824 (XGBoost)
# RMSE: 5.7525 (LightGBM)
# RMSE: 5.6439 (CatBoost) ###


#en iyi skorları veren 3 model ile hiperparametre optimizasyonu yapalım.

###########################################################
#XGBOOST
###########################################################


#optimizasyon yapmadan tahminleme
xgb_model = XGBRegressor().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#5.71

#XGBoost hiperparametre op.
xgb_model = XGBRegressor()
xgb_model.get_params
rmse = np.mean(np.sqrt(-cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
#5.78
#5.77
#5.75
#5.76
#5.92 X,y ile,5.89

xgb_params = {"learning_rate": [None],
               "max_depth": [3],
               "n_estimators": [600],
               "colsample_bytree": [1]}

xgb_gs_best = GridSearchCV(xgb_model,
                            xgb_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

xgb_gs_best.best_params_
#{'colsample_bytree': 1, 'learning_rate': None, 'max_depth': 3, 'n_estimators': 600}

final_model_xgb = xgb_model.set_params(**xgb_gs_best.best_params_).fit(X_train, y_train)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_xgb, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
# 5.694


y_pred = final_model_xgb.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#5.669

#tahminde bulunalım
Xs = test1.drop(["Degerlendirme_Puani", "id"], axis=1)
ys = test1["Degerlendirme_Puani"]
y_pred = final_model_xgb.predict(Xs)

dictionary = {"id":ys.index, "Degerlendirme Puani":y_pred}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("xgb2_Xy_submission.csv", index=False)

###########################################################
#LİGHTGBM
###########################################################

#optimizasyon yapmadan tahminleme
lgbm_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#5.69

#lightGBM hiperparametre op.
lgbm_model = LGBMRegressor()
lgbm_model.get_params()

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
#5.746
#5.75
# X,y ile 5.92
lgbm_params = {"learning_rate": [0.1, 0.05],
               "n_estimators": [100, 3500, 4500],
               "max_depth": [3]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

lgbm_gs_best.best_params_
#{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 3500}
#{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 4500}

final_model_lgbm = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X_train, y_train)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_lgbm, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
#5.68
#5.67
#5.64
y_pred = final_model_lgbm.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#5.39
#5.27

#test setinin tahminini yapalım

Xs = test1.drop(["Degerlendirme_Puani", "id"], axis=1)
ys = test1["Degerlendirme_Puani"]
y_pred = final_model_lgbm.predict(Xs)

dictionary = {"id":ys.index, "Degerlendirme Puani":y_pred}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("lgbm7_ttd20_submission.csv", index=False)

###########################################################
#CATBOOST
###########################################################
#optimizasyon yapmadan tahminleme
catboost_model = CatBoostRegressor().fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#5.58

#CatBoost hiperparametre op.

catboost_model = CatBoostRegressor()
catboost_model.get_params
rmse = np.mean(np.sqrt(-cross_val_score(catboost_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", verbose=False)))
#5.78
#5.63
#5.64
#5.58 X,y ile 5.57

catboost_params = {"iterations": [500],
                   "learning_rate": [0.1],
                   "depth": [6]}

catboost_gs_best = GridSearchCV(catboost_model,
                            catboost_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=False).fit(X_train, y_train)


catboost_gs_best.best_params_
#{'depth': 6, 'iterations': 500, 'learning_rate': 0.1}
final_model_cat = catboost_model.set_params(**catboost_gs_best.best_params_).fit(X_train, y_train)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_cat, X_train, y_train, cv=5, scoring="neg_mean_squared_error", verbose=False)))
#5.828
#5.79

y_pred = final_model_cat.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


Xs = test1.drop(["Degerlendirme_Puani", "id"], axis=1)
ys = test1["Degerlendirme_Puani"]
y_pred = final_model_cat.predict(Xs)

dictionary = {"id":ys.index, "Degerlendirme Puani":y_pred}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("cat4_tt_submission.csv", index=False)


###########################################
###########################################
xgb_model = XGBRegressor().fit(X, y)
lgbm_model = LGBMRegressor().fit(X, y)
catboost_model = CatBoostRegressor().fit(X, y)


def voting_regressor(X, y):
    print("Voting regressor...")

    voting_reg = VotingRegressor(estimators=[('XGB', xgb_model),
                                              ('LGBM', lgbm_model),
                                              ('CatBoost', catboost_model)]).fit(X, y)
    rmse = np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=5, scoring="neg_mean_squared_error", verbose=False)))
    print("rmse",  rmse)

    return voting_reg

voting_reg = voting_regressor(X, y)
#5.66



Xs = test1.drop(["Degerlendirme_Puani", "id"], axis=1)
ys = test1["Degerlendirme_Puani"]
y_pred = voting_reg.predict(Xs)

dictionary = {"id":ys.index, "Degerlendirme Puani":y_pred}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("vot4_ttd_submission.csv", index=False)


##########################################################
##########################################################
#feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X_train, y_train)

plot_importance(model, X_train)
model.feature_importances_

def des_importance(model, features):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    return feature_imp

fea_impt = des_importance(model,X_train)
drop_list = list(fea_impt[fea_impt["Value"] <= 1]["Feature"])
fea_impt.sort_values(by="Value")

# modeli yeniden

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=11)
cat_cols = cat_cols + ["Ikametgah_Sehri"]

new_columns = num_cols + cat_cols
dff = df[new_columns]

num_cols = [col for col in num_cols if col not in ['Degerlendirme_Puani', 'id']]
num_cols = num_cols + ["İng_level"]

#ss = StandardScaler()
#dff[num_cols] = ss.fit_transform(dff[num_cols])

#minmaxscaler deneyelim
mms = MinMaxScaler()
dff[num_cols] = mms.fit_transform(dff[num_cols])

cat_cols = [col for col in cat_cols if col not in ["İng_level"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe
dff = one_hot_encoder(dff, cat_cols)
dff.head()
dff.shape
dff_new_columns = [col for col in dff.columns if col not in drop_list]
dff = dff[dff_new_columns]

train1 = dff[dff["Degerlendirme_Puani"].notnull()]
test1 = dff[dff["Degerlendirme_Puani"].isnull()]

X = train1.drop(["Degerlendirme_Puani", "id"], axis=1)
y = train1["Degerlendirme_Puani"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

