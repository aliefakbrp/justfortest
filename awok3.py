from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
# from ini_func import *
import pandas as pd #pandas
import numpy as np #numpy
import re #regex
import string #string population
from nltk.tokenize import word_tokenize #tokenize
from nltk.corpus import stopwords #stopword
from indoNLP.preprocessing import replace_slang #slank word
from nltk.stem.porter import PorterStemmer #stemming
from tqdm.auto import tqdm #status bar
import nltk
from sklearn.decomposition import PCA #PCA
import os
import warnings
from streamlit_option_menu import option_menu
warnings.filterwarnings("ignore")
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

import ast
import pandas as pd
from collections import Counter

import pandas as pd #pandas
import numpy as np #numpy
import re #regex
import string #string population
from nltk.tokenize import word_tokenize #tokenize
from nltk.corpus import stopwords #stopword
from nltk.stem.porter import PorterStemmer #stemming
from tqdm.auto import tqdm #status bar
import nltk
from sklearn.decomposition import PCA #PCA
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import time
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

# data =pd.read_csv('/content/drive/MyDrive/DRAF/DATA/preprocessing/preprofix.csv', delimiter=";")
## Membaca file Excel dari Google Drive dan menyimpannya ke dalam DataFrame
data = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/preprofix.csv", delimiter=";")


# TF IDF
# Fungsi untuk mengubah string list ke list of words
def convert_text_list(texts):
    try:
        texts = ast.literal_eval(texts)
        return texts
    except (SyntaxError, ValueError):
        return []

# Menerapkan fungsi convert_text_list pada kolom 'stemming'
data["stemming_list"] = data["stemming"].apply(convert_text_list)


# Random Over Sampling
# split data
y = data['Sentimen']
print(pd.Series(y).value_counts())
ay = y.value_counts().plot.pie(autopct = '%.2f')
grafik = ay.set_title("BEFORE SMOTE")


# Melakukan random oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(data[["komentar",	"clean",	"tokenize",	"normalize",	"stopword",	"stemming",	"stemming_list"]], y)
# Menampilkan distribusi kelas setelah oversampling
print(pd.Series(y_resampled).value_counts())
ax = y_resampled.value_counts().plot.pie(autopct='%.2f')
grafik = ax.set_title("Oversampling Result")


# # Memastikan 'stemming_list' adalah list of words
# print("Data Stemming List:")
# print(data[["stemming_list"]])

# Fungsi untuk menghitung term frequency (TF) dari list of words
def compute_tf(text_list):
    term_count = Counter(text_list)
    total_terms = len(text_list)
    tf_dict = {term: count / total_terms for term, count in term_count.items()}
    return tf_dict

# fungsi untuk menghitung Term Frequency (TF) dari suatu dokumen
def calculate_tf(term_frequency):
    # menghitung total kemunculan semua term dalam dokumen
    total_terms = sum(term_frequency.values())

    # menghitung nilai TF untuk setiap term dalam dokumen
    tf = {term: count / total_terms for term, count in term_frequency.items()}

    # mengembalikan dictionary yang berisi nilai TF untuk setiap term dalam dokumen.
    return tf

# fungsi untuk menghitung Inverse Document Frequency (IDF) dari kumpulan dokumen
def calculate_idf(documents):
    # jumlah total dokumen dalam kumpulan
    total_documents = len(documents)

    # inisialisasi dictionary untuk menyimpan nilai IDF untuk setiap term
    idf = {}

    # iterasi melalui setiap dokumen dalam kumpulan
    for document in documents:
        # menggunakan set untuk memastikan hanya satu perhitungan IDF untuk setiap term dalam satu dokumen
        unique_terms = set(document)

        # iterasi melalui setiap term dalam dokumen
        for term in unique_terms:
            # jika term belum ada dalam dictionary IDF, hitung dan simpan nilai IDF-nya
            if term not in idf:
                # menghitung jumlah dokumen yang mengandung term
                document_count = sum(1 for doc in documents if term in doc)
                # menggunakan rumus IDF dengan penambahan 1 untuk menghindari pembagian oleh 0
                idf[term] = math.log(total_documents / (1 + document_count))

    # mengembalikan dictionary yang berisi nilai IDF untuk setiap term dalam dokumen.
    return idf

# mengimpor modul Counter dari pustaka collections
# counter digunakan untuk menghitung frekuensi kemunculan term dalam suatu iterable (objek yang dapat diulang, for,loop), seperti list.
# counter akan digunakan untuk menghitung frekuensi kemunculan setiap term dalam dokumen, sehingga dapat dihitung nilai Term Frequency (TF).

preprocessing_data = X_resampled['stemming_list']

# membuat dictionary document_tf yang menyimpan Term Frequency (TF) untuk setiap dokumen dalam preprocessing_data
# dengan menggunakan fungsi calculate_tf pada setiap Counter(tokens) untuk setiap dokumen
# fungsi enumerate digunakan untuk mengambil term dari sebuah iterable bersama dengan indeks atau nomor urutnya.
document_tf = {i: calculate_tf(Counter(tokens)) for i, tokens in enumerate(preprocessing_data)}

# membuat list documents yang berisi setiap kunci (term) dari dictionary Term Frequency (TF)
# untuk setiap dokumen dalam document_tf.values()
documents = [doc_tf.keys() for doc_tf in document_tf.values()]

# menghitung Inverse Document Frequency (IDF) menggunakan fungsi calculate_idf
# dengan menggunakan list documents sebagai parameter
idf = calculate_idf(documents)
# membuat dictionary document_tfidf yang menyimpan Term Frequency-Inverse Document Frequency (TF-IDF) untuk setiap dokumen
# dengan mengalikan Term Frequency (TF) dari setiap term dalam dokumen dengan Inverse Document Frequency (IDF) dari term tersebut
document_tfidf = {}

# iterasi melalui setiap item dalam dictionary document_tf
for doc_id, doc_tf in document_tf.items():
    # menghitung TF-IDF untuk setiap term dalam dokumen dan menyimpannya dalam doc_tfidf
    doc_tfidf = {term: tf * idf[term] for term, tf in doc_tf.items()}

    # menyimpan doc_tfidf dalam dictionary document_tfidf dengan dokumen_id sebagai kunci
    document_tfidf[doc_id] = doc_tfidf

# Membuat DataFrame tf_idf dari dictionary document_tfidf
tf_idf = pd.DataFrame(document_tfidf)

# Mengubah orientasi DataFrame menjadi transpose (T) untuk mendapatkan dokumen sebagai baris dan term sebagai kolom
# Dengan mengubah orientasi, mendapatkan representasi DataFrame yang lebih umum digunakan dalam pemrosesan teks dan analisis data.
tf_idf = tf_idf.T
# Menggantikan nilai-nilai yang kosong (NaN) dengan 0 dalam DataFrame tfidf
tfidf = tf_idf.fillna(0)


# Split data
# Membagi Data Menjadi Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(tfidf, y_resampled, test_size=0.2, random_state=42)

# Modelling
# Model training
time_start = time.perf_counter()
modelc50 = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=42)

# Evaluasi dengan K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Menyimpan nilai akurasi, presisi, recall, dan f1-score untuk setiap fold
model_accuracy_scores = []
model_precision_scores = []
model_recall_scores = []
model_f1_scores = []
temp_for_joblib=1
print('Evaluasi K-Fold Cross Validation:')
for i, (train_index, val_index) in enumerate(cv.split(X_train), 1):
    # print(train_index)
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Melatih model pada data latih fold
    # modelc50=joblib.load(f'/content/drive/MyDrive/DRAF/DATA/MODEL/model_C50{temp_for_joblib}.joblib')
    modelc50=joblib.load(f'https://github.com/08-Ahlaqul-Karimah/Pariwisata-New/blob/main/mean_C50_{temp_for_joblib}.joblib')
    # modelc50.fit(X_train_fold, y_train_fold)
    # modelc50.feature_names=X_train_fold.columns


    # Prediksi pada data validasi fold
    y_val_pred = modelc50.predict(X_val_fold)

    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_val_fold, y_val_pred)
    precision = precision_score(y_val_fold, y_val_pred, average='macro')
    recall = recall_score(y_val_fold, y_val_pred, average='macro')
    f1 = f1_score(y_val_fold, y_val_pred, average='macro')
    # modelc50.accuracy = accuracy
    # modelc50.precision = precision
    # modelc50.recall = recall
    # modelc50.f1 = f1

    # joblib.dump(modelc50,f'/content/drive/MyDrive/DRAF/DATA/MODEL/model_C50{temp_for_joblib}.joblib')

    model_accuracy_scores.append(accuracy)
    model_precision_scores.append(precision)
    model_recall_scores.append(recall)
    model_f1_scores.append(f1)

    print(f"K-Fold {i}:")
    print("  Accuracy:  {:.2f}%".format(accuracy * 100))
    print("  Precision: {:.2f}%".format(precision * 100))
    print("  Recall:    {:.2f}%".format(recall * 100))
    print("  F1-Score:  {:.2f}%".format(f1 * 100))
    temp_for_joblib+=1

print("\nRata-rata Evaluasi dari semua fold:")
print(f"Rata-rata Accuracy:  {np.mean(model_accuracy_scores)*100:.2f}%")
print(f"Rata-rata Precision: {np.mean(model_precision_scores)*100:.2f}%")
print(f"Rata-rata Recall:    {np.mean(model_recall_scores)*100:.2f}%")
print(f"Rata-rata F1-Score:  {np.mean(model_f1_scores)*100:.2f}%")
model_accuracy_scores.append(np.mean(model_accuracy_scores))
model_precision_scores.append(np.mean(model_precision_scores))
model_recall_scores.append(np.mean(model_recall_scores))
model_f1_scores.append(np.mean(model_f1_scores))


nilai_maksimum = max(model_accuracy_scores[:-1])
indeks_maksimum = model_accuracy_scores.index(nilai_maksimum)+1
modelc50=joblib.load(f'https://github.com/08-Ahlaqul-Karimah/Pariwisata-New/blob/main/mean_C50_{indeks_maksimum}.joblib')

# Evaluasi model pada data testing
print("\nEvaluasi pada data Testing:")
model_y_test_pred = modelc50.predict(X_test)
accuracy1 = accuracy_score(y_test, model_y_test_pred)
model_accuracy_scores.append(accuracy1)
print("Accuracy:  {:.2f}%".format(accuracy1 * 100))
precision1 = precision_score(y_test, model_y_test_pred, average='macro')
model_precision_scores.append(precision1)
print("Precision: {:.2f}%".format(precision1 * 100))
recall1 = recall_score(y_test, model_y_test_pred, average='macro')
model_recall_scores.append(recall1)
print("Recall:    {:.2f}%".format(recall1 * 100))
f1score = f1_score(y_test, model_y_test_pred, average='macro')
model_f1_scores.append(f1score)
print("F1-Score:  {:.2f}%".format(f1score * 100))
print("Classification Report:")
print(classification_report(y_test, model_y_test_pred))

# Membuat confusion matrix
confusion = confusion_matrix(y_test, model_y_test_pred)
class_label = ["Positif", "Negatif", "Netral"]  # Sesuaikan ini dengan label dataset Anda
confusion_df = pd.DataFrame(confusion, index=class_label, columns=class_label)
sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix C5.0")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("Confusion Matrix:")
print(confusion)

# Komputasi
modelling_time_elapsed = (time.perf_counter() - time_start)
modelling_memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ("waktu komputasi modelling = %5.1f secs %5.1f MByte" % (modelling_time_elapsed,modelling_memMb))































data_prepro = pd.read_csv("https://raw.githubusercontent.com/aliefakbrp/dtset/main/paling_baru_Preprocessing_setelah_dibuang_yang_kosong.csv", delimiter=";").drop(columns=["Unnamed: 0","index"])
data = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Total%20Data.csv", delimiter=";",on_bad_lines="skip")
tfidf=pd.read_csv("D:\\Puny\\adit\\dataset\\The_new_tf_idf_22_06_2021_18,10.csv", delimiter=";").drop(columns=["Unnamed: 0"])
BTdaramista=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Bukit%20Tinggi%20Daramista.csv", delimiter=";",encoding= 'unicode_escape')


with st.sidebar:
      selected="Home"
      selected=option_menu(
                  menu_title='Navigation',
                  options=['Home','Data','Pengujian','Report','Word Cloud','Prediksi Baru'],
                  icons=['person-circle','person-add','trophy-fill','chat-fill','info-circle-fill','info-circle-fill'],
                  menu_icon='list',
                  default_index=1,
                  styles={
                        "container": {"padding": "5!important","background-color":'black'},
            "icon": {"color": "white", "font-size": "23px"}, 
            "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
            "nav-link-selected": {"background-color": "#02ab21"},}
                  
                  )
      
      # Dataset,TF_IDF, Prediksi = st.tabs(["Dataset", "TF-IDF","Prediksi"])
if selected=="HOME":
      st.write("kontol")
elif selected=="Data":
      # with Dataset:
      st.header("Dataset")
      Dataset_asli, Setelah_prepro = st.tabs(["Dataset","Preprocessing"])
      with Dataset_asli:
            st.subheader("Dataset")
            opsi_wisata=["Semua wisata","Gili Labak","Pantai 9","Museum Keraton Sumenep","Boekit Tinggi Daramista","Goa Soekarno","Toron Samalem","Puncak Ratu","Air Terjun Toroan","Bukit Jaddih","Pantai Lombang"]
            wisata_selection=st.selectbox("Pilih wisata :",options=opsi_wisata)
            st.write("Data Wisata :",wisata_selection)
            if wisata_selection=="Semua wisata":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Total%20Data.csv", delimiter=";",on_bad_lines="skip")
            elif wisata_selection=="Gili Labak":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Gili_labak.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Pantai 9":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Pantai_Sembilan.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Museum Keraton Sumenep":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/museumkeraton.csv", delimiter=";")
            elif wisata_selection=="Boekit Tinggi Daramista":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Bukit%20Tinggi%20Daramista.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Goa Soekarno":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/goa%20soekarno.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Toron Samalem":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/toron_simalem.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Puncak Ratu":
                  data_yang_ditampilkanpd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Puncak%20ratu.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Air Terjun Toroan":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/air_terjun_teroan.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Bukit Jaddih":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/bukit_jaddih.csv", delimiter=";", encoding='windows-1252')
            elif wisata_selection=="Pantai Lombang":
                  data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/pantai_lombang2.csv", delimiter=";", encoding='windows-1252')
            # BTdaramista=pd.read_csv("https://github.com/08-Ahlaqul-Karimah/Pariwisata-New/blob/main/Puncak%20ratu.csv", delimiter=";",encoding= 'unicode_escape')
            st.write(data_yang_ditampilkan)

            # import streamlit as st

            # option = st.selectbox(
            # "How would you like to be contacted?",
            # ("Bukit_1", "Bukit_2", "Bukit_3"),
            # index=None,
            # placeholder="Select contact method...",
            # )
      with Setelah_prepro:
            st.write("Preprocessing")
            st.write(data_prepro)
elif selected=="Pengujian":
      # with TF_IDF:
      st.write("TF-IDF")
      st.write(tfidf)



elif selected=="Report":
      st.header("Report")
      
elif selected=="Word Cloud":
      # with TF_IDF:
      st.subheader("Word Cloud")
      # st.subheader("Dataset")
      opsi_wisata=["Semua wisata","Gili Labak","Pantai 9","Museum Keraton Sumenep","Boekit Tinggi Daramista","Goa Soekarno","Toron Samalem","Puncak Ratu","Air Terjun Toroan","Bukit Jaddih","Pantai Lombang"]
      wisata_selection=st.selectbox("Pilih wisata :",options=opsi_wisata)
      st.write("Data Wisata :",wisata_selection)
      if wisata_selection=="Semua wisata":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Total%20Data.csv", delimiter=";",on_bad_lines="skip")
      elif wisata_selection=="Gili Labak":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Gili_labak.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Pantai 9":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Pantai_Sembilan.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Museum Keraton Sumenep":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/museumkeraton.csv", delimiter=";")
      elif wisata_selection=="Boekit Tinggi Daramista":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Bukit%20Tinggi%20Daramista.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Goa Soekarno":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/goa%20soekarno.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Toron Samalem":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/toron_simalem.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Puncak Ratu":
            data_yang_ditampilkanpd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/Puncak%20ratu.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Air Terjun Toroan":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/air_terjun_teroan.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Bukit Jaddih":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/bukit_jaddih.csv", delimiter=";", encoding='windows-1252')
      elif wisata_selection=="Pantai Lombang":
            data_yang_ditampilkan=pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/Pariwisata-New/main/pantai_lombang2.csv", delimiter=";", encoding='windows-1252')
      # BTdaramista=pd.read_csv("https://github.com/08-Ahlaqul-Karimah/Pariwisata-New/blob/main/Puncak%20ratu.csv", delimiter=";",encoding= 'unicode_escape')
      Positif, Negatif, Netral= st.tabs(["Positif","Negatif","Netral"])
      warnings.filterwarnings("ignore")
      st.set_option('deprecation.showPyplotGlobalUse', False)
      with Positif:
            datawc=data_yang_ditampilkan.loc[data_yang_ditampilkan["Sentimen"] == "Positif"]
            positifWords=""
            for i in datawc['komentar']:
                  positifWords=positifWords +" "+i
            # Membuat WordCloud dengan menggunakan string yang telah digabungkan
            positifwordCloud = WordCloud(colormap="Blues", width=1600, height=800, background_color="lightcoral",  random_state=30, max_font_size=200, min_font_size=20).generate(positifWords)
            # Menampilkan WordCloud
            plt.imshow(positifwordCloud, interpolation='bilinear')
            plt.axis('off')
            plotnya=plt.show()
                  # st.write(data_yang_ditampilkan)
            st.pyplot(plotnya)
      with Negatif:
            datawc=data_yang_ditampilkan.loc[data_yang_ditampilkan["Sentimen"] == "Negatif"]
            NegatifWords=""
            for i in datawc['komentar']:
                  NegatifWords=NegatifWords +" "+i
            # Membuat WordCloud dengan menggunakan string yang telah digabungkan
            NegatifwordCloud = WordCloud(colormap="Blues", width=1600, height=800, background_color="lightcoral",  random_state=30, max_font_size=200, min_font_size=20).generate(NegatifWords)
            # Menampilkan WordCloud
            plt.imshow(NegatifwordCloud, interpolation='bilinear')
            plt.axis('off')
            plotnya=plt.show()
                  # st.write(data_yang_ditampilkan)
            st.pyplot(plotnya)
      with Netral:
            datawc=data_yang_ditampilkan.loc[data_yang_ditampilkan["Sentimen"] == "Netral"]
            NetralWords=""
            for i in datawc['komentar']:
                  NetralWords=NetralWords +" "+i
            # Membuat WordCloud dengan menggunakan string yang telah digabungkan
            NetralwordCloud = WordCloud(colormap="Blues", width=1600, height=800, background_color="lightcoral",  random_state=30, max_font_size=200, min_font_size=20).generate(NetralWords)
            # Menampilkan WordCloud
            plt.imshow(NetralwordCloud, interpolation='bilinear')
            plt.axis('off')
            plotnya=plt.show()
                  # st.write(data_yang_ditampilkan)
            st.pyplot(plotnya)

elif selected=="Prediksi Baru":
      # with Prediksi:
      class Prepocessing:
            def __init__(self):
                  self.listStopword =  set(stopwords.words('indonesian'))
                  self.stemmer = PorterStemmer()

            def remove_emoji(self, string): #remove emoji
                  emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
                  return emoji_pattern.sub(r' ', string)

            def remove_unwanted(self, document): #clean text
                  # remove user mentions
                  document = re.sub("@[A-Za-z0-9_]+"," ", document)
                  # menggantdocumentkan karakter newldocumentne ("\n") dengan spasdocument
                  document = re.sub("\n", " ", document)
                  # menghdocumentlangkan URL yang ddocumentmuladocument dengan 'http' atau 'https' dan 'www'
                  document = re.sub(r'http\S+|www\.\s+', ' ', document)
                  # menghdocumentlangkan karakter non-ASCdocumentdocument dardocument teks
                  document = re.sub(r'[^\x00-\x7F]+', '', document)
                  # remove hashtags
                  document = re.sub("#[A-Za-z0-9_]+","", document)
                  # remove emojdocument's
                  document = self.remove_emoji(document)
                  # remove punctuatdocumenton
                  document = re.sub("[^0-9A-Za-z ]", " " , document)
                  # remove double spaces
                  document = document.replace('  '," ")
                  # menghdocumentlangkan ddocumentgdocumentt/angka dardocument teks
                  document = re.sub("\d+", "", document)
                  # memecah teks menjaddocument daftar kata (ldocumentst of words)
                  document = document.split()
                  # menggabungkan kembaldocument daftar kata menjaddocument teks dengan spasdocument sebagadocument pemdocumentsah
                  document = " ".join(document)
                  #hapus tab, new ldocumentne
                  document = document.replace("\\t", " ").replace ("\\n", " ").replace ("\\u", " ").replace ("\\", "")
                  #hapus non ASCdocumentdocument (emote, bahasa cdocumentna dll)
                  document = str(document.encode("ascii", "replace").decode("ascii"))
                  # documents.append(i)
                  return document
            
            def tokenize(self, text): #tokenize -> memisah kalimat
                  return word_tokenize(str(text.translate(str.maketrans('', '', string.punctuation)).lower()))

            def normalize(self, text): #slank word -> mengganti kata yang tidak baku
                  return [replace_slang(kata) for kata in text]
      st.write("Prediksi")
      text_input = st.text_input("Masukkan kalimat", "Masukkan kalimat disini")
      st.write("Kalimatnya", text_input)
      df2 = pd.DataFrame({"lokasi": "","Sentimen": "","komentar": text_input},index=["test"])

      preprocessing = Prepocessing()
      df2['clean'] = df2['komentar'].apply([(lambda x: preprocessing.remove_unwanted(x))])
      df2['tokenize'] = df2['clean'].apply(lambda x: preprocessing.tokenize(x))
      df2['normalize'] = df2['tokenize'].apply(lambda x: preprocessing.normalize(x))

      from nltk.corpus import stopwords
      nltk.download('stopwords')

      # ----------------------- get stopword from NLTK stopword -------------------------------
      # get stopword indonesia
      list_stopwords = stopwords.words('indonesian')


      # ---------------------------- manualy add stopword  ------------------------------------
      # append additional stopword
      list_stopwords.extend(["nya", "yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                        'kalo', 'amp', 'biar', 'bikin', 'bilang',
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah', 'banget'])

      # ----------------------- add stopword from txt file ------------------------------------
      # read txt stopword using pandas
      txt_stopword = pd.read_csv("https://raw.githubusercontent.com/masdevid/ID-Stopwords/master/id.stopwords.02.01.2016.txt", names= ["stopwords"], header = None)

      # convert stopword string to list & append additional stopword
      list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

      # ---------------------------------------------------------------------------------------

      # convert list to dictionary
      list_stopwords = set(list_stopwords)


      #remove stopword pada list token
      def stopwords_removal(words):
            for word in words:
                  print("ini word",word)
            return [word for word in words if word not in list_stopwords]

      df2['stopword'] = df2['normalize'].apply(stopwords_removal)
      # _= !pip install swifter
      # _= !pip install Sastrawi
# import Sastrawi package
      from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
      import swifter


      # create stemmer
      factory = StemmerFactory()
      stemmer = factory.create_stemmer()

      # stemmed
      def stemmed_wrapper(term):
            return stemmer.stem(term)

      term_dict = {}

      for document in df2['stopword']:
            for term in document:
                  print("term in line 16",term)
                  if term not in term_dict:
                        term_dict[term] = ' '
      # print("isi dari term_dict",term_dict)
      # print(len(term_dict))
      # print("------------------------")

      for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)
            # print(term,":" ,term_dict[term])

      # print(term_dict)
      # print("------------------------")


      # apply stemmed term to df2frame
      def get_stemmed_term(document):
            for term in document:
                  print("ini term",term)
            return [term_dict[term] for term in document]

      df2['stemming'] = df2['stopword'].swifter.apply(get_stemmed_term)
      df2['stemming_list']=df2['stemming']
      cleaning_text=""
      for i in df2['stemming_list'][0]:
            cleaning_text+=i+" "
      st.write("Cleaning text :", cleaning_text)
      df3=data_prepro
      # print(df3["stemming"])
      df3["stemming_list"] = df3["stemming"].apply(convert_text_list)
      df3=df3._append(df2, ignore_index = True)

      #TFIDF
      # mengimpor modul numpy untuk operasi numerik
      import numpy as np
      # mengimpor modul math untuk fungsi-fungsi matematis
      import math

      # fungsi untuk menghitung Term Frequency (TF) dari suatu dokumen
      def calculate_tf(term_frequency):
            # menghitung total kemunculan semua term dalam dokumen
            total_terms = sum(term_frequency.values())

            # menghitung nilai TF untuk setiap term dalam dokumen
            tf = {term: count / total_terms for term, count in term_frequency.items()}

            # mengembalikan dictionary yang berisi nilai TF untuk setiap term dalam dokumen.
            return tf

      # fungsi untuk menghitung Inverse Document Frequency (IDF) dari kumpulan dokumen
      def calculate_idf(documents):
            # jumlah total dokumen dalam kumpulan
            total_documents = len(documents)

            # inisialisasi dictionary untuk menyimpan nilai IDF untuk setiap term
            idf = {}

            # iterasi melalui setiap dokumen dalam kumpulan
            for document in documents:
                  # menggunakan set untuk memastikan hanya satu perhitungan IDF untuk setiap term dalam satu dokumen
                  unique_terms = set(document)

                  # iterasi melalui setiap term dalam dokumen
                  for term in unique_terms:
                        # jika term belum ada dalam dictionary IDF, hitung dan simpan nilai IDF-nya
                        if term not in idf:
                              # menghitung jumlah dokumen yang mengandung term
                              document_count = sum(1 for doc in documents if term in doc)
                              # menggunakan rumus IDF dengan penambahan 1 untuk menghindari pembagian oleh 0
                              idf[term] = math.log(total_documents / (1 + document_count))

            # mengembalikan dictionary yang berisi nilai IDF untuk setiap term dalam dokumen.
            return idf
      # mengimpor modul Counter dari pustaka collections
      # counter digunakan untuk menghitung frekuensi kemunculan term dalam suatu iterable (objek yang dapat diulang, for,loop), seperti list.
      # counter akan digunakan untuk menghitung frekuensi kemunculan setiap term dalam dokumen, sehingga dapat dihitung nilai Term Frequency (TF).
      from collections import Counter
      preprocessing_data = df3['stemming_list']

      # membuat dictionary document_tf yang menyimpan Term Frequency (TF) untuk setiap dokumen dalam preprocessing_data
      # dengan menggunakan fungsi calculate_tf pada setiap Counter(tokens) untuk setiap dokumen
      # fungsi enumerate digunakan untuk mengambil term dari sebuah iterable bersama dengan indeks atau nomor urutnya.
      document_tf = {i: calculate_tf(Counter(tokens)) for i, tokens in enumerate(preprocessing_data)}
      # membuat list documents yang berisi setiap kunci (term) dari dictionary Term Frequency (TF)
      # untuk setiap dokumen dalam document_tf.values()
      documents = [doc_tf.keys() for doc_tf in document_tf.values()]

      # menghitung Inverse Document Frequency (IDF) menggunakan fungsi calculate_idf
      # dengan menggunakan list documents sebagai parameter
      idf = calculate_idf(documents)
      # membuat dictionary document_tfidf yang menyimpan Term Frequency-Inverse Document Frequency (TF-IDF) untuk setiap dokumen
      # dengan mengalikan Term Frequency (TF) dari setiap term dalam dokumen dengan Inverse Document Frequency (IDF) dari term tersebut
      document_tfidf = {}

      # iterasi melalui setiap item dalam dictionary document_tf
      for doc_id, doc_tf in document_tf.items():
            # menghitung TF-IDF untuk setiap term dalam dokumen dan menyimpannya dalam doc_tfidf
            doc_tfidf = {term: tf * idf[term] for term, tf in doc_tf.items()}

            # menyimpan doc_tfidf dalam dictionary document_tfidf dengan dokumen_id sebagai kunci
            document_tfidf[doc_id] = doc_tfidf
      # Membuat DataFrame tf_idf dari dictionary document_tfidf
      tf_idf = pd.DataFrame(document_tfidf)

      # Mengubah orientasi DataFrame menjadi transpose (T) untuk mendapatkan dokumen sebagai baris dan term sebagai kolom
      # Dengan mengubah orientasi, mendapatkan representasi DataFrame yang lebih umum digunakan dalam pemrosesan teks dan analisis data.
      tf_idf = tf_idf.T
      # Menggantikan nilai-nilai yang kosong (NaN) dengan 0 dalam DataFrame tfidf
      tfidf = tf_idf.fillna(0)
      from sklearn import datasets
      from sklearn.model_selection import train_test_split
      from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
      from sklearn import tree
      from sklearn.metrics import classification_report
      from sklearn import preprocessing
      from sklearn.feature_extraction.text import TfidfVectorizer
      import pandas as pd
      import matplotlib.pyplot as plt
      from sklearn.naive_bayes import MultinomialNB
      from sklearn.naive_bayes import GaussianNB
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.pipeline import Pipeline
      from sklearn.model_selection import RandomizedSearchCV
      from jcopml.tuning import random_search_params as rsp
      from sklearn.feature_extraction.text import CountVectorizer
      from jcopml.plot import plot_confusion_matrix
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import accuracy_score, classification_report
      from sklearn.tree import DecisionTreeClassifier, plot_tree
      # Import Libraries
      from imblearn.over_sampling import RandomOverSampler
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.model_selection import KFold
      from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
      import matplotlib.pyplot as plt
      import seaborn as sns
      import pandas as pd
      import numpy as np

      # Model training
      # modelc50 = DecisionTreeClassifier(criterion="entropy", min_samples_split=3, max_depth=None, random_state=42)
      modelc50 = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=42)

      # Evaluasi dengan K-Fold Cross-Validation
      # split data
      x = tfidf[:-1]
      y = df3['Sentimen'][:-1]
      x_train_tf, x_test_tf, y_train_tf, y_test_tf = train_test_split(x, y, test_size=0.2, random_state=8)
      # print(f"training size {x_train_tf.shape}")
      # print(f"Testing size {x_test_tf.shape}")
      from imblearn.over_sampling import SMOTE
      smote = SMOTE(random_state=42)
      # Melakukan random oversampling
      ros = RandomOverSampler(random_state=42)
      X_train_resampled, y_train_resampled = ros.fit_resample(x_train_tf, y_train_tf)

      # Menampilkan distribusi kelas setelah oversampling

      ax = y_train_resampled.value_counts().plot.pie(autopct='%.2f')
      grafik = ax.set_title("Oversampling Result")
      st.write(ax)
      # X_train_resampled, y_train_resampled = smote.fit_resample(x_train_tf, y_train_tf)
      modelc50.fit(X_train_resampled, y_train_resampled)
      predictions = modelc50.predict(tfidf[-1:])
      st.write("Sentimen",predictions[0])

      # print(dataps['stopword'].head())