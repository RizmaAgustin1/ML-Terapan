# Laporan Proyek Machine Learning - Rizma Agustin

## Domain Proyek

Klasifikasi varietas biji kering (dry beans) merupakan aspek penting dalam industri pertanian dan pangan. Identifikasi yang akurat terhadap jenis biji kering tidak hanya membantu dalam menjaga kualitas produk, tetapi juga dalam proses seleksi benih, pengolahan, dan distribusi. Tradisionalnya, klasifikasi ini dilakukan secara manual oleh para ahli, yang memerlukan waktu dan rentan terhadap kesalahan manusia.

Dengan kemajuan teknologi, khususnya dalam bidang computer vision dan machine learning, memungkinkan otomatisasi proses ini dengan akurasi tinggi. Dataset Dry Bean dari UCI Machine Learning Repository menyediakan data morfologis dari 13.611 biji kering yang berasal dari tujuh varietas berbeda, yang diambil menggunakan kamera resolusi tinggi dan diekstraksi menjadi 16 fitur numerik.

**Rubrik/Kriteria Tambahan (Opsional)**:
### Pentingnya Masalah
- Automatisasi klasifikasi biji kering dapat meningkatkan efisiensi dalam rantai pasok pertanian, mengurangi biaya operasional, dan meningkatkan konsistensi kualitas produk. Selain itu, pendekatan ini dapat diterapkan pada komoditas pertanian lainnya, memperluas dampak positifnya dalam sektor pertanian secara keseluruhan.

### Hasil Riset atau Referensi
1. Koklu, M., & Ozkan, I. A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture, 175, 105552.

Menggunakan fitur citra dan ML klasik (k-NN, RF, SVM) pada dataset Dry Bean; mendokumentasikan preprocessing citra dan perbandingan model

2. Lee, C.-Y., Wang, W., & Huang, J.-Q. (2023). Clustering and Classification for Dry Bean Feature Imbalanced Data.

Menerapkan K-means untuk clustering awal lalu diikuti Decision Tree, Random Forest, dan SVM untuk klasifikasi; K-means+SVM memberikan akurasi terbaik, dan Compactness menjadi fitur paling penting.

3. Słowiński, G. (2021). Dry Beans Classification Using Machine Learning. CEUR Workshop Proceedings, Vol. 2951.

Menganalisis 13.611 sampel dengan teknik ML dan DL; akurasi berkisar 87.9 – 93.1% tergantung metode, menunjukkan potensi jaringan saraf pendalaman.

4. Zhang, L., & Kim, J. (2023). Data Mining Approach for Dry Bean Seeds Classification. Computers and Electronics in Agriculture, 205, 106639.

Mengembangkan sistem computer vision dan ML untuk klasifikasi biji, menggabungkan ekstraksi fitur geometris dan teknik ensemble; melaporkan peningkatan akurasi lewat tuning parameter.

### Sitasi

[1] M. Koklu and I. A. Ozkan, "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques," Computers and Electronics in Agriculture, vol. 175, p. 105552, 2020. [Online]. Available: https://www.semanticscholar.org/paper/Multiclass-classification-of-dry-beans-using-vision-Koklu-%C3%96zkan/e84c31138f2f261d15517d6b6bb8922c3fe597a1

[2] C.-Y. Lee, W. Wang, and J.-Q. Huang, "Clustering and Classification for Dry Bean Feature Imbalanced Data," Research Square, 2023. [Online]. Available: https://doi.org/10.21203/rs.3.rs-3616995/v1

[3] G. Słowiński, "Dry Beans Classification Using Machine Learning," in CEUR Workshop Proceedings, vol. 2951, 2021. [Online]. Available: http://ceur-ws.org/Vol-2951/paper16.pdf

[4] L. Zhang and J. Kim, "Data Mining Approach for Dry Bean Seeds Classification," Computers and Electronics in Agriculture, vol. 205, p. 106639, 2023. [Online]. Available: https://doi.org/10.1016/j.compag.2023.106639

## Business Understanding

Dalam industri pertanian dan pangan, identifikasi jenis tanaman secara akurat sangat penting untuk menjamin kualitas produk, efisiensi distribusi, dan pengambilan keputusan dalam rantai pasok. Salah satu produk pertanian yang umum dikonsumsi di berbagai negara adalah kacang kering (dry beans). Setiap jenis kacang memiliki karakteristik bentuk dan ukuran berbeda yang berpengaruh pada kegunaannya dalam masakan, nilai ekonomi, dan proses pengolahan.

Namun, identifikasi jenis kacang secara manual (misalnya berdasarkan bentuk visual atau pengukuran manual) memakan waktu dan rawan kesalahan. Oleh karena itu, diperlukan pendekatan otomatis berbasis data untuk mengklasifikasikan jenis kacang dengan lebih cepat dan akurat.

Bagian laporan ini mencakup:
### Problem Statements
1. Bagaimana membangun model klasifikasi yang akurat untuk mengidentifikasi varietas biji kering berdasarkan fitur morfologisnya?

2. Algoritma machine learning apa yang paling efektif dan efisien dalam mengklasifikasikan jenis kacang dari data numerik hasil ekstraksi morfologis?

3. Bagaimana cara mengatasi ketidakseimbangan jumlah sampel antar kelas serta meningkatkan performa model dalam konteks klasifikasi multi-kelas?

### Goals
1. Mengembangkan model machine learning yang mampu mengklasifikasikan tujuh varietas biji kering dengan akurasi tinggi berdasarkan fitur morfologis.

2. Mengevaluasi efektivitas tuning hyperparameter menggunakan Optuna pada model XGBoost dan membandingkannya dengan model baseline seperti Random Forest, Logistic Regression, SVM, dan KNN.

3. Melakukan proses penanganan data imbalance guna memperoleh model dengan generalisasi terbaik.

### Solution Statements
Untuk mencapai tujuan dalam mengklasifikasikan jenis kacang kering secara akurat dan efisien, dilakukan pendekatan sebagai berikut:

1. Mengimplementasikan model XGBoost dengan tuning hyperparameter menggunakan Optuna untuk mengoptimalkan performa klasifikasi.

2. Membandingkan performa model XGBoost (dengan dan tanpa tuning) dengan model baseline lainnya menggunakan metrik evaluasi seperti akurasi dan ROC AUC.

## Data Understanding

### Sumber Data
Dataset yang digunakan adalah Dry Bean Dataset dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), yang terdiri dari 13.611 sampel biji kering dari tujuh varietas berbeda. 


Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
Dataset ini memiliki 16 fitur numerik yang menggambarkan karakteristik morfologis biji, serta satu fitur target ('Class') yang menunjukkan varietas biji. Berikut adalah deskripsi singkat dari beberapa fitur:

- Area: Luas area biji dalam piksel.
- Perimeter: Keliling biji.
- MajorAxisLength: Panjang sumbu utama biji.
- MinorAxisLength: Panjang sumbu minor biji.
- AspectRation: Rasio aspek biji.
- Eccentricity: Eksentrisitas biji.
- ConvexArea: Luas area konveks biji.
- EquivDiameter: Diameter ekuivalen biji.
- Extent: Rasio area biji terhadap area bounding box.
- Solidity: Rasio area biji terhadap area konveks.
- Roundness: Tingkat kebulatan biji.
- Compactness: Tingkat kekompakan biji.
- ShapeFactor1-4: Faktor bentuk biji.
- Class: Label kelas yang menunjukkan varietas biji (Seker, Barbunya, Bombay, Cali, Horoz, Sira, Dermason).

**Rubrik/Kriteria Tambahan (Opsional)**:
### Exploratory Data Analysis
1. Visualisasi Distribusi Kelas
Tujuan: Memahami distribusi target (Class) sebelum pemodelan, karena ketidakseimbangan kelas bisa memengaruhi performa model klasifikasi.

2. Ringkasan Statistik Deskriptif
Tujuan: Memberikan gambaran umum nilai-nilai dari setiap fitur, mendeteksi kemungkinan outlier (nilai ekstrem), dan melihat sebaran nilai dari fitur numerik.

3. Boxplot untuk Menganalisis Hubungan Setiap Fitur dengan Kelas
Tujuan:

-- Melihat apakah fitur-fitur memiliki pola yang berbeda antar kelas (artinya fitur tersebut memiliki potensi baik untuk klasifikasi).

-- Mengidentifikasi sebaran data dan kemungkinan outlier pada masing-masing kelas.

-- Membantu dalam feature selection dan justifikasi kenapa fitur ini dipakai dalam modeling.

## Data Preparation
1. Reset Index : Mengatur ulang indeks DataFrame untuk memastikan setiap baris memiliki indeks yang berurutan
Alasan: Mencegah terjadinya duplikasi atau kesalahan saat mengakses baris berdasarkan indeks, serta menjaga integritas struktur data.

2. Seleksi Fitur Numerik : Memilih hanya kolom dengan tipe data numerik (float64, int64), karena hanya data numerik yang dapat digunakan dalam proses normalisasi dan pelatihan model.
 Alasan: Model supervised learning tidak dapat langsung menangani data kategorikal atau teks, sehingga fitur numerik menjadi fokus utama dalam preprocessing.

3. Normalisasi Data : Menggunakan StandardScaler untuk menstandarisasi nilai setiap fitur numerik agar memiliki mean = 0 dan standar deviasi = 1.
Alasan: Beberapa algoritma seperti SVM, KNN, dan XGBoost sensitif terhadap skala data.

4. Encoding Label : Label kelas berbentuk string diubah menjadi bilangan numerik menggunakan LabelEncoder.
Alasan: Model machine learning hanya dapat bekerja dengan input numerik. Encoding label diperlukan agar label target bisa dikenali dan diproses oleh model klasifikasi.

5. Handling Imbalanced Data, SMOTE : Membuat data sintetis untuk kelas minoritas menggunakan interpolasi.
Alasan: SMOTE menjaga keberagaman sampel dan meningkatkan generalisasi model.

6. Split Data: Data dibagi menjadi 67% untuk pelatihan dan 33% untuk pengujian, dengan stratifikasi pada label kelas.
Alasan: Stratifikasi memastikan distribusi kelas di train dan test set tetap proporsional dan menghindari overfitting


## Modeling
Pada tahap ini, dilakukan proses pelatihan model machine learning untuk menyelesaikan permasalahan klasifikasi jenis kacang berdasarkan fitur morfologinya. Proses modeling melibatkan beberapa algoritma populer, serta eksplorasi parameter untuk mendapatkan performa terbaik.

### Algoritma yang digunakan
1. XGBoost dengan Tuning Hyperparameter (Optuna)
- Parameter yang Dituning:
learning_rate: 0.005 - 0.05,
n_estimators: 100 - 1000,
reg_alpha: 1e-8 - 10.0,
reg_lambda: 1e-8 - 10.0,
max_depth: 3 - 20,
colsample_bytree: 0.3 - 1.0,
subsample: 0.5 - 1.0,
min_child_weight: 1 - 5
- Metode: Menggunakan StratifiedKFold dengan 10 fold untuk validasi silang.

2. Model Baseline untuk Perbandingan
- XGBoost Default
- Random Forest
- Logistic Regression
- SVM
- KNN

**Hasil Evaluasi Model**:
| Model               | Accuracy |  ROC AUC         |
| ------------------- | -------- | ---------------- |
| XGBoost + Optuna    | 95.05%   |  0.9972          |
| XGBoost (default)   | 94.93%   |  0.9971          |
| Random Forest       | 94.35%   |  0.9964          |
| Logistic Regression | 93.68%   |  0.9964          |
| SVM                 | 94.15%   |  0.9970          |
| KNN                 | 93.80%   |  0.9893          |

| Algoritma               | Kelebihan                                                                                                                    | Kekurangan                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **XGBoost (Optuna)**    | - Performa tinggi dan generalisasi baik <br> - Bisa menangani data imbang & tidak imbang <br> - Proses tuning terotomatisasi | - Butuh waktu komputasi lebih tinggi <br> - Banyak parameter yang perlu disesuaikan |
| **Random Forest**       | - Mudah digunakan <br> - Robust terhadap overfitting <br> - Explainable                                                      | - Kurang optimal untuk data high-dimensional <br> - Agak lambat jika banyak tree    |
| **Logistic Regression** | - Sederhana dan cepat <br> - Interpretasi mudah                                                                              | - Tidak menangani hubungan non-linear dengan baik                                   |
| **SVM**                 | - Bagus untuk dataset ukuran kecil hingga sedang <br> - Akurat pada margin sempit                                            | - Tidak efisien pada dataset besar <br> - Pemilihan kernel krusial                  |
| **KNN**                 | - Sederhana dan intuitif <br> - Tidak perlu pelatihan (lazy learner)                                                         | - Lambat pada prediksi data besar <br> - Sensitif terhadap outlier dan skala fitur  |


**Pemilihan Model Terbaik**
Model terbaik dipilih berdasarkan kombinasi skor akurasi dan ROC AUC:

XGBoost dengan tuning hyperparameter (Optuna) dipilih sebagai model terbaik, karena menghasilkan metrik evaluasi tertinggi dibandingkan semua model lain, tanpa mengalami overfitting (akurasi pelatihan ≈ 100%, akurasi pengujian 95.05%).


**Proses Improvement Model XGBoost**
Untuk meningkatkan performa model XGBoost, dilakukan proses hyperparameter tuning dengan library Optuna, yang merupakan framework otomatisasi tuning berbasis Bayesian Optimization.

Beberapa langkah dalam proses improvement:
1. Tuning Parameter:
Dilakukan pencarian kombinasi optimal parameter seperti learning_rate, max_depth, reg_alpha, subsample, dll.

2. Cross Validation:
Menggunakan StratifiedKFold dengan 10 fold untuk menghindari overfitting dan menjaga distribusi label antar fold.

3. Evaluasi dengan ROC AUC:
Menggunakan ROC AUC sebagai objective metric karena dapat menangani multi-class classification dengan baik menggunakan pendekatan One-vs-Rest.

4. Best Trial:

Trial terbaik ditemukan pada iterasi ke-83 dari total 100 trial, dengan hasil:

ROC AUC: 0.9972

Accuracy: 95.05%


# Kesimpulan Akhir
- Model XGBoost dengan tuning hyperparameter terbukti memberikan hasil terbaik untuk klasifikasi multi-kelas varietas biji kering, dengan akurasi 95.05% dan ROC AUC 0.9972.

- Penerapan model ini dapat menjadi solusi efisien dan akurat untuk membantu industri pertanian dalam proses identifikasi varietas biji secara otomatis.

- Proses tuning menggunakan Optuna terbukti meningkatkan performa dibandingkan model default atau baseline lainnya.
