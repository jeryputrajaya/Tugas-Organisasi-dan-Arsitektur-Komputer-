# Tugas Organisasi dan Arsitektur Komputer: Direct Loss Estimation (NannyML)

KELOMPOK 1 
1. Jery Putra Jaya ( G1A025002 )
2. Naufal Faris KS ( G1A025014 )
3. Ahmad Najib R ( G1A025022 )
4. M. Azhar Robbani ( G1A025032 )
5. Karel Desvalyudho ( G1A025040 )
6. M. Daffa Atallah Putra (G1A025044)
7. Maqbul Fath ( G1A025056 )
8. Raja Marganda ( G1A025086 )
9. M. Fakhrury Qhenno Afkar (G1A025098)
10. Muhammad Fadhil Rizky Sakti (G1A025036)

Repositori ini dibuat untuk memenuhi tugas mata kuliah **Organisasi dan Arsitektur Komputer** (Universitas Bengkulu). Proyek ini menjalankan, memvisualisasikan, dan menganalisis notebook Kaggle berjudul **"direct_loss_estimation (nannyml)"** karya Samuele Mazzanti menggunakan **Superconductor Dataset** melalui **Google Colab**.

---

## Daftar Isi
1. [Gambaran Besar & Masalah yang Diselesaikan](#1-gambaran-besar--masalah-yang-diselesaikan)
2. [Deskripsi Dataset & Pembagian Data](#2-deskripsi-dataset--pembagian-data)
3. [Metodologi & Alur Kerja Kode](#3-metodologi--alur-kerja-kode)
4. [Hasil Evaluasi Numerik (`metrics_log`)](#4-hasil-evaluasi-numerik-metrics_log)
5. [Penjelasan Ketujuh Visualisasi Output](#5-penjelasan-ketujuh-visualisasi-output)
6. [Analisis Kritis & Temuan Kelemahan Kode Asli](#6-analisis-kritis--temuan-kelemahan-kode-asli)
7. [Struktur File dalam Repositori](#7-struktur-file-dalam-repositori)
8. [Panduan Menjalankan Kode (Google Colab & Lokal)](#8-panduan-menjalankan-kode-google-colab--lokal)

---

## 1. Gambaran Besar & Masalah yang Diselesaikan

Dalam siklus hidup *machine learning*, performa model regresi umumnya diukur pada data uji (*test set*) menggunakan metrik seperti **Mean Absolute Error (MAE)** dan **Mean Squared Error (MSE)** karena label atau nilai target asli ($y$) masih tersedia. Namun, ketika model sudah diterapkan di dunia nyata (*production*), data baru yang masuk sering kali **belum memiliki label asli (*ground truth*)**. Misalnya, untuk mengetahui suhu kritis superkonduktor secara nyata diperlukan pengujian fisik di laboratorium yang memakan waktu dan biaya.

**Direct Loss Estimation (DLE)** menyelesaikan masalah ini dengan cara memperkirakan tingkat kesalahan (*loss/error*) model utama secara langsung pada data produksi **tanpa memerlukan label asli**.

### Cara Kerja Utama DLE:
1. **Model Utama:** Dilatih pada data latih (*train set*) untuk memprediksi target utama ($\hat{y}$).
2. **Model Penduga Error (*Error Estimator*):** Dilatih pada data uji (*test set*) di mana kesalahan model utama masih bisa dihitung ($e = \hat{y} - y$). Model kedua ini mempelajari pola kapan dan pada karakteristik fitur seperti apa model utama cenderung melakukan kesalahan besar atau kecil.
3. **Estimasi di Produksi:** Saat memproses data produksi tanpa label, model penduga *error* melihat fitur data produksi beserta angka tebakan model utama, lalu langsung mengestimasi nilai MAE dan MSE.

### Diagram Alur Sistem

```text
[BAGIAN 1: ILUSTRASI KONSEP (Data Sintetis)]
10.000 Sampel Lognormal (Rata-rata = 50.000)
   ├──> posterior.png (Sebaran kemungkinan nilai target)
   ├──> posterior_and_point.png (+ Garis merah Point Prediction di 50.000)
   └──> posterior_and_point_and_error.png (+ Garis oranye rentang Point Prediction +- MAE)

[BAGIAN 2: EKSPERIMEN DLE (Superconductor Dataset: 21.263 baris, 82 kolom)]
                   │
      (Diacak dengan random_state=123)
                   │
     ┌─────────────┴─────────────┬───────────────────────────┐
     ▼                           ▼                           ▼
[1. TRAIN SET]              [2. TEST SET]               [3. PRODUCTION SET]
(4.252 baris - 20%)         (4.253 baris - 20%)         (12.758 baris - 60%)
     │                           │                           │
Latih Model Utama ─────────> Prediksi: pred_test        Prediksi: pred_prod
(LGBMRegressor)              Hitung: error_test              │
                                 │                           │
                    Latih 3 Model Penduga Error              │
                    (Fitur: X_test + pred_test)              │
                    ├─> model_abs_error (|e|)   ────────┐    │
                    ├─> model_error (e)         ────────┼───>├─> Estimasi Error Produksi
                    └─> model_sq_error (e^2)    ────────┘    │   (Tanpa melihat y_prod!)
                                                             ▼
                                                Urutkan berdasarkan pred_prod,
                                                bagi 10 Fold & acak urutan Fold
                                                             │
                                                             ▼
                                                Tabel Evaluasi (metrics_log)
                                                ├─> mae_obs_vs_pred.png & mse_obs_vs_pred.png
                                                │   (Memakai pred_se_* -> Estimasi DLE Sangat Akurat)
                                                └─> mae_obs_vs_pred_se.png & mse_obs_vs_pred_se.png
                                                    (Memakai pred_e_* -> Pendekatan Naif Meleset Jauh)
```

---

## 2. Deskripsi Dataset & Pembagian Data

* **Sumber Dataset:** Superconductor Dataset (`train.csv`).
* **Dimensi Data:** $21.263$ baris dan $82$ kolom.
  * **Fitur ($X$):** $81$ kolom numerik yang merepresentasikan sifat fisik dan kimia unsur penyusun bahan superkonduktor (seperti massa atom rata-rata, jari-jari atom, afinitas elektron, dan konduktivitas termal).
  * **Target ($y$):** $1$ kolom target yaitu `critical_temp` (suhu kritis superkonduktor dalam satuan Kelvin).
* **Strategi Pembagian Data:**
  Seluruh baris dataset diacak terlebih dahulu menggunakan `.sample(frac=1, random_state=123)` agar distribusi data merata dan eksperimen bersifat *reproducible*, kemudian dibagi menjadi 3 bagian:

| Dataset | Proporsi | Jumlah Baris | Fungsi dalam Eksperimen |
| :--- | :--- | :--- | :--- |
| `df_train` | $\frac{1}{5}$ pertama ($20\%$) | $4.252$ baris | Melatih model regresi utama (`model`) untuk memprediksi `critical_temp`. |
| `df_test` | $\frac{1}{5}$ kedua ($20\%$) | $4.253$ baris | Mengevaluasi model utama dan melatih ketiga model penduga *error* (`model_abs_error`, `model_error`, `model_sq_error`). |
| `df_prod` | $\frac{3}{5}$ sisanya ($60\%$) | $12.758$ baris | Disimulasikan sebagai data produksi tanpa label untuk menguji keakuratan estimasi DLE pada 10 kelompok (*fold*). |

---

## 3. Metodologi & Alur Kerja Kode

### Bagian 1: Simulasi Distribusi Posterior & Prediksi Titik (Sel 1–4)
Sebelum masuk ke data superkonduktor, notebook mendemonstrasikan konsep ketidakpastian prediksi menggunakan $10.000$ sampel sintetis berdistribusi *log-normal* (`np.random.lognormal(0, .25, 10000) * 50000`) yang dinormalisasi sehingga rata-ratanya tepat bernilai $50.000$.
* **Point Prediction ($\hat{y}$):** Nilai rata-rata dari distribusi posterior ($50.000$).
* **Error ($e$):** Selisih antara prediksi titik dengan setiap kemungkinan nilai sebenarnya pada distribusi posterior.
* **Mean Absolute Error (MAE):** Rata-rata jarak mutlak antara kemungkinan nilai sebenarnya dengan prediksi titik.

### Bagian 2: Implementasi DLE dengan LightGBM (Sel 5–15)
1. **Pelatihan Model Utama (Sel 6):**
   Menggunakan algoritma *Gradient Boosting* berbasis pohon keputusan (`LGBMRegressor`) pada `X_train` dan `y_train`.
2. **Prediksi & Pemotongan Nilai Negatif di Test Set (Sel 7):**
   Hasil prediksi pada data uji dipotong menggunakan `.clip(0)` agar nilai minimumnya adalah $0$ (karena suhu kritis dalam satuan Kelvin tidak mungkin bernilai negatif). Selanjutnya dihitung selisih kesalahan:
   $$e_{\text{test}} = \hat{y}_{\text{test}} - y_{\text{test}}$$
3. **Pelatihan Tiga Model Penduga Error di Test Set (Sel 8):**
   Ketiga model penduga *error* dilatih pada data *test* (bukan data *train*, agar tidak mengalami *overfitting* karena *error* di data *train* sudah terlalu kecil). Fitur masukannya adalah gabungan fitur asli ditambah hasil prediksi model utama (`pd.concat([X_test, pred_test], axis=1)`):
   * `model_abs_error`: Belajar memprediksi kesalahan mutlak $|e_{\text{test}}|$ secara langsung (Pendekatan DLE untuk MAE).
   * `model_error`: Belajar memprediksi kesalahan bertanda $e_{\text{test}}$ yang bisa bernilai positif atau negatif (Pendekatan pembanding/naif).
   * `model_sq_error`: Belajar memprediksi kesalahan kuadrat $e_{\text{test}}^2$ secara langsung (Pendekatan DLE untuk MSE).
4. **Estimasi pada Data Produksi (Sel 9):**
   Model utama menghasilkan prediksi `pred_prod` pada `X_prod`. Ketiga model penduga *error* kemudian memprediksi kesalahan (`pred_error_prod`, `pred_abs_error_prod`, dan `pred_sq_error_prod`) hanya berbekal `X_prod` dan `pred_prod` tanpa menyentuh `y_prod`.
5. **Rumus Metrik Evaluasi (Sel 10):**
   * **Mean Absolute Error (MAE):**
     $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   * **Mean Squared Error (MSE):**
     $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
6. **Pembagian 10 Fold Produksi & Penyesuaian Kompatibilitas Google Colab (Sel 11):**
   Data produksi diurutkan berdasarkan nilai `pred_prod` dari terkecil hingga terbesar, dibagi menjadi 10 kelompok (*chunks/folds*) menggunakan `np.array_split`, lalu urutan kelompoknya diacak dengan `seed(123)`. Penggunaan pengurutan berdasarkan nilai prediksi bertujuan menciptakan perbedaan karakteristik data yang ekstrem antar-*fold* (simulasi *data drift*), sedangkan pengacakan urutan *fold* menunjukkan bahwa sumbu X bukan merupakan urutan waktu (*time series*).
   
   *(Catatan Kompatibilitas Colab: Pada Python 3.13 / NumPy versi terbaru di Google Colab, pemanggilan `np.random.permutation` langsung pada hasil `np.array_split` yang berbeda ukuran 1 baris antara 1.276 dan 1.275 baris akan memicu `ValueError`. Oleh karena itu, perulangan disesuaikan dengan mengacak indeks urutan chunk tanpa mengubah logika maupun angka hasil akhir):*
   ```python
   chunks = np.array_split(pred_prod.sort_values().index, 10)
   order = np.random.permutation(len(chunks))
   for i, k in enumerate(order):
       ix = chunks[k]
   ```

---

## 4. Hasil Evaluasi Numerik (`metrics_log`)

Berdasarkan hasil eksekusi pada Google Colab, nilai kesalahan pada *test set* bersifat konstan (dihitung satu kali untuk seluruh *test set*), yaitu:
* **`test_mae`** = $7,086210$ ($\approx 7,09$)
* **`test_mse`** = $131,806707$ ($\approx 131,81$)

### Tabel Lengkap `metrics_log` pada 10 Fold Data Produksi

| Fold | `test_mae` | `true_mae` | `pred_se_mae` (DLE) | `pred_e_mae` (Naif) | Selisih MAE DLE | `test_mse` | `true_mse` | `pred_se_mse` (DLE) | `pred_e_mse` (Naif) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $7,09$ | $6,30$ | $6,28$ | $2,26$ | $0,02$ | $131,81$ | $81,10$ | $64,68$ | $2,26$ |
| **1** | $7,09$ | $1,46$ | $1,62$ | $0,79$ | $0,16$ | $131,81$ | $5,58$ | $13,92$ | $0,79$ |
| **2** | $7,09$ | $14,27$ | $14,25$ | $6,37$ | $0,02$ | $131,81$ | $347,55$ | $342,20$ | $6,37$ |
| **3** | $7,09$ | $7,50$ | $7,61$ | $3,32$ | $0,11$ | $131,81$ | $107,21$ | $120,96$ | $3,32$ |
| **4** | $7,09$ | $10,64$ | $10,07$ | $5,13$ | $0,57$ | $131,81$ | $233,66$ | $207,89$ | $5,13$ |
| **5** | $7,09$ | $4,46$ | $4,54$ | $1,80$ | $0,08$ | $131,81$ | $34,66$ | $37,70$ | $1,80$ |
| **6** | $7,09$ | $2,13$ | $2,19$ | $0,82$ | $0,06$ | $131,81$ | $8,14$ | $12,81$ | $0,82$ |
| **7** | $7,09$ | $12,88$ | $11,96$ | $5,32$ | $0,92$ | $131,81$ | $315,70$ | $275,17$ | $5,32$ |
| **8** | $7,09$ | $9,03$ | $9,07$ | $5,40$ | $0,04$ | $131,81$ | $204,69$ | $206,29$ | $5,40$ |
| **9** | $7,09$ | $3,04$ | $2,95$ | $1,05$ | $0,09$ | $131,81$ | $20,06$ | $16,74$ | $1,05$ |
| **Rata-rata** | **$7,09$** | **$7,17$** | **$7,05$** | **$3,23$** | **$0,21$** | **$131,81$** | **$135,83$** | **$129,84$** | **$3,23$** |

### Temuan Penting dari Angka Tabel:
1. **Keakuratan Tinggi Metode DLE:** Rata-rata `pred_se_mae` ($7,05$) sangat dekat dengan rata-rata `true_mae` ($7,17$). Demikian pula rata-rata `pred_se_mse` ($129,84$) mendekati rata-rata `true_mse` ($135,83$).
2. **Selisih Terkecil dan Terbesar:**
   * Selisih mutlak terkecil antara `true_mae` dan `pred_se_mae` terjadi pada **Fold 0** ($|6,30 - 6,28| = 0,02$) dan **Fold 2** ($|14,27 - 14,25| = 0,02$).
   * Selisih mutlak terbesar terjadi pada **Fold 7** ($|12,88 - 11,96| = 0,92$).
3. **Variasi Ekstrem Antar-Fold:** Nilai `true_mae` di produksi berfluktuasi tajam mulai dari $1,46$ (Fold 1) hingga $14,27$ (Fold 2). Mengandalkan satu angka `test_mae` ($7,09$) saja tidak mampu mendeteksi lonjakan *error* pada kelompok data tertentu, sedangkan DLE berhasil melacaknya dengan presisi.

---

## 5. Penjelasan Ketujuh Visualisasi Output

### A. Kelompok Gambar Ilustrasi Konsep (Data Sintetis)
1. **`posterior.png`**
   * **Apa yang digambar:** Histogram berwarna abu-abu (50 *bins* dari $20.000$ hingga $100.000$) yang menampilkan sebaran $10.000$ kemungkinan nilai variabel target.
   * **Mengapa bentuknya miring ke kanan (*right-skewed*):** Karena dibangkitkan dari distribusi *log-normal*, di mana nilai-nilai ekstrem tinggi memiliki ekor sebaran yang memanjang ke sisi kanan.
2. **`posterior_and_point.png`**
   * **Apa yang digambar:** Histogram distribusi posterior ditambah garis vertikal putus-putus berwarna merah tepat di angka $50.000$ (`Point Prediction`).
   * **Makna:** Menunjukkan bahwa satu angka prediksi yang dikeluarkan model regresi hanyalah nilai tengah/ekspektasi dari suatu distribusi kemungkinan.
3. **`posterior_and_point_and_error.png`**
   * **Apa yang digambar:** Menambahkan dua garis vertikal putus-putus berwarna oranye di sebelah kiri dan kanan garis merah sebesar $\text{Point Prediction} \pm \text{MAE}$ (sekitar $50.000 \pm 10.000$, yakni di sekitar $40.000$ dan $60.000$).
   * **Makna:** Mengilustrasikan bahwa MAE adalah ukuran rata-rata lebar simpangan nilai sebenarnya terhadap titik tebakan model.

### B. Kelompok Gambar Evaluasi Superconductor Dataset (Data Produksi)
Pada keempat grafik berikut, **Sumbu X** adalah nomor kelompok data produksi (`Production fold` $0$ sampai $9$), **Sumbu Y** adalah besarnya nilai *error* (MAE atau MSE), **Garis Abu-abu Datar** adalah *error* pada *test set*, **Garis Biru** adalah *error* sebenarnya di produksi (`Observed`), dan **Garis Merah** adalah *error* hasil perkiraan model penduga (`Predicted`).

4. **`mae_obs_vs_pred.png` (MAE Asli vs Estimasi DLE `pred_se_mae`)**
   * **Mengapa bentuknya naik-turun tajam dan berimpit:** Grafik naik-turun secara ekstrem karena setiap *fold* berisi kelompok nilai prediksi yang berbeda rentangnya (yang diacak urutannya). Garis merah (`pred_se_mae` dari `model_abs_error`) hampir berimpit sempurna dengan garis biru (`true_mae`), membuktikan bahwa memprediksi $|e|$ secara langsung sangat efektif mengestimasi MAE tanpa label.
5. **`mse_obs_vs_pred.png` (MSE Asli vs Estimasi DLE `pred_se_mse`)**
   * **Analisis:** Garis merah (`pred_se_mse` dari `model_sq_error`) berhasil mengikuti puncak dan lembah garis biru (`true_mse`) dengan sangat akurat (misalnya menebak puncak Fold 2 di angka $342,20$ saat nilai aslinya $347,55$).
6. **`mae_obs_vs_pred_se.png` (MAE Asli vs Estimasi Naif `pred_e_mae`)**
   * **Mengapa garis merah jauh di bawah garis biru:** Grafik ini memakai `pred_e_mae` yang berasal dari `model_error` (model yang menebak *error* bertanda positif/negatif). Karena sifat regresi memprediksi nilai rata-rata kondisional $\mathbb{E}[e \mid X, \hat{y}]$, kesalahan positif dan negatif saling meniadakan sehingga tebakan model mendekati $0$. Akibatnya, saat diambil nilai mutlaknya, hasil estimasi MAE selalu terlalu rendah (*underestimated*).
7. **`mse_obs_vs_pred_se.png` (MSE Asli vs Estimasi Naif `pred_e_mse`)**
   * **Mengapa garis merah tampak rata di dekat angka nol:** Selain karena efek saling meniadakan pada `model_error`, terdapat *bug* pada fungsi `get_metrics` yang membuat nilai `pred_e_mse` sama persis dengan `pred_e_mae` (maksimal hanya $6,37$), sehingga tampak sangat kecil dan mendatar ketika disandingkan dengan skala `true_mse` yang mencapai $347,55$.

---

## 6. Analisis Kritis & Temuan Kelemahan Kode Asli

Berdasarkan pemeriksaan menyeluruh terhadap kode asli notebook Kaggle, terdapat beberapa catatan kritis yang penting untuk diketahui:

1. **Kesalahan Penulisan Rumus (*Bug*) pada Fungsi `get_metrics`:**
   Pada baris kode:
   ```python
   sq_error = sq_error if sq_error is not None else np.abs(error)
   ```
   Ketika `sq_error` tidak diberikan (yaitu saat menghitung `metrics_pred_error_prod`), kode justru mengisi `sq_error` dengan nilai absolut `np.abs(error)`, bukan mengkuadratkannya (`error ** 2`). Akibatnya, kolom `pred_e_mse` memiliki angka yang **100% identik** dengan kolom `pred_e_mae`, sehingga grafik `mse_obs_vs_pred_se.png` bukan menampilkan MSE yang sebenarnya.
2. **Penamaan Variabel dan File Gambar yang Menyesatkan (*Misleading Naming*):**
   * Variabel yang menampung hasil pendekatan DLE diberi nama `pred_se_*` (padahal untuk MAE berasal dari `model_abs_error`, bukan *squared error*), tetapi disimpan ke dalam file bernama `mae_obs_vs_pred.png` dan `mse_obs_vs_pred.png` (tanpa akhiran `_se`).
   * Sebaliknya, variabel pendekatan naif (`pred_e_*` dari `model_error`) justru disimpan ke file bernama `mae_obs_vs_pred_se.png` dan `mse_obs_vs_pred_se.png`.
3. **Tidak Mengimpor Pustaka `nannyml`:**
   Meskipun judul notebook mencantumkan `(nannyml)`, kode di dalamnya sama sekali tidak menggunakan `import nannyml`. Penulis mengimplementasikan cara kerja algoritma DLE milik NannyML secara manual menggunakan `LGBMRegressor` dari pustaka `lightgbm`.
4. **Sumbu X Bukan Urutan Waktu (*Time Series*):**
   Karena `df_prod` diurutkan berdasarkan besarnya nilai prediksi (`pred_prod.sort_values()`) sebelum dibagi menjadi 10 *fold* dan diacak, maka `Production fold` $0$ sampai $9$ pada sumbu X merepresentasikan kelompok rentang nilai prediksi yang diacak, bukan urutan waktu kronologis di produksi.

---

## 7. Struktur File dalam Repositori

* `README.md` — Dokumentasi lengkap proyek, penjelasan kode, tabel hasil, dan analisis visualisasi.
* `direct_loss_estimation_nannyml_dengan_penjelasan.ipynb` — File Jupyter Notebook (Google Colab) lengkap dengan penjelasan bahasa Indonesia di setiap sel dan output grafik.
* `direct_loss_estimation_nannyml_Kelompok1.py` — File skrip Python murni hasil ekspor kode eksperimen.
* `metrics_log.csv` — Tabel hasil perhitungan numerik metrik MAE dan MSE pada seluruh 10 *fold* produksi.
* `outputs/` (atau file gambar `.png`):
  * `posterior.png`
  * `posterior_and_point.png`
  * `posterior_and_point_and_error.png`
  * `mae_obs_vs_pred.png`
  * `mse_obs_vs_pred.png`
  * `mae_obs_vs_pred_se.png`
  * `mse_obs_vs_pred_se.png`
* File Presentasi (`.pptx` / `.pdf`) — Materi presentasi tugas kelompok.

---

## 8. Panduan Menjalankan Kode (Google Colab & Lokal)

### Opsi A: Menjalankan di Google Colab (Direkomendasikan)
1. Buka [Google Colab](https://colab.research.google.com/) dan unggah file `direct_loss_estimation_nannyml_dengan_penjelasan.ipynb`.
2. Unggah file dataset `train.csv` (dari *Superconductor Dataset*) ke panel **Files** di sebelah kiri Google Colab.
3. Pastikan jalur pembacaan file pada sel Langkah 1 mengarah ke `'train.csv'` (atau gunakan URL repositori dataset yang tersedia di dalam skrip).
4. Klik menu **Runtime > Run all** untuk mengeksekusi seluruh sel dan mengunduh ketujuh gambar grafik `.png`.

### Opsi B: Menjalankan di Komputer Lokal
1. Buat dan aktifkan *virtual environment* Python, lalu pasang pustaka yang dibutuhkan:
   ```bash
   pip install numpy pandas scikit-learn lightgbm matplotlib
   ```
2. Letakkan file `train.csv` di dalam folder yang sama dengan skrip Python.
3. Jalankan skrip melalui terminal atau Command Prompt:
   ```bash
   python direct_loss_estimation_nannyml_Kelompok1.py
   ```
