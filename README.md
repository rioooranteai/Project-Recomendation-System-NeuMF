# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Pada era anu, keterbukaan dan kemudahan untuk mengakses informasi merupakan suatu keuntungan yang didapatkan. Misalnya saja, 

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

## Data Understanding
Dataset yang digunakan adalah **Stroke Prediction Dataset** dari Kaggle ([https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)). Terdiri atas 5.110 entri data dengan 2 kelas. Data ini cukup mendekati karakteristik variabel yang menandai stroke berdasarkan riset yang telah disampaikan pada latar belakang.

### 3.1 Deskripsi Dataset

* a. Ratings
| No. | Kolom    | Tipe Data | Jumlah Non-Null | Deskripsi                                                              |
| --- | -------- | --------- | --------------- | ---------------------------------------------------------------------- |
| 1   | user\_id | int64     | 981,756         | ID pengguna yang memberikan rating pada buku                           |
| 2   | book\_id | int64     | 981,756         | ID buku yang diberi rating                                             |
| 3   | rating   | int64     | 981,756         | Nilai rating yang diberikan oleh pengguna terhadap buku (biasanya 1–5) |

* b. Books
| No. | Kolom                       | Tipe Data | Jumlah Non-Null | Deskripsi                                                   |
| --- | --------------------------- | --------- | --------------- | ----------------------------------------------------------- |
| 1   | id                          | int64     | 10,000          | ID unik untuk setiap entri buku                             |
| 2   | book\_id                    | int64     | 10,000          | ID buku, kemungkinan digunakan untuk relasi antar tabel     |
| 3   | best\_book\_id              | int64     | 10,000          | ID dari versi terbaik buku (representatif satu buku)        |
| 4   | work\_id                    | int64     | 10,000          | ID karya umum (bisa terdiri dari banyak edisi)              |
| 5   | books\_count                | int64     | 10,000          | Jumlah edisi berbeda dari buku tersebut                     |
| 6   | isbn                        | object    | 9,300           | ISBN 10-digit dari buku                                     |
| 7   | isbn13                      | float64   | 9,415           | ISBN 13-digit dari buku                                     |
| 8   | authors                     | object    | 10,000          | Nama penulis buku                                           |
| 9   | original\_publication\_year | float64   | 9,979           | Tahun asli penerbitan pertama buku                          |
| 10  | original\_title             | object    | 9,415           | Judul asli dari buku (tanpa subtitle atau versi terjemahan) |
| 11  | title                       | object    | 10,000          | Judul yang ditampilkan di dataset                           |
| 12  | language\_code              | object    | 8,916           | Kode bahasa buku (misal: 'en', 'spa')                       |
| 13  | average\_rating             | float64   | 10,000          | Rata-rata rating dari semua pengguna Goodreads              |
| 14  | ratings\_count              | int64     | 10,000          | Jumlah total rating yang diterima buku                      |
| 15  | work\_ratings\_count        | int64     | 10,000          | Jumlah rating untuk seluruh edisi (berdasarkan `work_id`)   |
| 16  | work\_text\_reviews\_count  | int64     | 10,000          | Jumlah ulasan teks yang diberikan pengguna                  |
| 17  | ratings\_1                  | int64     | 10,000          | Jumlah rating bintang 1                                     |
| 18  | ratings\_2                  | int64     | 10,000          | Jumlah rating bintang 2                                     |
| 19  | ratings\_3                  | int64     | 10,000          | Jumlah rating bintang 3                                     |
| 20  | ratings\_4                  | int64     | 10,000          | Jumlah rating bintang 4                                     |
| 21  | ratings\_5                  | int64     | 10,000          | Jumlah rating bintang 5                                     |
| 22  | image\_url                  | object    | 10,000          | URL gambar sampul buku                                      |
| 23  | small\_image\_url           | object    | 10,000          | URL gambar sampul kecil (thumbnail)                         |

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**
