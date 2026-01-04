# Klasifikasi Teks Ensemble untuk Dewey Decimal Classification (DDC)

Proyek ini mengimplementasikan sistem klasifikasi teks otomatis berbasis *Deep Learning* untuk mengategorikan data bibliografi ke dalam sepuluh kelas utama Dewey Decimal Classification (DDC 000-900). Sistem ini menggunakan pendekatan *ensemble* yang menggabungkan tiga model Transformer pra-latih (*pre-trained*) untuk mencapai akurasi prediksi yang lebih tinggi dibandingkan model tunggal.

## Ringkasan Proyek

Klasifikasi subjek buku secara manual memerlukan keahlian domain yang mendalam dan waktu yang signifikan. Solusi ini mengotomatisasi proses tersebut dengan memanfaatkan metadata buku (judul dan deskripsi). Arsitektur sistem dibangun di atas kerangka kerja *Hugging Face Transformers* dan *PyTorch*, menggunakan strategi penggabungan model (*model ensembling*) untuk meningkatkan generalisasi dan ketahanan model terhadap variasi data teks.

## Metodologi

### 1. Dataset dan Pra-pemrosesan
Data yang digunakan bersumber dari *OpenLibrary Data Dumps* yang telah melalui proses pembersihan dan penyaringan bahasa (hanya Bahasa Inggris).

* **Pemisahan Data:** Dataset dibagi menjadi data latih (80%) dan data validasi (20%) menggunakan metode *stratified sampling* untuk mempertahankan distribusi kelas.
* **Penyeimbangan Kelas (Class Balancing):** Untuk mengatasi ketidakseimbangan jumlah data antar kelas, diterapkan teknik *Random Oversampling* pada data latih. Setiap kelas dengan jumlah sampel di bawah 400 direplikasi hingga mencapai target 400 sampel per kelas.

### 2. Rekayasa Fitur (Feature Engineering)
Sistem menerapkan strategi pembobotan semantik pada teks input. Mengingat judul buku mengandung informasi yang padat dan krusial, judul diduplikasi dalam konstruksi input untuk memberikan atensi lebih besar pada mekanisme *self-attention* model.

Format input yang dikonstruksi:
$$Input = [Judul] + [Spasi] + [Judul] + [Spasi] + [Deskripsi]$$

Implementasi kode:
```python
df['text'] = (df['judul_buku'].astype(str) + " " + 
              df['judul_buku'].astype(str) + " " + 
              df['deskripsi'].astype(str))
