# Smart Parking Berbasis YOLOv8s dan Line Crossing untuk Monitoring Parkir Real Time
Proyek ini merupakan sistem pemantauan dan penghitungan kendaraan (motor) parkir otomatis berbasis YOLOv8 dan metode _Line Crossing_ yang dilengkapi dengan fitur deteksi cahaya rendah.

### Tujuan Proyek 
  1. Otomatisasi Penghitungan: Memantau dan menghitung jumlah motor yang masuk ke area parkir secara _real-time_ menggunakan kamera atau video.
  2. Manajemen Slot Parkir: Menghitung sisa slot parkir secara otomatis berdasarkan kapasitas maksimum yang telah ditentukan.
  3. Adaptasi Cahaya: Meningkatkan kualitas visual video pada kondisi pencahayaan rendah (gelap atau malam hari) menggunakan teknik _Low Light Enhancement_ (_CLAHE dan Gamma Correction_).

### Fitur Utama 
  1. _Object Tracking_: Menggunakan model YOLOv8 yang dikombinasikan dengan ByteTrack dan IoU untuk menghindari _double counting_ dan masalah oklusi.
  2. _Adaptive Lighting_: Sistem secara otomatis mendeteksi tingkat kecerahan _frame_ dan menerapkan CLAHE serta koreksi gamma jika video terdeteksi gelap.
  3. _Logging_ & Evaluasi: Menyimpan hasil pemantauan ke dalam format file CSV (log_parkir_uksw.csv dan hasil_evaluasi.csv) untuk keperluan analisis data serta evaluasi akurasi terhadap _Ground Truth_.

### Cara Menjalankan Proyek
  1. Masuk ke Direktori Proyek
     Buka terminal Anda (Anaconda Prompt atau Command Prompt) dan arahkan ke folder tempat penyimpanan proyek ini: `cd/path/ke/folder-proyek-anda`. <br>
     (Contoh di Windows: `cd D:\PROJECT\YOLOV8_Project`)

### Instal Dependencies
  1. Instal seluruh pustaka Python yang dibutuhkan dengan menjalankan perintah: `pip install -r requirements.txt`
  2. Atau instal secara manual: `pip install ultralytics opencv-python numpy pandas`

### Menjalankan Skrip Python
1. Jalankan program melalui terminal dengan format perintah universal berikut: <br>
   `python YOLOV8.py --input <nama_folder>/<nama_video_sumber>.mp4 --output <nama_folder_hasil>/<nama_video_output>.avi`

### Keterangan Argumen
1. `input`: Diisi dengan direktori dan nama file video asli yang ingin Anda analisis. <br>
   (contoh: `videos/1.mp4`).
3. `output`: Diisi dengan direktori tujuan dan nama bebas untuk file video hasil pemrosesan. <br>
   (contoh: `output/1_test_result.avi`).

### Contoh Perintah Nyata
`python YOLOV8.py --input videos/1.mp4 --output output/1_test_result.avi`

### Kontak
Untuk pertanyaan atau diskusi lebih lanjut, Anda dapat menghubungi saya melalui: <br>
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:maresyatwattimena@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maresyawattimena)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/echawttmna_/)


Terima kasih telah membaca! Saya terbuka untuk masukan dan saran terkait proyek ini. <br>
God bless you!
