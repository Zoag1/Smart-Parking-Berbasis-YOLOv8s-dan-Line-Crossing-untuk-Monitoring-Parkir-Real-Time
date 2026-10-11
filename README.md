# Smart Parking System (YOLOv8 & Line Crossing)

Proyek ini merupakan sistem pemantauan dan penghitungan kendaraan (motor) parkir otomatis berbasis YOLOv8 dan metode Line Crossing yang dilengkapi dengan fitur deteksi cahaya rendah.

1. Tujuan Proyek (Goals)
1.1. Otomatisasi Penghitungan: Memantau dan menghitung jumlah motor yang masuk ke area parkir secara real-time menggunakan kamera atau video.
1.2. Manajemen Slot Parkir: Menghitung sisa slot parkir secara otomatis berdasarkan kapasitas maksimum yang telah ditentukan.
1.3. Adaptasi Cahaya: Meningkatkan kualitas visual video pada kondisi pencahayaan rendah (gelap atau malam hari) menggunakan teknik Low Light Enhancement (CLAHE dan Gamma Correction).

2. Fitur Utama (Insights & Features)
2.1. Object Tracking: Menggunakan model YOLOv8 yang dikombinasikan dengan ByteTrack dan IoU untuk menghindari double counting dan masalah oklusi.
2.2. Adaptive Lighting: Sistem secara otomatis mendeteksi tingkat kecerahan frame dan menerapkan CLAHE serta koreksi gamma jika video terdeteksi gelap.
2.3. Logging & Evaluasi: Menyimpan hasil pemantauan ke dalam format file CSV (log_parkir_uksw.csv dan hasil_evaluasi.csv) untuk keperluan analisis data serta evaluasi akurasi terhadap Ground Truth.

3. Cara Menjalankan Proyek (How to Run)

3.1. Masuk ke Direktori Proyek
Buka terminal Anda (Anaconda Prompt atau Command Prompt) dan arahkan ke folder tempat penyimpanan proyek ini:
cd /path/ke/folder-proyek-anda
*(Contoh di Windows: cd D:\PROJECT\YOLOV8_Project)*

3.2. Instalasi Ketergantungan (Dependencies)
Instal seluruh pustaka Python yang dibutuhkan dengan menjalankan perintah:
pip install -r requirements.txt
*(Atau instal secara manual: pip install ultralytics opencv-python numpy pandas)*

3.3. Menjalankan Skrip Python
Jalankan program melalui terminal dengan format perintah universal berikut:
python YOLOV8.py --input <nama_folder>/<nama_video_sumber>.mp4 --output <nama_folder_hasil>/<nama_video_output>.avi

Keterangan Argumen:
- --input: Diisi dengan direktori dan nama file video asli yang ingin Anda analisis (contoh: videos/1.mp4).
- --output: Diisi dengan direktori tujuan dan nama bebas untuk file video hasil pemrosesan (contoh: output/1_test_result.avi).

Contoh Perintah Nyata:
python YOLOV8.py --input videos/1.mp4 --output output/1_test_result.avi

4. Kontak
Untuk pertanyaan atau diskusi lebih lanjut mengenai proyek ini, Anda dapat menghubungi saya melalui:
- Email: maresyatwattimena@gmail.com
- LinkedIn: linkedin.com/in/maresyawattimena

Terima kasih telah membaca! Saya terbuka untuk masukan dan saran terkait proyek ini. God bless you!
