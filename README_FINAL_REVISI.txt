VERSI FINAL REVISI

Fitur yang digabungkan:
1. Buka Kamera dari browser.
2. Tampilan kamera memiliki grid Braille.
3. Tampilan kamera memiliki 4 sudut panduan scan seperti QR/document scanner.
4. Foto dari kamera/upload dapat diluruskan otomatis jika kertas miring.
5. YOLO .pt tetap dipakai untuk segmentasi/grid per 1 karakter Braille.
6. Hasil karakter A-Z tetap ditampilkan, tidak dihapus.
7. Hasil karakter digabung menjadi suku kata sederhana pola konsonan + vokal.
   Contoh: b+u=bu, k+u=ku.
8. Suku kata digabung menjadi kata.
   Contoh: i+bu=ibu, bu+ku=buku.
9. Tombol suara membacakan alur pembelajaran:
   b u menjadi bu. k u menjadi ku. bu ditambah ku menjadi buku.

Catatan:
- Akurasi tetap bergantung pada kualitas model lama dan kualitas foto.
- Jika masih muncul huruf/tanda yang salah, itu berasal dari prediksi model, bukan dari logika penggabungan.


Revisi sinkronisasi suara-grid:
- Saat suara membaca unit suku kata, misalnya "bu" lalu "ditambah ku", grid suku kata ikut aktif/ melayang sesuai unit yang sedang dibaca.
- Saat suara membaca kata final, misalnya "menjadi buku", grid berhenti sejenak agar anak dan orang tua fokus pada hasil kata.
- Setelah kata final selesai, grid lanjut bergerak pada gabungan suku kata berikutnya.
