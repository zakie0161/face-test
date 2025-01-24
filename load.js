import http from 'k6/http';

export const options = {
  vus: 300, // Jumlah Virtual Users (bisa diatur sesuai kebutuhan)
  duration: '10s', // Durasi pengujian
};

const img1 = open('/home/wit.indonesia/Downloads/eminem.jpg', 'b');


export default function () {
  const url = 'http://localhost:8000/recognize';

  // Form data (multipart/form-data)
  const payload = {
    image: http.file(img1, 'eminem.jpg'), // Path ke file gambar
    name: 'Asep Sutisna', // Nama opsional yang diinputkan melalui form
  };

  // Kirim POST request dengan form-data
  const response = http.post(url, payload);

  // Log status response
  console.log(`Response status: ${response.status}`);
  console.log(`Response body: ${response.body}`);
}
