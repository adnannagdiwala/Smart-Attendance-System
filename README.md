# ⚡ Smart Attendance System

A face recognition-powered attendance tracking system with a modern web dashboard. Built with **Flask**, **OpenCV**, **PyTorch**, and **FaceNet** — everything runs locally through your browser.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)

---

## ✨ Features

- **📷 Live Webcam Attendance** — Open your camera in the browser, faces are auto-recognized and attendance is logged instantly
- **📸 Webcam Registration** — Capture face photos directly from your camera to register new people
- **📤 Photo Upload Registration** — Upload images to register people without a camera
- **📊 Dashboard** — Real-time stats: registered people, check-ins today, latest log
- **📅 Attendance History** — Browse past records by date with CSV export
- **👥 People Management** — View, add, and remove registered individuals
- **🔒 Cooldown System** — Prevents duplicate logs within a 5-minute window
- **🌙 Premium Dark UI** — Glassmorphism design with smooth animations

---

## 🖥️ Screenshots

| Dashboard | Mark Attendance |
|:-:|:-:|
| Stats cards + today's attendance table | Live webcam with auto-recognition |

| People | Register via Webcam |
|:-:|:-:|
| Grid view with avatar cards | Camera feed + capture controls |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam (for live attendance and webcam registration)
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/adnannagdiwala/Smart-Attendance-System.git
cd Smart-Attendance-System

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
Smart-Attendance-System/
├── app.py                    # Flask web app (routes + APIs)
├── recognition_engine.py     # Face recognition engine + MJPEG streaming
├── attendance_db.py          # SQLite attendance database
├── register_face.py          # CLI face registration (standalone)
├── face_recognition_system.py # CLI webcam attendance (standalone)
├── embeddings.py             # Batch embedding generator
├── requirements.txt          # Python dependencies
│
├── templates/                # HTML templates
│   ├── base.html             # Shared layout + sidebar
│   ├── dashboard.html        # Dashboard page
│   ├── mark_attendance.html  # Live webcam attendance
│   ├── history.html          # Attendance history
│   ├── people.html           # Registered people
│   ├── register.html         # Upload photo registration
│   └── register_webcam.html  # Webcam registration
│
├── static/
│   ├── css/style.css         # Dark theme stylesheet
│   └── js/main.js            # Client-side interactivity
│
├── dataset/                  # Face photos (per person)
├── attendance_logs/          # SQLite database
└── embeddings.npy            # Face embedding vectors
```

---

## 🔧 How It Works

1. **Face Registration** — Photos are captured (webcam or upload) and processed through **InceptionResnetV1 (FaceNet)** to generate 512-dimensional face embeddings, stored in `embeddings.npy`.

2. **Face Recognition** — During attendance, the webcam feed is processed frame-by-frame: faces are detected using **Haar Cascade**, embeddings are extracted, and compared against stored embeddings using **cosine similarity**.

3. **Attendance Logging** — When a face is recognized (similarity > threshold), attendance is logged to an **SQLite** database with a 5-minute cooldown to prevent duplicates.

4. **MJPEG Streaming** — The Flask server opens the webcam via OpenCV, processes frames server-side, and streams annotated video to the browser in real-time.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend** | Flask (Python) |
| **Face Detection** | OpenCV Haar Cascade |
| **Face Recognition** | FaceNet (InceptionResnetV1 via PyTorch) |
| **Database** | SQLite |
| **Frontend** | HTML, CSS, JavaScript |
| **Streaming** | MJPEG over HTTP |

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/attendance/<date>` | GET | Get attendance records (JSON) |
| `/api/export/<date>` | GET | Download attendance CSV |
| `/api/delete-person/<name>` | POST | Delete a registered person |
| `/api/capture-frame` | POST | Capture webcam frame for registration |
| `/api/camera-stop` | POST | Release the webcam |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/adnannagdiwala">Adnan Nagdiwala</a>
</p>
