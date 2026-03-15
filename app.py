"""
Smart Attendance System — Flask Web Dashboard
"""

import os
import csv
import io
import numpy as np
import cv2
import torch
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, Response
)
from attendance_db import (
    init_db, log_attendance, get_today_attendance, get_attendance_by_date,
    get_all_dates, get_attendance_stats, delete_attendance_by_name
)
from recognition_engine import (
    engine, camera, generate_attendance_frames,
    generate_register_frames, capture_current_frame
)

# ============ APP SETUP ============
app = Flask(__name__)
app.secret_key = "smart-attendance-secret-key"

DATASET_FOLDER = "dataset"
EMBEDDINGS_FILE = "embeddings.npy"
UPLOAD_FOLDER = "uploads_temp"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize attendance database
init_db()


# ============ HELPERS ============
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_registered_people():
    """Get list of all registered people from embeddings file."""
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        return sorted(db.keys())
    return []


def get_people_count():
    """Get the number of registered people."""
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        return len(db)
    return 0


def register_person_from_uploads(person_name, files):
    """
    Register a person using uploaded image files.
    Saves photos to dataset folder and generates embedding.
    """
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained='vggface2').eval()

    person_folder = os.path.join(DATASET_FOLDER, person_name)
    os.makedirs(person_folder, exist_ok=True)

    embeddings_list = []
    saved_count = 0

    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            # Read image from upload
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Save photo
            photo_path = os.path.join(person_folder, f"{i + 1}.jpg")
            cv2.imwrite(photo_path, img)
            saved_count += 1

            # Generate embedding
            face_resized = cv2.resize(img, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                emb = model(face_tensor).numpy()[0]
            embeddings_list.append(emb)

    if len(embeddings_list) == 0:
        return False, "No valid face images were uploaded."

    # Average embedding
    avg_embedding = np.mean(embeddings_list, axis=0)

    # Load or create embeddings database
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    else:
        db = {}

    db[person_name] = avg_embedding
    np.save(EMBEDDINGS_FILE, db)

    return True, f"Registered '{person_name}' with {saved_count} photo(s)."


def delete_person(name):
    """Delete a person from embeddings and their dataset folder."""
    # Remove from embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        if name in db:
            del db[name]
            np.save(EMBEDDINGS_FILE, db)

    # Remove dataset folder
    person_folder = os.path.join(DATASET_FOLDER, name)
    if os.path.exists(person_folder):
        import shutil
        shutil.rmtree(person_folder)

    # Remove attendance records
    delete_attendance_by_name(name)


# ============ ROUTES ============

@app.route("/")
def dashboard():
    """Dashboard — today's attendance + summary stats."""
    stats = get_attendance_stats()
    stats["people_count"] = get_people_count()
    attendance = get_today_attendance()
    return render_template("dashboard.html", stats=stats, attendance=attendance)


@app.route("/history")
def history():
    """Attendance history — filter by date."""
    dates = get_all_dates()
    selected_date = request.args.get("date", "")
    attendance = []
    if selected_date:
        attendance = get_attendance_by_date(selected_date)
    return render_template(
        "history.html",
        dates=dates,
        selected_date=selected_date,
        attendance=attendance,
    )


@app.route("/people")
def people():
    """Registered people list."""
    names = get_registered_people()
    return render_template("people.html", people=names)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register a new person via photo upload."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        files = request.files.getlist("photos")

        if not name:
            flash("Please enter a name.", "error")
            return redirect(url_for("register"))

        if not files or all(f.filename == "" for f in files):
            flash("Please upload at least one photo.", "error")
            return redirect(url_for("register"))

        success, message = register_person_from_uploads(name, files)
        flash(message, "success" if success else "error")
        return redirect(url_for("people") if success else url_for("register"))

    return render_template("register.html")


# ============ API ENDPOINTS ============

@app.route("/api/attendance/<date>")
def api_attendance(date):
    """JSON API for attendance by date."""
    records = get_attendance_by_date(date)
    return jsonify(records)


@app.route("/api/export/<date>")
def api_export(date):
    """Download CSV export for a date."""
    records = get_attendance_by_date(date)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Date", "Time"])
    for r in records:
        writer.writerow([r["name"], r["date"], r["time"]])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=attendance_{date}.csv"},
    )


@app.route("/api/delete-person/<name>", methods=["POST"])
def api_delete_person(name):
    """Delete a registered person."""
    delete_person(name)
    flash(f"Deleted '{name}' successfully.", "success")
    return redirect(url_for("people"))


# ============ WEBCAM ROUTES ============

@app.route("/mark-attendance")
def mark_attendance():
    """Live webcam attendance marking page."""
    return render_template("mark_attendance.html")


@app.route("/video-feed/attendance")
def video_feed_attendance():
    """MJPEG stream with face recognition overlay."""
    return Response(
        generate_attendance_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/video-feed/register")
def video_feed_register():
    """MJPEG stream for registration (face detection only)."""
    return Response(
        generate_register_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/register-webcam", methods=["GET", "POST"])
def register_webcam():
    """Register a new person via webcam capture."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        count = int(request.form.get("photo_count", 0))

        if not name or count == 0:
            flash("Please enter a name and capture at least one photo.", "error")
            return redirect(url_for("register_webcam"))

        # Photos were already captured via API — now finalize the registration
        # Load saved embeddings for this person and average them
        person_folder = os.path.join(DATASET_FOLDER, name)
        if not os.path.exists(person_folder) or len(os.listdir(person_folder)) == 0:
            flash("No photos captured. Please capture photos first.", "error")
            return redirect(url_for("register_webcam"))

        # Generate embeddings from saved photos
        embeddings_list = []
        for img_file in sorted(os.listdir(person_folder)):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                emb = engine.get_embedding(img)
                embeddings_list.append(emb)

        if len(embeddings_list) == 0:
            flash("Could not generate embeddings from captured photos.", "error")
            return redirect(url_for("register_webcam"))

        avg_embedding = np.mean(embeddings_list, axis=0)

        # Save to embeddings database
        if os.path.exists(EMBEDDINGS_FILE):
            db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        else:
            db = {}
        db[name] = avg_embedding
        np.save(EMBEDDINGS_FILE, db)

        # Reload engine embeddings
        engine.reload_embeddings()

        # Stop camera
        camera.stop()

        flash(f"Registered '{name}' with {count} photo(s).", "success")
        return redirect(url_for("people"))

    return render_template("register_webcam.html")


@app.route("/api/capture-frame", methods=["POST"])
def api_capture_frame():
    """Capture a frame from webcam for registration."""
    data = request.get_json()
    name = data.get("name", "").strip()
    index = data.get("index", 1)

    if not name:
        return jsonify({"success": False, "message": "Name is required."})

    success, message, emb = capture_current_frame(name, index)
    return jsonify({"success": success, "message": message})


@app.route("/api/camera-stop", methods=["POST"])
def api_camera_stop():
    """Stop the camera."""
    camera.stop()
    return jsonify({"success": True, "message": "Camera stopped."})


# ============ MAIN ============
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
