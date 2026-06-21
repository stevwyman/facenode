# service.py  (angepasst, nutzt DeepFace.extract_faces)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np, cv2
from deepface import DeepFace
import io

app = FastAPI(title="DeepFace‑Detection‑Service")

def _to_cv2_image(file_bytes: bytes) -> np.ndarray:
    """Konvertiert Bytes → OpenCV‑Bild (BGR)."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Bild konnte nicht dekodiert werden.")
    return img


async def detect_faces_old(file: UploadFile = File(...)):
    """
    Verwendet DeepFace.extract_faces (RetinaFace‑Backend) und gibt
    Bounding‑Boxes in Form von x, y, width, height zurück.
    """
    try:
        img = _to_cv2_image(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bild‑Lese‑Fehler: {exc}")

    try:
        detections = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False,
            align=False
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DeepFace‑Fehler: {exc}")

    result = []
    for det in detections:
        # 🔥 HIER IST DER FIX: Der Key heißt 'facial_area'
        region = det.get('facial_area')
        
        # DeepFace liefert mittlerweile bei fast allen Backends (auch RetinaFace) eine Confidence mit!
        confidence = det.get('confidence', 0.0)

        # Wir prüfen, ob eine Region da ist UND filtern die "Fake"-Gesichter heraus,
        # die DeepFace bei enforce_detection=False mit confidence=0 anlegt.
        if region and confidence > 0:
            result.append({
                "x": int(region["x"]),
                "y": int(region["y"]),
                "width": int(region["w"]),
                "height": int(region["h"]),
                "confidence": float(confidence)
            })

    return JSONResponse(content={"faces": result})


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Findet Gesichter UND berechnet deren mathematische Embeddings (Vektoren).
    Nutzt RetinaFace für die Suche und Facenet512 für die Wiedererkennung.
    """
    try:
        img = _to_cv2_image(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bild-Lese-Fehler: {exc}")

    try:
        # 🔥 NEU: DeepFace.represent liefert Box + Embedding!
        detections = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",       # Hochpräzises Erkennungs-Modell (512 Zahlen)
            detector_backend="retinaface",
            enforce_detection=False,
            align=True                     # WICHTIG: Richtet das Gesicht gerade aus
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DeepFace-Fehler: {exc}")

    result = []
    for det in detections:
        region = det.get('facial_area')
        embedding = det.get('embedding') # Das ist unsere Liste aus 512 Zahlen
        confidence = det.get('confidence', 0.0)

        # Fake-Treffer filtern
        if region and confidence > 0 and embedding:
            result.append({
                "x": int(region["x"]),
                "y": int(region["y"]),
                "width": int(region["w"]),
                "height": int(region["h"]),
                "confidence": float(confidence),
                "embedding": embedding  # 🔥 Wir geben den Vektor an Django weiter!
            })

    return JSONResponse(content={"faces": result})