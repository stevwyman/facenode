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

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Verwendet DeepFace.extract_faces (RetinaFace‑Backend) und gibt
    Bounding‑Boxes in Form von x, y, width, height zurück.
    """
    try:
        img = _to_cv2_image(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bild‑Lese‑Fehler: {exc}")

    # DeepFace.extract_faces gibt eine Liste von dicts (oder numpy‑Arrays)
    # Wir setzen detector_backend explizit auf retinaface, weil wir das Modell
    # bereits heruntergeladen haben (automatischer Download beim ersten Aufruf).
    try:
        detections = DeepFace.extract_faces(
            img_path=img,               # es kann ein numpy‑Array sein
            detector_backend="retinaface",
            enforce_detection=False,    # gibt leere Liste zurück, wenn nichts gefunden
            align=False                 # wir brauchen keine Ausrichtung
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DeepFace‑Fehler: {exc}")

    # Jede Detektion ist ein dict mit 'region' (bounding box)
    # Beispiel‑Eintrag: {'face': <ndarray>, 'region': {'x': 124, 'y': 45, 'w': 98, 'h': 98}}
    result = []
    for det in detections:
        region = det.get('region')
        if region:
            result.append({
                "x": int(region["x"]),
                "y": int(region["y"]),
                "width": int(region["w"]),
                "height": int(region["h"]),
                "confidence": 1.0  # RetinaFace liefert kein Confidence‑Score per DeepFace‑Wrapper
            })

    return JSONResponse(content={"faces": result})