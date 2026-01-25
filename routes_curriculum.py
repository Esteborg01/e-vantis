from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

print(">>> LOADED routes_curriculum.py FROM:", __file__)

router = APIRouter()

@router.get("/__curriculum_ping__")
def curriculum_ping():
    return {"router": "curriculum", "ok": True}

BASE_DIR = Path(__file__).resolve().parent
CURRICULUM_PATH = BASE_DIR / "curriculum" / "evantis.curriculum.v1.json"

@router.get("/curriculum/{subject_id}")
def get_curriculum(subject_id: str):
    if not CURRICULUM_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Curriculum file not found: {CURRICULUM_PATH}")

    try:
        curriculum = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read curriculum JSON: {str(e)}")

    subjects = curriculum.get("subjects", [])
    subject = next((s for s in subjects if s.get("id") == subject_id), None)

    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")

    return {
        "spec": curriculum.get("spec"),
        "version": curriculum.get("version"),
        "subject": subject
    }

