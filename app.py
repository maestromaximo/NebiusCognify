# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import uvicorn
import os
import tempfile
import asyncio

app = FastAPI(title="Faster-Whisper API", description="API for audio transcription using Faster-Whisper.", version="1.0.0")

# Initialize the Whisper model
# Choose model size based on your requirements and system resources
MODEL_SIZE = "large-v3"  # Options: tiny, base, small, medium, large, large-v3, distil-large-v3

# Device: "cuda" for GPU, "cpu" for CPU
DEVICE = "cuda" if os.environ.get("USE_GPU", "true").lower() in ["true", "1", "yes"] else "cpu"

# Compute type: "float16", "int8", etc.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Initialize the model
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"Loaded Whisper model '{MODEL_SIZE}' on {DEVICE} with compute type '{COMPUTE_TYPE}'.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    raise e

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), beam_size: int = 5):
    """
    Endpoint to transcribe uploaded audio files.
    
    - **file**: Audio file to transcribe (mp3, wav, etc.)
    - **beam_size**: Beam size for transcription (default: 5)
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Perform transcription
    try:
        segments, info = model.transcribe(tmp_path, beam_size=beam_size)
        transcription = ""
        async for segment in segments:
            transcription += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # Clean up the temporary file
        os.remove(tmp_path)

    response = {
        "language": info.language,
        "language_probability": info.language_probability,
        "transcription": transcription.strip()
    }

    return JSONResponse(content=response)

if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
