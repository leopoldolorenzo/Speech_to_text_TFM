# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # NUEVO
from backend.api.endpoints import router as api_router

app = FastAPI(title="TFM ASR API")

# Permitir CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto a http://localhost:5173 si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos de audio procesados
app.mount("/audios", StaticFiles(directory="data/diarizacion"), name="audios") 

# Incluir el router principal
app.include_router(api_router)

# Solo ejecuta el servidor si se llama directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
