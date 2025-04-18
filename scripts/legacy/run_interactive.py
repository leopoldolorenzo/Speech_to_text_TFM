import os
import subprocess

MODELS_DIR = "models"
AUDIO_DIR = "audios/convertidos"

def list_dir(path):
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) or f.endswith(".wav")])

def choose_option(options, prompt="Selecciona una opción"):
    print("\n" + prompt)
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    idx = int(input("→ Número: ")) - 1
    return options[idx]

def main():
    print("🎙️  Lanzador interactivo de transcripciones\n")

    models = list_dir(MODELS_DIR)
    if not models:
        print("❌ No hay modelos en la carpeta 'models/'")
        return

    model = choose_option(models, "📦 Elige el modelo a usar:")

    audios = list_dir(AUDIO_DIR)
    if not audios:
        print("❌ No hay archivos .wav en 'audios/convertidos/'")
        return

    audio = choose_option(audios, "🎧 Elige el audio a transcribir:")

    model_path = os.path.join(MODELS_DIR, model)
    audio_path = os.path.join(AUDIO_DIR, audio)

    print("\n🚀 Transcribiendo...")
    subprocess.run(["python", "scripts/pipeline_asr.py", model, audio_path])

if __name__ == "__main__":
    main()
