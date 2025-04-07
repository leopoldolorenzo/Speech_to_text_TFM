# scripts/reorganize_project.py

import os
import shutil

def move_tokenizer_to_model():
    src = "models/tokenizer_v1"
    dst = "training/v1_experimento/tokenizer"

    if os.path.exists(src):
        print(f"📦 Moviendo tokenizer de {src} a {dst}")
        shutil.move(src, dst)
    else:
        print("⚠️ Tokenizer no encontrado en models/tokenizer_v1")

def delete_old_models():
    paths = ["models/v1", "models/v1_reducido"]
    for path in paths:
        if os.path.exists(path):
            print(f"🗑️ Eliminando carpeta: {path}")
            shutil.rmtree(path)
        else:
            print(f"✅ No existe: {path}, nada que borrar")

def show_final_structure():
    print("\n📁 Estructura recomendada:")
    print("""
asr_project/
├── models/
│   └── base/
├── training/
│   └── v1_experimento/
│       └── tokenizer/
├── data/
│   └── common_voice_subset/
├── logs/
├── scripts/
""")

if __name__ == "__main__":
    print("🚀 Reorganizando proyecto...")
    move_tokenizer_to_model()
    delete_old_models()
    show_final_structure()
    print("\n✅ Proyecto reorganizado.")
