import os
import shutil

old_dir = "training/v1_experimento"
new_dir = "training/spanish_w2v2_v1_gpu12gb"
script_path = "scripts/train_model.py"

# 1. Renombrar carpeta del modelo
if os.path.exists(old_dir):
    print(f"🔁 Renombrando: {old_dir} → {new_dir}")
    os.rename(old_dir, new_dir)
else:
    print(f"❌ No se encontró la carpeta {old_dir}")

# 2. Actualizar OUTPUT_DIR en el script de entrenamiento
with open(script_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(old_dir, new_dir)

with open(script_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"✅ Script actualizado: {script_path}")
print("🎯 ¡Todo listo con el nuevo nombre limpio y descriptivo!")
