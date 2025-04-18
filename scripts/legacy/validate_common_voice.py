# scripts/validate_common_voice.py

from datasets import load_from_disk

DATASET_PATH = "data/common_voice"

print(f"📂 Cargando dataset desde: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)

for split in ["train", "test"]:
    if split not in dataset:
        print(f"⚠️ No se encontró el split '{split}'")
        continue

    print(f"\n🔍 Validando split: {split}")
    total = len(dataset[split])
    valid = 0
    missing_transcript = 0
    missing_audio = 0

    for example in dataset[split]:
        has_audio = example.get("audio") is not None and "array" in example["audio"]
        has_text = example.get("sentence") is not None and example["sentence"].strip() != ""

        if has_audio and has_text:
            valid += 1
        if not has_audio:
            missing_audio += 1
        if not has_text:
            missing_transcript += 1

    print(f"🟢 Válidos: {valid}/{total}")
    print(f"❌ Sin audio: {missing_audio}")
    print(f"❌ Sin transcripción: {missing_transcript}")

    # Mostrar algunos ejemplos válidos
    print("\n🔎 Ejemplos:")
    for example in dataset[split].select(range(3)):
        print(f"📝 Transcripción: {example['sentence']}")
        print(f"🎧 Audio sample rate: {example['audio']['sampling_rate']}")
        print("-" * 40)
