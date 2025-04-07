# scripts/export_lexicon.py
VOCAB_PATH = "data/vocab/vocab.txt"
LEXICON_PATH = "data/vocab/lexicon.txt"

with open(VOCAB_PATH, "r", encoding="utf-8") as vocab_file:
    vocab = [line.strip() for line in vocab_file if line.strip()]

with open(LEXICON_PATH, "w", encoding="utf-8") as f:
    for token in vocab:
        if token in ["<s>", "</s>", "<unk>"]:
            f.write(f"{token} {' '.join(token)}\n")
        elif len(token) == 1:
            f.write(f"{token} {token}\n")
        else:
            f.write(f"{token} {' '.join(list(token))}\n")

print("✅ lexicon.txt generado:", LEXICON_PATH)
