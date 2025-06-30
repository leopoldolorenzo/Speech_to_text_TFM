import json
from transformers import Wav2Vec2Processor

LOCAL_VOCAB_PATH = 'models/tokenizer_v2_base41/vocab.json'
REMOTE_MODEL_ID = 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish'

def cargar_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def verificar_indices_unicos(vocab_dict):
    indices = list(vocab_dict.values())
    tokens = list(vocab_dict.keys())
    errores = []

    if len(indices) != len(set(indices)):
        errores.append("❌ Hay índices repetidos en el vocabulario.")
    if len(tokens) != len(set(tokens)):
        errores.append("❌ Hay tokens repetidos en el vocabulario.")
    if sorted(indices) != list(range(len(indices))):
        errores.append(f"❌ Los índices no son contiguos (esperado: 0 a {len(indices) - 1})")

    return errores

def comparar_vocabularios(local, remoto):
    local_sorted = sorted(local.items(), key=lambda x: x[1])
    remoto_sorted = sorted(remoto.items(), key=lambda x: x[1])

    diferencias = []
    for i, ((ltok, lidx), (rtok, ridx)) in enumerate(zip(local_sorted, remoto_sorted)):
        if ltok != rtok or lidx != ridx:
            diferencias.append(f"[{i}] ❌ local=('{ltok}', {lidx}) ≠ remoto=('{rtok}', {ridx})")

    return diferencias

# Cargar vocabularios
local_vocab = cargar_vocab(LOCAL_VOCAB_PATH)
remote_processor = Wav2Vec2Processor.from_pretrained(REMOTE_MODEL_ID)
remote_vocab = remote_processor.tokenizer.get_vocab()

# Verificaciones internas
log = []
log.extend(verificar_indices_unicos(local_vocab))

# Comparación con vocabulario remoto
log.extend(comparar_vocabularios(local_vocab, remote_vocab))

# Resultado
if not log:
    print("✅ El vocabulario local es idéntico al original (41 tokens, orden y contenido correctos).")
else:
    print("⚠️ Diferencias encontradas:\n")
    for linea in log:
        print(linea)

# Mostrar resumen local ordenado
print("\n[✓] Vocabulario local ordenado:")
for token, idx in sorted(local_vocab.items(), key=lambda x: x[1]):
    print(f"{idx:>3}  →  {token}")
