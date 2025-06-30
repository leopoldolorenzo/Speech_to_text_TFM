# backend/services/postprocessing.py

import re
import requests
from spellchecker import SpellChecker

def spell_check(text):
    """
    Aplica corrección ortográfica palabra por palabra usando SpellChecker.
    """
    spell = SpellChecker(language="es")
    corrected_words = []
    for word in text.split():
        if spell.unknown([word]):
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def prepare_for_lt(text):
    """
    Ajusta el texto para LanguageTool: capitalización y puntuación final.
    """
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text

def grammar_check(text, port=8081):
    """
    Aplica corrección gramatical con LanguageTool (HTTP local).
    """
    url = f"http://localhost:{port}/v2/check"
    response = requests.post(url, data={"language": "es", "text": text})
    response.raise_for_status()
    matches = response.json().get("matches", [])

    # Aplicar reemplazos en orden inverso para evitar desplazamiento de offsets
    corrected = list(text)
    for match in sorted(matches, key=lambda m: m["offset"], reverse=True):
        if match["replacements"]:
            replacement = match["replacements"][0]["value"]
            offset = match["offset"]
            length = match["length"]
            corrected[offset:offset+length] = replacement
    return "".join(corrected)
