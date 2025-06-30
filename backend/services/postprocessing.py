# backend/services/postprocessing.py

def simple_postprocess(text):
    """
    Postprocesado básico: capitaliza la primera letra y añade un punto final si falta.
    """
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text
