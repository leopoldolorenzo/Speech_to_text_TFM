#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensuciar_audio.py

Script para 'ensuciar' un archivo de audio MP3 de manera flexible:
- Añade ruido blanco.
- Permite ajustar la cantidad de ruido.
- Puede incluir distorsión y pitch shifting.
- Opcionalmente añade eco.
"""

import argparse
from pydub import AudioSegment
import numpy as np
import os

def cargar_audio(ruta_audio):
    """Carga un archivo de audio MP3 y lo devuelve como AudioSegment y muestras NumPy."""
    audio = AudioSegment.from_file(ruta_audio)
    muestras = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        muestras = muestras.reshape((-1, 2))
    return audio, muestras

def agregar_ruido(muestras, factor_ruido):
    """Añade ruido blanco gaussiano a las muestras de audio."""
    ruido = np.random.normal(0, factor_ruido * np.max(np.abs(muestras)), muestras.shape)
    muestras_ruidosas = muestras + ruido
    # Clipping
    muestras_ruidosas = np.clip(muestras_ruidosas, np.iinfo(muestras.dtype).min, np.iinfo(muestras.dtype).max)
    return muestras_ruidosas.astype(muestras.dtype)

def distorsionar(muestras, ganancia):
    """Aplica distorsión simulando recorte."""
    muestras_distorsionadas = muestras * ganancia
    muestras_distorsionadas = np.clip(muestras_distorsionadas, np.iinfo(muestras.dtype).min, np.iinfo(muestras.dtype).max)
    return muestras_distorsionadas.astype(muestras.dtype)

def cambiar_pitch(audio, semitonos):
    """Cambia el pitch del audio (cambia la velocidad)."""
    audio_shifted = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * (2 ** (semitonos / 12)))
    }).set_frame_rate(audio.frame_rate)
    return audio_shifted

def agregar_eco(audio, retraso_ms=200, atenuacion=0.6):
    """Añade un eco simple al audio."""
    # Crear un segmento de silencio con la duración del retraso
    silencio = AudioSegment.silent(duration=retraso_ms)
    # Crear la pista del eco: silencio seguido de la pista original, atenuado
    eco = silencio + (audio - (30 * atenuacion))
    # Combinar la pista original con el eco
    audio_con_eco = audio.overlay(eco)
    return audio_con_eco

def exportar_audio(audio, ruta_salida):
    """Exporta el audio a un archivo MP3."""
    audio.export(ruta_salida, format="mp3")
    print(f"¡Listo! Audio exportado a: {ruta_salida}")

def main():
    parser = argparse.ArgumentParser(description="Ensucia un archivo MP3 con ruido, distorsión y efectos.")
    parser.add_argument("archivo_entrada", help="Ruta al archivo de audio MP3 de entrada.")
    parser.add_argument("archivo_salida", help="Ruta al archivo de salida MP3.")
    parser.add_argument("--ruido", type=float, default=0.05, help="Factor de ruido blanco (0.0 - 1.0). Default: 0.05")
    parser.add_argument("--distorsion", type=float, default=1.0, help="Factor de distorsión (1.0 = sin distorsión). Default: 1.0")
    parser.add_argument("--pitch", type=float, default=0.0, help="Cambio de pitch en semitonos. Default: 0.0")
    parser.add_argument("--eco", action="store_true", help="Añadir eco básico.")
    parser.add_argument("--delay", type=int, default=200, help="Retardo del eco en ms. Default: 200ms")
    parser.add_argument("--atenuacion", type=float, default=0.6, help="Atenuación del eco (0.0 - 1.0). Default: 0.6")
    args = parser.parse_args()

    if not os.path.exists(args.archivo_entrada):
        print("ERROR: El archivo de entrada no existe.")
        return

    audio, muestras = cargar_audio(args.archivo_entrada)

    # Añadir ruido
    muestras = agregar_ruido(muestras, args.ruido)

    # Añadir distorsión si corresponde
    if args.distorsion != 1.0:
        muestras = distorsionar(muestras, args.distorsion)

    # Convertir de nuevo a AudioSegment
    audio_procesado = audio._spawn(muestras.tobytes())

    # Aplicar pitch shifting si corresponde
    if args.pitch != 0.0:
        audio_procesado = cambiar_pitch(audio_procesado, args.pitch)

    # Agregar eco si se selecciona
    if args.eco:
        audio_procesado = agregar_eco(audio_procesado, args.delay, args.atenuacion)

    exportar_audio(audio_procesado, args.archivo_salida)

if __name__ == "__main__":
    main()
