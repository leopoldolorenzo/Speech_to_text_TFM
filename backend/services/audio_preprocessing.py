#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocesar_audio.py

Script para limpiar y preprocesar un archivo de audio:
- Analiza si necesita limpieza.
- Aplica normalización.
- Convierte a mono.
- Resamplea a 16kHz.
- Revierte el cambio de pitch si corresponde.
- Aplica limpieza automática:
  - Filtro paso alto (ruido grave)
  - Filtro paso bajo (ruido agudo)
  - Reducción de ruido avanzada con noisereduce.
- Exporta a .wav.

Nota: Solo limpia el audio si detecta suciedad (o si se usa --forzar).
"""

import os
import numpy as np
from pydub import AudioSegment, effects
import scipy.signal
import librosa
import noisereduce as nr

def needs_filtering(input_path, low_freq_threshold=0.005, high_freq_threshold=0.01):
    """
    Analiza el audio y decide si aplicar filtros:
    - Mide la proporción de energía en bajas (<100Hz) y altas (>7000Hz) frecuencias.
    - Devuelve True si alguna supera el umbral configurado.
    """
    print(f"🔎 Analizando si el audio necesita limpieza: {input_path}")

    try:
        y, sr = librosa.load(input_path, sr=16000)
    except Exception as e:
        print(f"❌ Error al cargar el archivo con librosa: {e}")
        return False  # Por defecto, no limpiar si no podemos analizar

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    low_freq_energy = np.sum(S[(freqs < 100), :]) / np.sum(S)
    high_freq_energy = np.sum(S[(freqs > 7000), :]) / np.sum(S)

    print(f"🔎 Energía baja: {low_freq_energy:.4f}, energía alta: {high_freq_energy:.4f}")

    if low_freq_energy > low_freq_threshold or high_freq_energy > high_freq_threshold:
        print("⚠️ Se detectó ruido. Aplicando limpieza.")
        return True
    else:
        print("✅ El audio está limpio. No se aplica limpieza.")
        return False

def revert_pitch(audio, semitonos):
    """
    Revierte el cambio de pitch aplicado (ej. por el script de ensuciar_audio).
    """
    print(f"🔄 Revirtiendo pitch en {semitonos} semitonos.")
    reverted_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate / (2 ** (semitonos / 12)))
    }).set_frame_rate(audio.frame_rate)
    return reverted_audio

def apply_noise_reduction(y, sr):
    """
    Reducción avanzada de ruido usando noisereduce.
    """
    print("🔧 Aplicando reducción de ruido con noisereduce.")
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    return reduced_noise

def preprocess_audio(input_path, output_path, force_cleaning=False, revert_pitch_shift=2):
    """
    Limpia y preprocesa el audio:
    - Normaliza volumen
    - Convierte a mono
    - Resamplea a 16kHz
    - Revierte el cambio de pitch si corresponde
    - Aplica limpieza si se detecta suciedad o si se fuerza
    - Exporta a .wav
    """
    print(f"🔧 Preprocesando el audio: {input_path}")

    if not os.path.isfile(input_path) or os.path.getsize(input_path) < 1024:
        raise RuntimeError(f"El archivo {input_path} es demasiado pequeño o no existe.")

    AudioSegment.converter = "/usr/bin/ffmpeg"
    AudioSegment.ffprobe = "/usr/bin/ffprobe"

    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        raise RuntimeError("No se pudo abrir el archivo. Verifica ffmpeg y el archivo de audio.") from e

    print(f"ℹ️ Duración: {len(audio) / 1000:.2f} segundos")

    # Normalizar
    audio = effects.normalize(audio)

    # Convertir a mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
        print("🎚️ Convertido a mono")

    # Resamplear
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
        print("🔄 Resampleado a 16kHz")

    # Revertir pitch
    if revert_pitch_shift != 0:
        audio = revert_pitch(audio, revert_pitch_shift)

    # Verificar si necesita limpieza
    apply_filter = force_cleaning or needs_filtering(input_path)

    if apply_filter:
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= np.max(np.abs(samples))  # Normalizar para noisereduce
        sr = 16000

        # Filtro paso alto (ruidos graves)
        sos_hp = scipy.signal.butter(6, 100, btype='highpass', fs=sr, output='sos')
        samples = scipy.signal.sosfilt(sos_hp, samples)

        # Filtro paso bajo (ruidos agudos)
        sos_lp = scipy.signal.butter(6, 7000, btype='lowpass', fs=sr, output='sos')
        samples = scipy.signal.sosfilt(sos_lp, samples)

        # Reducción de ruido con noisereduce
        samples = apply_noise_reduction(samples, sr)

        # Convertir de nuevo a int16
        samples_int16 = np.int16(samples * 32767)
        audio = AudioSegment(
            samples_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        print("✅ Filtros aplicados.")
    else:
        print("✅ Audio limpio: no se aplicaron filtros.")

    audio.export(output_path, format="wav")
    print(f"✅ Audio exportado a: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Limpia y preprocesa un archivo de audio.")
    parser.add_argument("archivo_entrada", help="Archivo de audio de entrada.")
    parser.add_argument("archivo_salida", help="Archivo de audio de salida.")
    parser.add_argument("--forzar", action="store_true", help="Forzar limpieza incluso si el audio está limpio.")
    parser.add_argument("--revert_pitch", type=float, default=2, help="Revertir cambio de pitch en semitonos. Default: 2.")
    args = parser.parse_args()

    preprocess_audio(args.archivo_entrada, args.archivo_salida,
                     force_cleaning=args.forzar,
                     revert_pitch_shift=args.revert_pitch)
