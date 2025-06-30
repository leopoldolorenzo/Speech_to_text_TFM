import React, { useState, useEffect, useRef } from 'react';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [transcription, setTranscription] = useState(null);
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [activeIndex, setActiveIndex] = useState(null);
  const activeSegmentRef = useRef(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('Por favor selecciona un archivo de audio');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/transcribe', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Error en el servidor');
      }

      const data = await response.json();
      console.log('Respuesta del backend:', data);
      setTranscription(data.transcription);
      setAudioUrl(data.audio_url);
    } catch (error) {
      console.error('Error al transcribir:', error);
      alert('Ocurrió un error al transcribir');
      setTranscription([]);
      setAudioUrl(null);
    } finally {
      setLoading(false);
    }
  };

  const speakerColors = {
    SPEAKER_00: '#60a5fa', // Azul-400
    SPEAKER_01: '#34d399', // Verde-400
    SPEAKER_02: '#22d3ee', // Cyan-400
    SPEAKER_03: '#c084fc', // Púrpura-400
    SPEAKER_04: '#f472b6', // Rosa-400
  };

  // Hook para actualizar el segmento activo mientras se reproduce el audio
  useEffect(() => {
    const audio = document.getElementById('audioPlayer');
    if (!audio || !transcription) return;

    const handleTimeUpdate = () => {
      const currentTime = audio.currentTime;
      const currentIndex = transcription.findIndex(
        (seg) =>
          currentTime >= parseFloat(seg.start) &&
          currentTime <= parseFloat(seg.end)
      );

      if (currentIndex !== activeIndex) {
        setActiveIndex(currentIndex);
      }
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [transcription, activeIndex]);

  // Hook para hacer scroll automático al segmento activo
  useEffect(() => {
    if (activeSegmentRef.current) {
      activeSegmentRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }, [activeIndex]);

  return (
    <div className="max-w-xl mx-auto mt-10 p-6 bg-gray-900 rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-4 text-center text-white">
        Transcripción de Audio
      </h1>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          className="border border-gray-600 rounded p-2 bg-gray-800 text-white"
        />

        <button
          type="submit"
          className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition duration-200"
          disabled={loading}
        >
          {loading ? 'Procesando...' : 'Subir y Transcribir'}
        </button>
      </form>

      {audioUrl && (
        <div className="mt-4">
          <h2 className="text-white font-semibold mb-2">Reproducir Audio:</h2>
          <audio id="audioPlayer" controls src={audioUrl} className="w-full">
            Tu navegador no soporta el reproductor de audio.
          </audio>
        </div>
      )}

      {transcription && Array.isArray(transcription) && transcription.length > 0 ? (
        <div className="mt-6 max-h-96 overflow-y-auto">
          <h2 className="text-xl font-semibold mb-2 text-white">Transcripción:</h2>
          <div className="space-y-2">
            {transcription.map((seg, idx) => {
              const isActive = activeIndex === idx;
              const speakerColor = speakerColors[seg.speaker] || '#d1d5db';
              const textColor = isActive ? 'text-[#FFD700]' : '';

              return (
                <div
                  key={idx}
                  ref={isActive ? activeSegmentRef : null}
                  className="p-2 border-b border-gray-700"
                >
                  <div className="font-bold text-gray-400">
                    {seg.speaker || 'Desconocido'}
                  </div>
                  <div className="text-sm text-gray-500">
                    {seg.start?.toFixed(2) || 0}s - {seg.end?.toFixed(2) || 0}s
                  </div>
                  <div
                    className={`text-lg font-semibold ${textColor}`}
                    style={{ color: !isActive ? speakerColor : undefined }}
                  >
                    {seg.grammar_corrected || seg.raw_text || 'No transcripción disponible'}
                  </div>
                  {seg.translation && (
                    <div className="text-green-400 mt-1">
                      Traducción: {seg.translation}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        transcription && (
          <p className="mt-4 text-center text-red-500">
            No se pudo obtener la transcripción. Por favor intenta con otro archivo.
          </p>
        )
      )}
    </div>
  );
};

export default UploadForm;
