# preparar_batches_ctc.py
# Clase para preparar lotes (batches) con padding automático para entrenamiento CTC.

from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

@dataclass
class PreparadorBatchesCTC:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separar audio (input_values) y transcripciones (labels)
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Aplicar padding a los audios
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Aplicar padding a las etiquetas (transcripciones)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Reemplazar los tokens de relleno con -100 para ignorarlos en el cálculo de la pérdida
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

# ✅ Mensaje de prueba para confirmar que se puede importar correctamente
print("✅ Módulo de preparación de batches CTC disponible.")
