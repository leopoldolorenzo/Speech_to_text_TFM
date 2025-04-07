import os
import matplotlib.pyplot as plt
from transformers.integrations import TensorBoardCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# === CONFIG ===
LOG_DIR = "logs"  # carpeta usada en TrainingArguments

# === Buscar los archivos de eventos de TensorBoard ===
def find_event_file(log_dir):
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                return os.path.join(root, file)
    raise FileNotFoundError("❌ No se encontró archivo de eventos en el log.")

# === Extraer WER y pérdida ===
def extract_metrics(event_file):
    ea = EventAccumulator(event_file)
    ea.Reload()

    steps = []
    wer_values = []
    loss_values = []

    if "eval/wer" in ea.Tags()["scalars"]:
        for event in ea.Scalars("eval/wer"):
            steps.append(event.step)
            wer_values.append(event.value)

    if "eval/loss" in ea.Tags()["scalars"]:
        loss_values = [event.value for event in ea.Scalars("eval/loss")]

    return steps, wer_values, loss_values

# === Graficar ===
def plot_metrics(steps, wer, loss):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("WER", color="tab:red")
    ax1.plot(steps, wer, color="tab:red", label="WER")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="tab:blue")
    ax2.plot(steps, loss, color="tab:blue", label="Loss")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Evaluación durante Fine-Tuning")
    fig.tight_layout()
    plt.grid()
    plt.show()

# === Ejecutar ===
if __name__ == "__main__":
    event_file = find_event_file(LOG_DIR)
    steps, wer_values, loss_values = extract_metrics(event_file)
    plot_metrics(steps, wer_values, loss_values)
