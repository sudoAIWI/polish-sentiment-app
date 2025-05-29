# src/scripts/export_model_to_onnx.py
import torch
from transformers import AutoModelForSequenceClassification
from pathlib import Path

def export_onnx():
    model_dir = "models/original"
    output_path = Path("models/onnx/model.onnx")
    output_path.parent.mkdir(exist_ok=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Dummy input for tracing
    dummy_input = torch.randint(0, 10000, (1, 128))  # (batch_size, seq_length)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=15,  # ONNX operator set version
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )

if __name__ == "__main__":
    export_onnx()