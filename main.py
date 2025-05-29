# src/scripts/main.py
import argparse
from download_artifacts import download_model
from export_model_to_onnx import export_onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", choices=["download", "export"], required=True)
    args = parser.parse_args()

    if args.script == "download":
        download_model()
    elif args.script == "export":
        export_onnx()