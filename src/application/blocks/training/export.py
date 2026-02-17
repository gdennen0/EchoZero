"""
Model Export

Provides ONNX export and model quantization for production deployment.
ONNX models are compatible with the PyTorchAudioClassifyBlockProcessor.
"""
import sys
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from src.utils.message import Log

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    _mod = sys.modules.get(__name__)
    if _mod is not None and not getattr(_mod, "_logged_onnx", False):
        Log.debug("onnx not available - ONNX export disabled")
        setattr(_mod, "_logged_onnx", True)


def export_onnx(
    model: "nn.Module",
    config: Dict[str, Any],
    model_path: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Export a trained model to ONNX format.

    Creates a .onnx file alongside the .pth model file.
    Compatible with the PyTorchAudioClassifyBlockProcessor
    which uses onnxruntime for inference.

    Args:
        model: Trained PyTorch model in eval mode
        config: Training configuration (for input shape)
        model_path: Path to the .pth model file (used to derive ONNX path)
        output_path: Optional custom output path for the ONNX file

    Returns:
        Path to the exported ONNX file, or None if export failed
    """
    if not HAS_PYTORCH:
        Log.error("PyTorch required for ONNX export")
        return None

    if not HAS_ONNX:
        Log.warning("onnx package not available. Install with: pip install onnx")
        return None

    try:
        model.eval()

        # Determine input shape from config
        n_mels = config.get("n_mels", 128)
        max_length = config.get("max_length", 22050)
        hop_length = config.get("hop_length", 512)
        time_steps = max_length // hop_length + 1

        # Create dummy input: (batch=1, channels=1, freq=n_mels, time=time_steps)
        dummy_input = torch.randn(1, 1, n_mels, time_steps)

        # Determine output path
        if output_path:
            onnx_path = Path(output_path)
        else:
            onnx_path = Path(model_path).with_suffix(".onnx")

        # Move model to CPU for export
        model_cpu = model.cpu()

        # Export
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        # Validate the exported model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        Log.info(f"ONNX model exported to {onnx_path}")
        return str(onnx_path)

    except Exception as e:
        Log.error(f"ONNX export failed: {e}")
        return None


def quantize_model(
    model: "nn.Module",
    model_path: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Apply dynamic quantization to reduce model size.

    Converts float32 weights to int8, reducing model size by ~4x
    with minimal accuracy loss. Useful for production deployment.

    Args:
        model: Trained PyTorch model
        model_path: Path to the original .pth model file
        output_path: Optional custom output path

    Returns:
        Path to the quantized model file, or None if quantization failed
    """
    if not HAS_PYTORCH:
        Log.error("PyTorch required for quantization")
        return None

    try:
        model.eval()
        model_cpu = model.cpu()

        # Dynamic quantization: quantizes weights of Linear and LSTM layers
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )

        # Determine output path
        if output_path:
            quant_path = Path(output_path)
        else:
            quant_path = Path(model_path).with_name(
                Path(model_path).stem + "_quantized.pth"
            )

        torch.save(quantized_model.state_dict(), quant_path)

        # Report size reduction
        original_size = Path(model_path).stat().st_size if Path(model_path).exists() else 0
        quant_size = quant_path.stat().st_size
        if original_size > 0:
            reduction = (1 - quant_size / original_size) * 100
            Log.info(
                f"Quantized model saved to {quant_path} "
                f"({reduction:.1f}% size reduction)"
            )
        else:
            Log.info(f"Quantized model saved to {quant_path}")

        return str(quant_path)

    except Exception as e:
        Log.error(f"Model quantization failed: {e}")
        return None
