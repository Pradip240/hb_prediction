import json

import numpy as np
import torch

from common import config
from hb_model.features import extract_features
from hb_model.model import HbMLP
from hr_model.dataset import SpectralDataset
from hr_model.model import HRSpectralNet

_HR_MODEL: HRSpectralNet | None = None
_HB_MODEL: HbMLP | None = None


def predict_hr(
    model_dir: str,
    sample_path: str,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Predict heart rate from a prepared signal segment.

    The trained HRSpectralNet is loaded lazily on the first call and
    cached for subsequent predictions. The input signal is prepared using
    the same preprocessing used during training.

    Args:
        model_dir: Directory containing the trained model weights and model configuration.
        sample_path: Path to the prepared signal archive.
        device: Device used for inference.

    Returns:
        Tuple containing:
            - Predicted heart rate from the soft-argmax in BPM.
            - Confidence based on probability mass around the predicted HR.
            - HR-frequency bins in BPM.
            - Spectral probability distribution.
    """
    global _HR_MODEL

    if _HR_MODEL is None:
        config_path = f"{model_dir}/model_config.json"
        model_path = f"{model_dir}/best_model.pt"

        # Load the model configuration saved during training.
        with open(config_path, encoding="utf-8") as file:
            model_config = json.load(file)

        # Recreate the model using the same configuration used during training.
        _HR_MODEL = HRSpectralNet( # type: ignore
            n_channels=model_config["n_channels"],
            fps=model_config["fps"],
            nfft=model_config["nfft"],
            hr_min=model_config["hr_min"],
            hr_max=model_config["hr_max"],
        )

        # Load the trained model weights.
        state_dict = torch.load(model_path, map_location=device)
        _HR_MODEL.load_state_dict(state_dict)

        # Move the model to the requested device and switch to evaluation mode.
        _HR_MODEL.to(device)
        _HR_MODEL.eval()

    # Load the singnal
    signal = SpectralDataset.prepare_signal(sample_path)

    # Add the batch dimension expected by HRSpectralNet.
    signal = signal.unsqueeze(0).to(device)

    # Run inference without calculating gradients.
    with torch.no_grad():
        predicted_bpm, _, probability = _HR_MODEL(signal)

    # Continuous BPM estimate produced by the soft-argmax.
    predicted_bpm = float(predicted_bpm[0].item())

    # Extract the probability distribution for the first sample.
    probability = probability[0]

    # Calculate confidence as the probability mass within +/- 6 BPM of the predicted heart rate.
    confidence_mask = torch.abs(_HR_MODEL.band_bpm - predicted_bpm) <= 6.0 # type: ignore
    confidence = float(probability[confidence_mask].sum().item())

    # Move spectral information to CPU for plotting or further processing.
    bpm_bins = _HR_MODEL.band_bpm.detach().cpu().numpy() # type: ignore
    probability = probability.detach().cpu().numpy()
    return predicted_bpm, confidence, bpm_bins, probability # type: ignore


def predict_hb(model_dir: str, sample_path: str, device: torch.device) -> float:
    """
    Predict hemoglobin from a prepared signal segment.

    The trained HbMLP is loaded lazily on the first call and cached for
    subsequent predictions. The input signal is converted into the same
    engineered RGB and pixel-count features used during training.

    Args:
        model_dir: Directory containing the trained model weights and
            model configuration.
        sample_path: Path to the prepared signal archive.
        device: Device used for inference.

    Returns:
        Predicted hemoglobin concentration in g/dL.
    """
    global _HB_MODEL

    if _HB_MODEL is None:
        config_path = f"{model_dir}/model_config.json"
        model_path = f"{model_dir}/best_model.pt"

        # Load the model configuration saved during training.
        with open(config_path, encoding="utf-8") as file:
            model_config = json.load(file)

        # Recreate the model using the same configuration used during training.
        _HB_MODEL = HbMLP( # type: ignore
            n_in=model_config["n_features"],
            width=model_config["hidden_width"],
            dropout=model_config["dropout"],
        )

        # Load the trained model weights.
        state_dict = torch.load(model_path, map_location=device)
        _HB_MODEL.load_state_dict(state_dict)

        # Move the model to the requested device and switch to evaluation mode.
        _HB_MODEL.to(device)
        _HB_MODEL.eval()

    # Load the prepared signal archive.
    data = np.load(sample_path)

    signals = np.asarray(data["signals"], dtype=np.float32)
    pixel_counts = np.asarray(data["pixel_counts"], dtype=np.float32)
    fps = float(data["fps"])

    if signals.ndim != 3:
        raise ValueError(f"Expected signals with shape (T, R, 3), got {signals.shape} in {sample_path}.")
    if signals.shape[-1] != 3:
        raise ValueError(f"Expected RGB signals with 3 channels, got {signals.shape[-1]} in {sample_path}.")
    if pixel_counts.ndim != 2:
        raise ValueError(f"Expected pixel_counts with shape (T, R), got {pixel_counts.shape} in {sample_path}.")
    if signals.shape[:2] != pixel_counts.shape:
        raise ValueError(f"signals and pixel_counts shapes do not match: {signals.shape} vs {pixel_counts.shape}.")

    # Convert the archive format:
    #   signals:      (T, R, 3)
    #   pixel_counts: (T, R)
    #
    # into the format expected by extract_features():
    #   signals:      (R, T, 3)
    #   pixel_counts: (R, T)
    features = extract_features(
        signals=np.transpose(signals, (1, 0, 2)),
        pixel_counts=np.transpose(pixel_counts, (1, 0)),
        fps=fps,
        region_order=list(config.REGION_ORDER)
    )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Non-finite Hb features generated for {sample_path}.")

    # Convert the feature vector into a batch of one sample.
    feature_tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device) # type: ignore

    # Run inference without calculating gradients.
    with torch.no_grad():
        predicted_hb = _HB_MODEL(feature_tensor)

    # Extract the scalar hemoglobin prediction.
    return float(predicted_hb[0].item())
