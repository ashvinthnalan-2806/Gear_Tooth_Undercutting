import os
from typing import Dict, Any, Union
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.preprocessing import get_val_transforms
from src.model import build_gear_cnn_model
from src.augmentation_utils import detect_and_crop_photo

CLASS_DISPLAY_NAMES = {
    0: "No Undercut",
    1: "Undercut",
    "no_undercut": "No Undercut",
    "undercut": "Undercut"
}

class GearImagePredictor:
    """Inference engine for Gear Tooth Undercutting Detector."""

    def __init__(self, model_path: str = "models/gear_cnn_model.pt", model_name: str = "mobilenet_v2"):
        self.model_path = model_path
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_val_transforms()
        self.model = None
        self.metadata = {}
        self._load_model_if_exists()

    def _load_model_if_exists(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model_name = checkpoint.get("model_name", self.model_name)
                self.model = build_gear_cnn_model(
                    model_name=self.model_name,
                    num_classes=2,
                    pretrained=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                self.metadata = checkpoint.get("best_metrics", {})
                return True
            except Exception as e:
                print(f"Warning: Failed to load checkpoint from {self.model_path}: {e}")
                self.model = None
                return False
        return False

    def is_model_trained(self) -> bool:
        """Returns True if a trained checkpoint is available on disk."""
        if self.model is None and os.path.exists(self.model_path):
            self._load_model_if_exists()
        return os.path.exists(self.model_path) and self.model is not None

    def predict(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Runs image inference on an uploaded gear photograph.
        Strictly honest: will NOT fabricate predictions if no trained model exists.
        """
        if not self.is_model_trained():
            # Refresh check in case model was just trained
            if not self._load_model_if_exists():
                return {
                    "status": "model_not_trained",
                    "has_trained_model": False,
                    "prediction": None,
                    "confidence": None,
                    "message": (
                        "No trained gear image model found. The model cannot produce real predictions "
                        "until labeled gear photographs are placed in the dataset directories and trained."
                    ),
                    "instructions": {
                        "undercut_dir": os.path.abspath("dataset/undercut"),
                        "no_undercut_dir": os.path.abspath("dataset/no_undercut"),
                        "train_command": "python train_image_classifier.py"
                    }
                }

        # Open image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Auto-crop black letterboxing or phone screenshot margins to focus on gear
        image = detect_and_crop_photo(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        p_no_undercut = float(probabilities[0])
        p_undercut = float(probabilities[1])

        pred_class_idx = 1 if p_undercut >= p_no_undercut else 0
        predicted_label = CLASS_DISPLAY_NAMES[pred_class_idx]
        confidence = float(max(p_undercut, p_no_undercut))

        return {
            "status": "success",
            "has_trained_model": True,
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Undercut": round(p_undercut * 100, 2),
                "No Undercut": round(p_no_undercut * 100, 2)
            },
            "model_metadata": self.metadata
        }
