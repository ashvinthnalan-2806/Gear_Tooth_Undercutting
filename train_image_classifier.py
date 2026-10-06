#!/usr/bin/env python
"""
Gear Tooth Undercutting Detector - CLI Training Script
Usage:
    python train_image_classifier.py
    python train_image_classifier.py --model efficientnet_b0 --epochs 20
"""

import argparse
import sys
from src.train import train_image_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Gear Tooth Undercutting Image Classifier")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to image dataset folder")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "efficientnet_b0"], help="Pretrained CNN backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/gear_cnn_model.pt", help="Path to save trained weights")
    return parser.parse_args()

def main():
    args = parse_args()
    result = train_image_model(
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
    if result["status"] != "success":
        sys.exit(1)

if __name__ == "__main__":
    main()
