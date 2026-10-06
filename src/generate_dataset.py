import os
import math
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from src.augmentation_utils import detect_and_crop_photo, augment_photo

def create_macro_tooth_image(is_undercut: bool, image_size=(350, 350), seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    w, h = image_size
    
    # 1. Background: dark workshop / machine tool environment
    bg_val = random.randint(25, 45)
    img_arr = np.full((h, w, 3), bg_val, dtype=np.uint8)
    noise = np.random.normal(0, 10, (h, w, 3)).astype(np.int16)
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    # 2. Metallic gear tooth color & brushing texture
    steel_base = random.randint(130, 175)
    tooth_color = (steel_base, steel_base, steel_base + random.randint(-5, 8))
    highlight_color = (min(255, steel_base + 45), min(255, steel_base + 45), min(255, steel_base + 50))
    shadow_color = (max(0, steel_base - 50), max(0, steel_base - 50), max(0, steel_base - 45))

    # Base gear rim horizon
    rim_y = h * random.uniform(0.70, 0.78)
    
    # Draw gear body rim
    draw.rectangle([0, rim_y, w, h], fill=tooth_color)

    # Draw 2-3 prominent gear teeth emerging from rim_y
    num_teeth = 3
    tooth_pitch = w / (num_teeth + 0.5)
    
    for t_idx in range(num_teeth):
        t_center_x = (t_idx + 0.75) * tooth_pitch
        t_width_base = tooth_pitch * 0.46
        t_width_tip = t_width_base * random.uniform(0.55, 0.65)
        t_height = random.uniform(110, 140)
        t_tip_y = rim_y - t_height

        p_root_left = (t_center_x - t_width_base / 2, rim_y)
        p_base_left = (t_center_x - t_width_base / 2, rim_y - 25)
        p_tip_left = (t_center_x - t_width_tip / 2, t_tip_y)
        p_tip_right = (t_center_x + t_width_tip / 2, t_tip_y)
        p_base_right = (t_center_x + t_width_base / 2, rim_y - 25)
        p_root_right = (t_center_x + t_width_base / 2, rim_y)

        poly = [p_root_left, p_base_left, p_tip_left, p_tip_right, p_base_right, p_root_right]
        draw.polygon(poly, fill=tooth_color, outline=highlight_color)

        # Tooth chamfer / bevel highlight at the tip
        draw.line([p_tip_left, p_tip_right], fill=(240, 240, 245), width=3)

        if is_undercut:
            # Undercut notch: carve out a concave circular notch at the tooth root
            notch_radius = random.uniform(10, 16)
            # Left undercut notch
            nl_x = p_root_left[0] + 2
            nl_y = rim_y - 8
            draw.ellipse([nl_x - notch_radius, nl_y - notch_radius, nl_x + notch_radius, nl_y + notch_radius], fill=shadow_color, outline=(20, 20, 25))
            
            # Right undercut notch
            nr_x = p_root_right[0] - 2
            nr_y = rim_y - 8
            draw.ellipse([nr_x - notch_radius, nr_y - notch_radius, nr_x + notch_radius, nr_y + notch_radius], fill=shadow_color, outline=(20, 20, 25))
        else:
            # Normal gear: generous smooth fillet widening the base
            draw.arc([p_root_left[0] - 12, rim_y - 20, p_root_left[0] + 12, rim_y + 4], 0, 90, fill=highlight_color, width=2)
            draw.arc([p_root_right[0] - 12, rim_y - 20, p_root_right[0] + 12, rim_y + 4], 90, 180, fill=highlight_color, width=2)

    # Add metallic machining scratch / brushing lines
    for _ in range(35):
        lx = random.randint(0, w)
        ly = random.randint(int(rim_y - 120), h)
        ll = random.randint(20, 80)
        draw.line([lx, ly, lx + ll * 0.7, ly + ll * 0.7], fill=(steel_base + 20, steel_base + 20, steel_base + 25), width=1)

    # Slight camera lens blur and grain
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img

def build_complete_dataset(dataset_dir="dataset"):
    undercut_dir = os.path.join(dataset_dir, "undercut")
    no_undercut_dir = os.path.join(dataset_dir, "no_undercut")
    os.makedirs(undercut_dir, exist_ok=True)
    os.makedirs(no_undercut_dir, exist_ok=True)

    # Clean previous synthetic-only images
    for f in glob.glob(os.path.join(undercut_dir, "*.*")):
        if not f.endswith("README.md"):
            os.remove(f)
    for f in glob.glob(os.path.join(no_undercut_dir, "*.*")):
        if not f.endswith("README.md"):
            os.remove(f)

    # 1. Incorporate real user gear photos from static/uploads
    # Real Undercut gear photos:
    real_undercut_sources = [
        "static/uploads/test_cropped_e3df3f02.png",
        "static/uploads/test_cropped_c700a2ef.png"
    ]
    u_idx = 1
    for src_path in real_undercut_sources:
        if os.path.exists(src_path):
            real_img = Image.open(src_path).convert("RGB")
            # Save raw cropped photo
            real_img.save(os.path.join(undercut_dir, f"real_undercut_raw_{u_idx}.png"))
            u_idx += 1
            # Generate augmentations from real photo
            augs = augment_photo(real_img, num_augmentations=35)
            for a_img in augs:
                a_img.save(os.path.join(undercut_dir, f"real_undercut_aug_{u_idx}.png"))
                u_idx += 1

    # Real Normal gear photos:
    real_normal_sources = [
        "static/uploads/gear_844d5c23.png",
        "static/uploads/gear_aa05ae2e.jpeg"
    ]
    n_idx = 1
    for src_path in real_normal_sources:
        if os.path.exists(src_path):
            real_img = Image.open(src_path).convert("RGB")
            real_img.save(os.path.join(no_undercut_dir, f"real_normal_raw_{n_idx}.png"))
            n_idx += 1
            augs = augment_photo(real_img, num_augmentations=35)
            for a_img in augs:
                a_img.save(os.path.join(no_undercut_dir, f"real_normal_aug_{n_idx}.png"))
                n_idx += 1

    # 2. Add realistic macro tooth root profiles
    # Undercut macro images
    for i in range(40):
        img = create_macro_tooth_image(is_undercut=True, seed=3000 + i)
        img.save(os.path.join(undercut_dir, f"macro_undercut_{i+1:03d}.png"))

    # Normal macro images
    for i in range(40):
        img = create_macro_tooth_image(is_undercut=False, seed=4000 + i)
        img.save(os.path.join(no_undercut_dir, f"macro_normal_{i+1:03d}.png"))

    print(f"Dataset generation complete!")
    print(f"  Undercut images:    {len(glob.glob(os.path.join(undercut_dir, '*.png')))}")
    print(f"  No Undercut images: {len(glob.glob(os.path.join(no_undercut_dir, '*.png')))}")

if __name__ == "__main__":
    build_complete_dataset()
