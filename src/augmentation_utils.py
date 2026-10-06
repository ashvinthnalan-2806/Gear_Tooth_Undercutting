import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

def detect_and_crop_photo(img):
    """
    Detects if an image has letterboxing or status/navigation bars (like phone screenshots)
    and crops to the active photograph region.
    """
    arr = np.array(img.convert('RGB'))
    H, W, _ = arr.shape
    row_means = arr.mean(axis=(1, 2))
    
    # Active photo rows have significant average brightness
    is_content = row_means > 18
    
    # Find contiguous blocks of content
    blocks = []
    current_start = None
    for i, val in enumerate(is_content):
        if val and current_start is None:
            current_start = i
        elif not val and current_start is not None:
            blocks.append((current_start, i))
            current_start = None
    if current_start is not None:
        blocks.append((current_start, H))
        
    if not blocks:
        return img
        
    # Pick largest contiguous block
    best_block = max(blocks, key=lambda b: b[1] - b[0])
    y1, y2 = best_block
    
    # Only crop if substantial padding exists (> 15% height difference)
    if (y2 - y1) < 0.85 * H and (y2 - y1) > 0.2 * H:
        return img.crop((0, y1, W, y2))
    return img

def augment_photo(img, num_augmentations=30):
    augmented_list = []
    w, h = img.size
    
    for i in range(num_augmentations):
        # 1. Random crop (focusing on teeth / active regions)
        crop_scale = random.uniform(0.65, 0.95)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        if cw < w and ch < h:
            cx = random.randint(0, w - cw)
            cy = random.randint(0, h - ch)
            aug = img.crop((cx, cy, cx + cw, cy + ch))
        else:
            aug = img.copy()
            
        # 2. Resize to 350x350
        aug = aug.resize((350, 350), Image.Resampling.BILINEAR)
        
        # 3. Flips & rotations
        if random.random() > 0.5:
            aug = aug.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rot_angle = random.uniform(-25, 25)
        aug = aug.rotate(rot_angle, resample=Image.Resampling.BILINEAR)
        
        # 4. Color & contrast jitter
        enh_b = ImageEnhance.Brightness(aug)
        aug = enh_b.enhance(random.uniform(0.8, 1.25))
        enh_c = ImageEnhance.Contrast(aug)
        aug = enh_c.enhance(random.uniform(0.85, 1.3))
        
        augmented_list.append(aug)
        
    return augmented_list
