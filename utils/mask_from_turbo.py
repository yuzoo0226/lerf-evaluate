import cv2
import numpy as np
import argparse
import os
import torch
from matplotlib import colormaps
from tqdm import tqdm

# Turboカラーマップ: (256, 3) の RGB 配列（[0, 1] 範囲）
turbo_colors = np.array(colormaps["turbo"].colors)  # shape: (256, 3)
turbo_colors_uint8 = (turbo_colors * 255).astype(np.uint8)  # shape: (256, 3)

def turbo_colormap_inverse(image_rgb):
    """
    Turbo colormap を使用して RGB 画像からスカラー値 (0〜1) を復元。
    Nerfstudioの apply_float_colormap() に対応した逆処理。
    """
    h, w, _ = image_rgb.shape
    reshaped = image_rgb.reshape(-1, 3)

    # 最近傍探索 (RGB 空間内で)
    diffs = reshaped[:, None, :] - turbo_colors_uint8[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    indices = np.argmin(dists, axis=1)

    # スカラー値（0〜1）に変換
    scalar = indices.astype(np.float32) / 255.0
    return scalar.reshape(h, w)

def generate_threshold_mask_for_file(filepath, threshold, output_path):
    img_bgr = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scalar_img = turbo_colormap_inverse(img_rgb)
    mask = (scalar_img >= threshold).astype(np.uint8) * 255
    cv2.imwrite(output_path, mask)

def process_folder(input_dir, threshold):
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            base, _ = os.path.splitext(filename)
            output_dir = os.path.join(input_dir, "masks")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base}.png")
            generate_threshold_mask_for_file(input_path, threshold, output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate masks from Turbo-colored PNGs (Nerfstudio compatible)")
    parser.add_argument("--input_dir", required=True, help="Path to input directory containing .png files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold value in range [0, 1]")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist.")
        return
    
    process_folder(args.input_dir, args.threshold)
    print("✅ Done.")

if __name__ == "__main__":
    main()
