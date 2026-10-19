import os
import torch
import gc
import cv2
import numpy as np
import imageio

def upscale_video_frames(input_path, output_path=None, scale=2, fps=10):
    """
    Upscales video frames while removing noise and preserving sharp anime character edges.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input video file not found at: {input_path}")
        return None

    print(f"\n[INFO] Starting Crisp Upscaling & Edge-Preserving Denoising (Scale: {scale}x)...")
    
    # 1. Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Prepare paths
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    base_dir = os.path.abspath(os.path.join(current_dir, "..")) if "src" in current_dir else current_dir
    
    if output_path is None:
        output_dir = os.path.join(base_dir, "outputs", "upscaled")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_crisp_upscaled.mp4")

    # 3. Read frames from video
    reader = imageio.get_reader(input_path)
    processed_frames = []

    print("[INFO] Processing frames with edge preservation...")
    for idx, frame in enumerate(reader):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Step A: Light denoising so it doesn't blur features (h values reduced)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(frame_bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
        
        # Step B: High-quality scaling (Cubic/Lanczos)
        height, width = denoised_bgr.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        scaled_bgr = cv2.resize(denoised_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Step C: Unsharp Masking (Sharpening filter to bring back character lines and sharp details)
        gaussian_blur = cv2.GaussianBlur(scaled_bgr, (0, 0), 2.0)
        sharpened_bgr = cv2.addWeighted(scaled_bgr, 1.3, gaussian_blur, -0.3, 0)
        
        # Convert back to RGB
        output_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
        processed_frames.append(output_rgb)
        
        print(f"[PROGRESS] Crisp-processed frame #{idx + 1}")

    # 4. Save processed video
    imageio.mimsave(output_path, processed_frames, fps=fps, codec='libx264', quality=8)
    print(f"[SUCCESS] Crisp upscaled video saved to: {output_path}")

    # Clean memory
    gc.collect()
    torch.cuda.empty_cache()

    return output_path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    base_dir = os.path.abspath(os.path.join(current_dir, "..")) if "src" in current_dir else current_dir
    
    input_video = os.path.join(base_dir, "outputs", "final", "final_story_sequence.mp4")
    
    upscale_video_frames(input_path=input_video, scale=2, fps=10)