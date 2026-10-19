import torch
import gc
import os
import traceback
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils.logging import disable_progress_bar
import imageio

disable_progress_bar()

def initialize_pipeline():
    try:
        print("[INFO] Initializing pipeline with memory guards...")
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        base_dir = os.path.abspath(os.path.join(current_dir, "..")) if "src" in current_dir else current_dir
        fused_model_dir = os.path.join(base_dir, "models", "base", "animagine_fused_full")

        # تنظيف عميق للذاكرة قبل التحميل
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("[INFO] Loading VAE in float32...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float32
        )

        print("[INFO] Loading Motion Adapter...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-sdxl-beta", 
            torch_dtype=torch.float16
        )

        print("[INFO] Loading Fused Pipeline (this uses heavy VRAM)...")
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            fused_model_dir,
            motion_adapter=adapter,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        print("[INFO] Applying strict VRAM offloading...")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, 
            use_karras_sigmas=True
        )
        
        # تفعيل الـ Offload بحذر لمنع الانهيار الصامت
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()

        # تنظيف إضافي بعد التحميل
        gc.collect()
        torch.cuda.empty_cache()

        return pipe, base_dir
    except Exception as e:
        print("\n[ERROR] Critical error during initialization:")
        traceback.print_exc()
        exit(1)

def generate_shot(pipe, base_dir, shot_index=1, num_frames=16, custom_prompt="", seed=42):
    try:
        print(f"\n[INFO] Generating Story Shot #{shot_index}...")
        
        gc.collect()
        torch.cuda.empty_cache()

        trigger_word = "mycustomboy" 
        base_quality = "masterpiece, best quality, sharp focus, clean 2D anime style, cowboy shot, centered character, consistent facial features, stable structure"
        prompt = f"{base_quality}, {trigger_word}, {custom_prompt}"
        
        negative_prompt = "wide shot, long shot, distant figure, full body wide shot, small character in frame, close-up, portrait, upper body only, lowres, bad anatomy, deformed, distorted, stretched, morphing face, multiple faces, duplicate facial features, shifting eyes, blurry, noise, grainy, pixelated, jpeg artifacts, grid lines, tiling artifacts, smoke, haze, fog, atmospheric dust, airborne particles, blurry background, yellow hair, blonde, long hair, blue eyes, female, girl, washed out, low contrast, grayish, faded colors"

        generator = torch.Generator("cuda").manual_seed(seed + shot_index)
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=768,
                width=768,
                guidance_scale=9.0,
                num_inference_steps=40,
                generator=generator
            )

        frames = output.frames[0]
        
        output_dir = os.path.join(base_dir, "outputs", "shots")
        os.makedirs(output_dir, exist_ok=True)
        output_gif_path = os.path.join(output_dir, f"shot_{shot_index:02d}.gif")
        
        from diffusers.utils import export_to_gif
        export_to_gif(frames, output_gif_path, fps=8)
        print(f"[SUCCESS] Shot #{shot_index} saved to: {output_gif_path}")
        return output_gif_path
    except Exception as e:
        print(f"\n[ERROR] Exception during generation of shot #{shot_index}:")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    storyboard_script = [
        "1boy, short black hair, red eyes, wearing green hooded adventurer cloak, standing firmly at the entrance of a dark cave, detailed dark stone dungeon background, stone floor, glowing wall torches, atmospheric lighting",
        "1boy, short black hair, red eyes, wearing green hooded adventurer cloak, pulling out a magical glowing torch to light the way, detailed dark stone dungeon background, stone floor, glowing wall torches, atmospheric lighting",
        "1boy, short black hair, red eyes, wearing green hooded adventurer cloak, looking around cautiously at ancient stone walls, detailed dark stone dungeon background, stone floor, glowing wall torches, atmospheric lighting",
        "1boy, short black hair, red eyes, wearing green hooded adventurer cloak, walking deeper into the mysterious corridor, detailed dark stone dungeon background, stone floor, glowing wall torches, atmospheric lighting",
        "1boy, short black hair, red eyes, wearing green hooded adventurer cloak, turning back with a serious heroic expression, detailed dark stone dungeon background, stone floor, glowing wall torches, atmospheric lighting"
    ]
    
    pipe, base_dir = initialize_pipeline()
    
    frames_per_shot = 16
    
    for i, story_action in enumerate(storyboard_script, start=1):
        generate_shot(
            pipe=pipe,
            base_dir=base_dir,
            shot_index=i, 
            num_frames=frames_per_shot, 
            custom_prompt=story_action, 
            seed=42
        )
        
    print("\n[INFO] All shots generated successfully!")