# Anime Generator Project: Scalable SDXL & AnimateDiff Production Pipeline

An advanced, end-to-end Generative AI animation framework engineered to produce temporally consistent, high-fidelity character-driven anime video sequences using custom transfer learning workflows.

---

## Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture & Pipeline Layers](#-system-architecture--pipeline-layers)
- [Technical Challenges & Engineering Solutions](#-technical-challenges--engineering-solutions)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Repository Structure](#-repository-structure)
- [Technical Whitepaper Report](#-technical-whitepaper-report)
- [Author](#-author)

---

## Project Overview
Generating coherent, multi-second animation sequences using latent diffusion models introduces a core architectural challenge: **balancing expressive visual fidelity with strict temporal and character consistency**. 

The **Anime Generator Project** establishes a modular, scriptable pipeline that bridges high-level artistic descriptions with low-level tensor optimizations. By combining fine-tuned character LoRAs with motion injection models, the system enables the synthesis of fluid, artifact-free anime video frames while maintaining strict structural identity across camera angles.

---

## System Architecture & Pipeline Layers
The pipeline is structured into four distinct modular layers to isolate component execution:
1. **Data Acquisition & Preprocessing:** Curation of character assets, tagging via Danbooru taxonomies, and dataset preparation for parameter-efficient fine-tuning via `sd-scripts`.
2. **Model Fusion & Weight Integration:** Mathematical calculation of weight deltas ($\Delta W$) to fuse character LoRA parameters directly into base model state dicts offline, eliminating runtime parsing overhead.
3. **Inference & Motion Engine:** Execution of AnimateDiff temporal attention modules combined with sequential CPU offloading and precision management.
4. **Post-Processing & Export:** Frame aggregation, color fidelity preservation, and export to animated formats (GIF/MP4).

---

## Technical Challenges & Engineering Solutions

| Challenge Encountered | Root Cause | Engineered Solution |
| :--- | :--- | :--- |
| **PEFT / Text Encoder Parsing Mismatch** | Dual text-encoder keys in `sd-scripts` trained LoRAs trigger `IndexError` during dynamic runtime PEFT parsing. | Developed a **Custom Offline Manual Fusion Routine** that calculates weight deltas ($\Delta W = (W_{\text{up}} \cdot W_{\text{down}}) \times \text{scale}$) and bakes them directly into UNet and text encoder weights. |
| **Color Washout & Latent Degradation** | VAE decoding mismatches when processing compressed latent spaces under strict precision casts. | Integrated custom fixed VAE pipelines (`madebyollin/sdxl-vae-fp16-fix`) combined with PyTorch AMP context alignment to secure high dynamic color contrast. |
| **Resource & Tensor Bottlenecks** | Joint temporal attention across multiple frames exceeding consumer hardware thresholds. | Implemented sequential module offloading and VAE slicing/tiling routines to guarantee stable execution. |

---

## Tech Stack & Dependencies
* **Core Base Model:** Animagine XL 3.1 (Stable Diffusion XL variant)
* **Motion Adapter:** AnimateDiff SDXL Beta (`animatediff-motion-adapter-sdxl-beta`)
* **Fine-Tuning Backend:** `sd-scripts` (Kohya_ss)
* **Frameworks:** PyTorch, Hugging Face `diffusers`, `transformers`
* **Documentation & Whitepaper:** LaTeX (Overleaf)

---

## Repository Structure
```text
cartoon-generator/
│
├── src/                    # Core pipeline execution scripts
│   ├── generator.py        # Base image and latent generation logic
│   ├── animator.py         # Temporal motion integration & frame rendering
│   └── upscale.py          # Resolution and post-processing modules
├── notebooks/              # Experimental Jupyter notebooks for prototyping
│   ├── dataset_generation.ipynb
│   ├── lora_testing.ipynb
│   └── video_pipeline.ipynb
├── report/                 # Full academic whitepaper (LaTeX source & compiled PDF)
│   ├── main.tex
│   └── Anime_Generator_Project_Report.pdf
├── .gitignore              # Environment, datasets, and weights exclusion rules
├── requirements.txt        # Python package dependencies
└── README.md               # Comprehensive project documentation


## Technical Whitepaper Report
For a deep dive into the mathematical formulations, literature review, implementation plans, and architectural decisions, read the official project whitepaper located in the [`report/`](./report/) directory:

* **Compiled PDF Report:** [`report/Anime_Generator_Project_Report.pdf`](./report/Anime_Generator_Project_Report.pdf)

---

## Author
**Bandr Albugami**  
*Operations & AI Systems Engineering*