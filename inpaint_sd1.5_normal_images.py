# pip install diffusers==0.30.0 transformers accelerate safetensors nibabel pydicom pillow torch torchvision
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import os
import wandb
from masks import generate_random_blob_mask
image_root = "/home/fl488644/medical_anomalies/normal_images"
image_list = []
mask_list = []
list_of_prompts=["dog", "giraffe", "airplane"
]
for image_name in os.listdir(image_root)[:10]:
    image_path = os.path.join(image_root, image_name)
    image_list.append(image_path)
pipe = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
for i, image_path in enumerate(image_list):

    img_pil = Image.open(image_path).convert("RGB").resize((768, 768))

    mk = generate_random_blob_mask(np.array(img_pil), num_blobs=6, blur_kernel_size=81)

    mask_pil = Image.fromarray(((mk!=0)*255).astype(np.uint8)).convert("L").resize((768, 768))

    # mask_pil.save("masked_img.png")
    # exit()
    for prompt in list_of_prompts:
        out = pipe(
            prompt=prompt,
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=50,
            guidance_scale=10.0, strength=1, height=768, width=768,
            generator=torch.Generator().manual_seed(42)
        ).images[0]

        result = np.concatenate((np.array(img_pil), np.array(mask_pil.convert("RGB")), np.array(out)), axis=1)
        os.makedirs(f"./results_sd1.5_normal_images/{prompt[:10]}/", exist_ok=True)
        result_image = Image.fromarray(result.astype(np.uint8))
        result_image.save(f"./results_sd1.5_normal_images/{prompt[:10]}/{i}.png")
        # wandb.log({prompt: wandb.Image(result.astype(np.uint8), caption=prompt)})
