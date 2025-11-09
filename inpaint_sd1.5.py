# pip install diffusers==0.30.0 transformers accelerate safetensors nibabel pydicom pillow torch torchvision
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
import wandb
from masks import generate_random_blob_mask
image_root = "/home/biodeep/alin/datasets/RESC/val/good/img"
mask_root = "/home/biodeep/alin/datasets/camelyon16_256/valid/Ungood/label"
image_list = []
mask_list = []
list_of_prompts=["MRI of the brain showing normal gray and white matter. Inpaint the masked region with anatomically consistent brain tissue",
"brain MRI with symmetric hemispheres. Fill the missing region with plausible cortical and subcortical structures",
"MRI slice of the human brain. Replace masked lesion with realistic matching intensity and texture of surrounding tissue",
"MRI of the brain without abnormalities. The masked region should be replaced by realistic white matter and gray matter consistent with adjacent intensity values",
"MRI of the brain. Inpaint the masked area with a hyperintense mass lesion surrounded by mild edema, consistent with a glioma",
"Inpaint the masked region with a hyperintense mass lesion with irregular margins",
"Inpaint with realistic tumor appearance",
"Replace masked patch with confluent hyperintense lesion extending through white matter",
"Insert a realistic small white matter lesion in a normal T2 MRI brain slice"
]
list_of_prompts_histopathology = [
    "Reconstruct the missing tissue in the masked region. Generate context-aware cellular structures.",
    "Inpaint a plausible continuation of the histopathologic structure into the empty space.",
    "Fill the masked area with tissue architecture (glands, stroma) that is consistent with the surrounding tissue.",
    "Generate histologically plausible tissue to bridge the gap in the whole-slide image, ensuring morphological continuity."
    "Reconstruct the missing tissue in the masked region. Generate anomalous cellular structures.",
    "Inpaint an anomalous continuation of the histopathologic structure into the empty space.",
    "Fill the masked area with tissue architecture (glands, stroma) that is not consistent with the surrounding tissue.",
    "Generate histologically not plausible tissue to bridge the gap in the whole-slide image, ensuring they are not morphological continuous."
]

list_of_prompts_chest = [
    "Generate a single, solid pulmonary nodule (approx. 1 cm) within the masked area.",
    "Inpaint a subtle, sub-solid nodule or ground-glass opacity (GGO) in the masked lung field.",
    "Create multiple small, calcified nodules consistent with old granulomatous disease.",
    "Inpaint a dense airspace consolidation with air bronchograms, consistent with lobar pneumonia, in the masked region.",
    "Generate patchy, bilateral infiltrates or opacities within the masked zones, suggestive of atypical pneumonia.",
    "Generate a pneumothorax in the masked apex, showing a visible visceral pleural line and an absence of lung markings.",
    "Create a pleural effusion in the masked area, characterized by blunting of the costophrenic angle and a meniscus sign."
    "Inpaint fibrotic scarring or reticular opacities in the masked lung base, consistent with interstitial lung disease."
]

list_of_prompts_liver = [
    "Generate a solid, hypervascular nodule in the masked region. Show bright arterial phase hyperenhancement and portal venous phase washout.",
     "Inpaint a hypodense lesion that shows peripheral, nodular enhancement in the arterial phase and gradually fills in centripetally (from outside-in) on the portal venous phase.",
    "Generate a hypodense, ring-enhancing lesion in the masked area, most visible in the portal venous phase.",
    "Inpaint a simple hepatic cyst. The lesion must be sharply defined, non-enhancing, and hypodense (near-water density, 0-20 HU).",
    "Inpaint the masked liver parenchyma to be hypodense (darker) than the spleen, consistent with hepatic steatosis.",
    "Inpaint black blobs"
]

list_of_prompts_retina = [
    "Generate soft drusen. Inpaint large, confluent, hyporeflective mounds underneath the RPE layer.",

"Inpaint a Pigment Epithelial Detachment (PED). Create a large, dome-shaped elevation of the RPE layer in the masked area.",

"Generate geographic atrophy. Inpaint a region of outer retinal thinning and RPE loss, causing increased signal transmission (hyper-transmission) into the choroid."

]
# wandb.init(project="Medical-inpainting", name="multiple_prompts")
 
for image_name in os.listdir(image_root)[:10]:
    image_path = os.path.join(image_root, image_name)
    image_list.append(image_path)
    mask_list.append(os.path.join(mask_root, image_name))
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    safety_checker=None, torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")
for i, image_path in enumerate(image_list):

    img_pil = Image.open(image_path).convert("RGB").resize((512,512))

    mk = generate_random_blob_mask(np.array(img_pil), num_blobs=6, blur_kernel_size=81)

    mask_pil = Image.fromarray(((mk!=0)*255).astype(np.uint8)).convert("L").resize((512,512))

    # mask_pil.save("masked_img.png")
    # exit()
    for prompt in list_of_prompts_retina:
        out = pipe(
            prompt=prompt,
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=50,
            guidance_scale=10.0, strength=1,
            generator=torch.Generator().manual_seed(42)
        ).images[0]

        result = np.concatenate((np.array(img_pil), np.array(mask_pil.convert("RGB")), np.array(out)), axis=1)
        os.makedirs(f"./results_sd1.5_retina_2/{prompt[:10]}/", exist_ok=True)
        result_image = Image.fromarray(result.astype(np.uint8))
        result_image.save(f"./results_sd1.5_retina_2/{prompt[:10]}/{i}.png")
        # wandb.log({prompt: wandb.Image(result.astype(np.uint8), caption=prompt)})
