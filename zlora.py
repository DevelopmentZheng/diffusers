# accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=/mnt/mydrive/datas/model/sd/stable-diffusion-v1-4 \
#   --dataset_name=/home/sty/zyx/sd/diffusers/pokemon-blip-captions  --caption_column="text" \
#   --resolution=512 --random_flip \
#   --train_batch_size=1 \
#   --num_train_epochs=100 --checkpointing_steps=5000 \
#   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --seed=42 \
#   --output_dir="sd-pokemon-model-lora" \
#   --validation_prompt="cute dragon creature" --report_to="wandb"

from diffusers import StableDiffusionPipeline
import torch

model_path = "/home/sty/zyx/sd/diffusers/examples/text_to_image/sd-pokemon-model-lora"
#pipe = StableDiffusionPipeline.from_pretrained("/mnt/mydrive/datas/sd/stable-diffusion-v1-4 ", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("/mnt/mydrive/datas/sd/stable-diffusion-v1-4").to("cuda")
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
