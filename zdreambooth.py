# from huggingface_hub import snapshot_download

# local_dir = "./dog"
# snapshot_download(
#     "diffusers/dog-example",
#     local_dir=local_dir,
#     repo_type="dataset",
#     ignore_patterns=".gitattributes",
# )
import torch
# from diffusers import DiffusionPipeline


# model_id = "/home/sty/zyx/sd/diffusers/examples/dreambooth/sd1-4dreambooth"
# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# prompt = "A photo of sks dog in a bucket"
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# image.save("/home/sty/zyx/sd/diffusers/examples/dreambooth/dogs/dog-bucket.png")

# ------DDIMScheduler
from diffusers import DiffusionPipeline, DDIMScheduler

model_id = "/home/sty/zyx/sd/diffusers/examples/dreambooth/sd1-4dreambooth"

ddim = DDIMScheduler.from_pretrained(model_id , subfolder="scheduler")
pipe = DiffusionPipeline.from_pretrained(model_id, scheduler=ddim, torch_dtype=torch.float16)

pipe.to("cuda") 
generator = torch.Generator("cuda").manual_seed(0)
prompt = "A photo of Asian person" 
# 
image = pipe(prompt, guidance_rescale=0.7,generator=generator,num_inference_steps=150 ).images[0]
image.save("/home/sty/zyx/sd/diffusers/examples/dreambooth/dogs/dog-bucket-ddim.png")
