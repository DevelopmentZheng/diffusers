

import torch
# from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained("/mnt/mydrive/datas/sd/stable-diffusion-v1-4")
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse1.png")

# from accelerate import PartialState
# from diffusers import DiffusionPipeline

# pipeline = DiffusionPipeline.from_pretrained("/mnt/mydrive/datas/sd/stable-diffusion-v1-4", torch_dtype=torch.float16)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)

# with distributed_state.split_between_processes(["a dog", "a cat","a panda","a bird","a tiger","a mouse","a pig","a fish"]) as prompt:
#     result = pipeline(prompt).images[0]
#     result.save(f"result_{distributed_state.process_index}.png")


from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("/mnt/mydrive/datas/model/sd/stable-diffusion-v1-4").to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# prompt = "a red cat playing with a ball"

# generator = torch.Generator(device="cuda").manual_seed(33)

# image1 = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
# image1.save("z1.png")

from compel import Compel

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt = "a Asian++ person"
prompt_embeds = compel_proc(prompt)
generator = torch.Generator(device="cuda").manual_seed(33)

image2 = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image2.save("z2.png")





