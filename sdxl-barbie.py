from diffusers import AutoPipelineForText2Image
import torch
import PIL
import re
import time

#https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call
myPrompt = "A photo of a pink cat in the style of TOK" 
myNegativePrompt = "" #"cartoon, drawing, ugly, deformed, noisy, blurry, low contrast, text, BadDream, 3d, cgi, render, fake, anime, open mouth, big forehead, long neck"
numInferenceSteps = 50 #default 50
guidanceScale = 7.5 

numImages = 4
outHeight=1024 #default 1024
outWidth=1024 #default 1024

#default sdxl
modelPath = "stabilityai/stable-diffusion-xl-base-1.0"
loraModelPath = "fofr/sdxl-barbie" #https://huggingface.co/fofr/sdxl-barbie

cleanedFileNameString = re.sub(r'\W+', '', f"{myPrompt[:20]}{str(time.time())[:10]}") #remove non-alphanumeric chars from string 
outFileName = cleanedFileNameString + ".png"

#pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")

#Example https://github.com/fofr/cog-sdxl-turbo/blob/main/predict.py
pipe = AutoPipelineForText2Image.from_pretrained(
    modelPath, 
    torch_dtype=torch.float16,
    use_safetensors=True,
    watermark=None,
    safety_checker=None,
    variant="fp16",
    ).to("cuda")

#https://huggingface.co/docs/diffusers/api/loaders/lora#diffusers.loaders.LoraLoaderMixin.get_active_adapters
pipe.load_lora_weights(loraModelPath, weight_name="lora.safetensors")
#pipe.fuse_lora() #https://github.com/huggingface/diffusers/issues/4919

#See parameters here 
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.FlaxStableDiffusionPipeline.__call__.prompt

for i in range(numImages):
    pipeOUT = pipe(
        prompt=myPrompt, 
        height=outHeight, #default 1024
        width=outWidth, #default 1024
        num_inference_steps=numInferenceSteps, 
        guidance_scale=guidanceScale,
        negative_prompt=myNegativePrompt,
        num_images_per_prompt=1
        )#[0] #.images[0]

    print(pipeOUT)
    
    image = pipeOUT.images
    print(myPrompt)
    
    image[0].save(f"./outputs/{i}_{outFileName}")
    print(f"==================== image {i} done and saved to ./outputs/{i}_{outFileName} ====================")
            