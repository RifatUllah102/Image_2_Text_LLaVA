#import libraries
import textwrap
import glob2
from io import BytesIO
import requests
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from termcolor import colored, cprint
import gc

# # Set device to cuda:1 at the beginning
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

#pretrained model path and 4 bit quantized model of LLava
# MODEL = 'llava-v1.6-vicuna-13b'
MODEL='llava-v1.6-34b'
# MODEL='llava-v1.5-7b'
# MODEL='llava-v1.5-13b'

model_name = get_model_name_from_path(MODEL)

print(f"Model Name: {model_name}")
print(os.listdir(MODEL))

#import the images from a picture directory

#path to thte pictures directory
path = 'Pictures/*.jpg'

#List to store images and their correponding file names
images = []
filenames = []

#read each images
for filename in glob2.glob(path):
    img = Image.open(filename)
    images.append(img)
    filenames.append(filename)

print(colored('Now, about to load the Pre-trained model.', 'yellow', attrs=['reverse', 'blink']))

# Import the tokenizer and the pretrained model for Llava
try:
    tokenizer,model,image_processor,context_len=load_pretrained_model(
        # model_path=MODEL,model_base=None,model_name=model_name#,load_4bit=True,
        model_path=MODEL,model_base=None,model_name=model_name, device_map='auto'#, load_4bit=True
    )
    # print("Model loaded successfully.")
    print(colored('Model loaded successfully...', 'green', attrs=['reverse', 'blink']))
except Exception as e:
    print(colored(f"Error loading pretrained model: {e}", 'red', attrs=['reverse', 'blink']))
    # print(f"Error loading pretrained model: {e}")
    raise

while True:
    # Show the user the image name and content of the chosen image.
    # The required image index must be within the range.
    print("Choose images from Files.")
    print(filenames)
    require_image = int(input("Enter an image index ranges from 0 to length of the list - 1: "))
    # require_image = 2
    if int(require_image) < len(filenames):
        print(f"Image Name: {filenames[require_image].split('/')[-1]}")  # Displaying the image name
        image = images[require_image]

        # To show the image
        img = mpimg.imread(filenames[require_image])
        imgplot = plt.imshow(img)
        plt.show()
    else:
        print(f"Image index {require_image} is out of range. Total images: {len(filenames)}")


    #Function to convert the required image into a tensor and process it for LLava model
    def process_image(image):
        args={"image_aspect_ratio":"pad"}
        image_tensor=process_images(image,image_processor,args)
        # return image_tensor.to(device,dtype=torch.float16)
        # # return image_tensor.to(device)
        return image_tensor.to(dtype=torch.float16)

    #processing the chosen image above using the function developed in the previous step
    processed_image=process_image(image)
    # print(type(processed_image), processed_image.shape)


    #Defining the model that will be used for conversation with the images
    CONV_MODE="llava_v0" #Change the version

    #using thte prompt template as per the LLava torch documentation
    def create_prompt(prompt:str):
        conv=conv_templates[CONV_MODE].copy()
        roles=conv.roles
        prompt=DEFAULT_IMAGE_TOKEN+"\n"+prompt
        conv.append_message(roles[0],prompt)
        conv.append_message(roles[1],None)
        return conv.get_prompt(),conv

    #the following is the default system message used by the AI model
    prompt,_=create_prompt("Describe the image")
    # print(prompt)
    print(colored(prompt, 'cyan', attrs=['reverse', 'blink']))

    #Creating a function to interrogate the images using the model downloded
    def ask_image(image:Image, prompt:str):
        image_tensor=process_image(image)
        prompt,conv=create_prompt(prompt)
        # input_ids=(
        #     tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt")
        #     .unsqueeze(0)
        #     .to(model.device)
        # )
        input_ids=(
            tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt")
            .unsqueeze(0)
            # .to(model.device)
        )

        stop_str=conv.sep if conv.sep_style!=SeparatorStyle.TWO else conv.sep2
        stopping_criteria=KeywordsStoppingCriteria(keywords=[stop_str],tokenizer=tokenizer,input_ids=input_ids)
        
        with torch.inference_mode():
            output_ids=model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.08, #change the value according to your needs
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        return tokenizer.decode(
            output_ids[0,input_ids.shape[1]:],skip_special_tokens=True
        ).strip()

    # %%time 
    #using the above paramers for the prompting as well as the processing of the image, tokenize of the prompt, encoding
    new_Prompt = input("Enter the prompt here: ")
    if new_Prompt =="exit": break
    print(colored('Explanation of the Image.', 'grey', attrs=['reverse', 'blink']))
    result=ask_image(image, new_Prompt)
    print(textwrap.fill(result,width=110))

    