import os
import gradio as gr
import sys 
sys.path.append(".")

#@title Import stuff
import gc

import subprocess
import shutil
from PIL import Image
import time

import imageio


# def run(initial_image, mask, Backgrounds, Backgrounds_complexity, Size, Angle, Steps, num_of_Images):
def run(source_img, Backgrounds, Backgrounds_complexity, Size, Angle, Steps, num_of_Images):
    print('-------------------starting to process-------------------')
    if os.path.exists('results'):
        shutil.rmtree("results")
    if os.path.exists('tmp'):
        shutil.rmtree("tmp")
    time.sleep(1)
    os.makedirs('results', exist_ok=True)
    os.makedirs('tmp/img', exist_ok=True)
    os.makedirs('tmp/mask', exist_ok=True)
    os.makedirs('tmp/bg', exist_ok=True)

    '''
    print('-----initial_image: ', initial_image)
    init_image = Image.open(initial_image)
    mask = Image.open(mask)
    init_image = init_image.resize((256,256))
    mask = mask.resize((256,256))
    init_image.save("tmp/img/input.JPEG")
    mask.save("tmp/mask/input.png")
    '''
    imageio.imwrite("tmp/img/input.JPEG", source_img["image"])
    imageio.imwrite("tmp/mask/input.png", source_img["mask"])
    
    initial_image = Image.open('tmp/img/input.JPEG').resize((256,256))
    initial_image.save('tmp/img/input.JPEG')
    mask = Image.open('tmp/mask/input.png').resize((256,256))
    mask.save('tmp/mask/input.png')


    if Backgrounds:
        background_specific = Backgrounds
        if background_specific is not None:
            background_specific = Image.open(background_specific).convert('RGB') # Specified background
            background_specific = background_specific.resize((256,256))
            background_specific.save('tmp/bg/bg.png')
            background_specific = '../tmp/bg/bg.png'
    else:
        background_specific = ""

    Backgrounds_complexity = Backgrounds_complexity
    Size = Size
    Angle = Angle
    Steps = Steps
    num_of_Images = num_of_Images
    print(Backgrounds_complexity, background_specific, Size, Angle, Steps, num_of_Images)
    p = subprocess.Popen(["sh", "run.sh", str(Backgrounds_complexity), background_specific, str(Size), str(Angle), str(Steps), str(num_of_Images)])
    
    # subprocess.Popen(["cd", "object_removal/TFill/"])
    # subprocess.Popen(["python", "test.py"])
    
    return_code = p.wait()
    print('----return_code: ', return_code)

    if os.path.exists('results/edited.png'):
        return Image.open('results/edited.png')
    else:
        return Image.open('tmp/img/input.JPEG')


image = gr.outputs.Image(type="pil", label="Your result")
css = ".output-image{height: 528px !important} .output-carousel .output-image{height:272px !important} a{text-decoration: underline}"
iface = gr.Interface(fn=run, inputs=[
    # gr.inputs.Image(type="filepath", label='initial_image'),
    gr.Image(source="upload", type="numpy", tool="sketch", elem_id="source_container"),
    # gr.inputs.Image(type="filepath", label='mask - object mask', optional=True),
    gr.inputs.Image(type="filepath", label='Backgrounds - optional, specified backgrounds'),
    gr.inputs.Slider(label="Backgrounds_complexity - How complicated you wish to the generated image to be", default=0, step=1, minimum=-30, maximum=30),
    gr.inputs.Slider(label="Size - Object pixel rates", default=0.1, step=0.02, minimum=0.01, maximum=0.5),
    gr.inputs.Slider(label="Angle - Object angle", default=0, step=10, minimum=-180, maximum=180),
    gr.inputs.Slider(label="Steps - more steps can increase quality but will take longer to generate",default=10,maximum=100,minimum=1,step=1),
    gr.inputs.Slider(label="num_of_Images - How many images you wish to generate", default=2, step=1, minimum=1, maximum=4),

    # gr.inputs.Radio(label="Width", choices=[32,64,128,256],default=256),
    # gr.inputs.Radio(label="Height", choices=[32,64,128,256],default=256),
    # gr.inputs.Textbox(label="Prompt - try adding increments to your prompt such as 'oil on canvas', 'a painting', 'a book cover'",default="chalk pastel drawing of a dog wearing a funny hat"),
    #gr.inputs.Slider(label="ETA - between 0 and 1. Lower values can provide better quality, higher values can be more diverse",default=0.0,minimum=0.0, maximum=1.0,step=0.1),
    ], 
    # outputs=[image,gr.outputs.Carousel(label="Individual images",components=["image"]),gr.outputs.Textbox(label="Error")],
    outputs=["image"],
    css=css,
    title="Image Editing with Controls of Object Attributes including Backgrounds, Sizes, Positions and Directions",
    description="Demo for Image Editing with Controls of Object Attributes. *** NOTE!!! Due to the requirements of GPU, this demo cannot work on this website currently(it always returns the input image). Please download the codes and run them at your server. ***",
    article="ImageNet-E")
iface.launch(enable_queue=True, share=True)
