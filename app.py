import gradio as gr
from fastai.vision.all import *


# Define a function for model inference
def classify_image(inp):
    learn = load_learner('model.pkl')
    img = PILImage.create(inp)
    img.thumbnail((128,128))
    pred,pred_idx,probs = learn.predict(img)
    if pred == "dog": 
        result = f"The image is a Dog with probability {probs[pred_idx]:.4f}"
    else:
        result = f"The image is look like a Cat {probs[pred_idx]:.4f}"
    return result

example = ['img/cat.jpeg', 'img/Dog.webp']

iface = gr.Interface(fn=classify_image, inputs="image", outputs="text", examples=example)
iface.launch()
