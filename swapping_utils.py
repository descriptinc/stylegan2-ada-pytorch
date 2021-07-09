# pyright: reportMissingImports=false

import os
import torch
import pickle
import copy
from PIL import Image
import numpy as np
from einops import rearrange

import stylegan2_ada.projector as projector

def load_pickle_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def swap_layers(model1, model2, swap_layer_idxs):
    if isinstance(model1, str):
        model1 = load_pickle_model(model1)["G_ema"]
    
    if isinstance(model2, str):
        model2 = load_pickle_model(model2)["G_ema"]

    if isinstance(swap_layer_idxs, int):
        swap_layer_idxs = list(range(2, swap_layer_idxs))

    swapped_model = copy.deepcopy(model2)
    swap_layer_res = [2**i for i in swap_layer_idxs]

    for res in swap_layer_res:
        # setattr(swapped_generator.synthesis, f"b{res}", getattr(faces_model["G_ema"].synthesis, f"b{res}"))
        setattr(swapped_model.synthesis, f"b{res}", getattr(model1.synthesis, f"b{res}"))
    
    return swapped_model

def save_swapped_model(model1, model2, swap_layer_idxs, swapped_model_name):
    swapped_model = swap_layers(model1, model2, swap_layer_idxs)

    with open(swapped_model_name, 'wb') as f:
        pickle.dump(swapped_model, f)

def image2ws(target_image, generator, num_steps, device, filename=None):
    if isinstance(target_image, str):
        target_image = Image.open(target_image)
    if isinstance(generator, str):
         generator = load_pickle_model(generator).to(device)

    if isinstance(target_image, Image):
        target_image = target_image.resize((1024, 1024), Image.ANTIALIAS)
        target_image = np.asarray(target_image)
        target_image = torch.from_numpy(target_image).type(torch.float32)
        target_image = rearrange(target_image, "H W C -> C H W")
    
    ws = projector.project(generator, num_steps=num_steps, target=target_image, device=device)

    if filename is not None:
        torch.save(ws, filename)

    return ws

def ws2image(ws, generator, noise_mode="const", ws_idx=-1):
    if isinstance(ws, str):
        ws = torch.load(ws)[ws_idx][None]

    out = generator.synthesis(ws, noise_mode=noise_mode)
    out = torch2img(out[0])
    return out

def torch2img(img, normalize=True, drange=255.0):
    if normalize:
        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
    
    img = (img * drange).type(torch.uint8)
    img = img.cpu().permute(1, 2, 0)
    return img 
