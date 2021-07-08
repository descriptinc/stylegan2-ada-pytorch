# pyright: reportMissingImports=false
# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
# %%
import sys
import os
import pathlib
import pickle
import numpy as np
import copy
from PIL import Image
import torch
import stylegan2_ada.projector as projector
from einops import rearrange
import matplotlib.pyplot as plt  # pyright: reportMissingModuleSource=false
# %%
home = pathlib.Path.home()
project_dir = os.path.join(home, "Code", "git", "stylegan2_ada")
pretrained_dir = os.path.join(project_dir, "pretrained")
target_image_path = os.path.join(project_dir, "data")
results_path = os.path.join(project_dir, "results")

faces_pkl = "ffhq-res1024-mirror-stylegan2-noaug"
cartoon_pkl = "cartoon"
target_image_file = "susho_crop2"
image_ext = "jpg"

device = torch.device("cuda:1")
num_steps = 10000

# %%
with open(os.path.join(pretrained_dir, faces_pkl+".pkl"), 'rb') as f:
    faces_model = pickle.load(f)

with open(os.path.join(pretrained_dir, cartoon_pkl+".pkl"), 'rb') as f:
    cartoon_model = pickle.load(f)

faces_G = faces_model["G_ema"].to(device).eval()
cartoon_G = cartoon_model["G_ema"].to(device).eval()

print("Faces Model", faces_model.keys())
print("Cartoon Model", cartoon_model.keys())

# %%
target_image = Image.open(os.path.join(target_image_path, target_image_file + "." + image_ext))
target_image = target_image.resize((1024, 1024), Image.ANTIALIAS)
target_image = np.asarray(target_image)
target_image = torch.from_numpy(target_image).type(torch.float32)
target_image = rearrange(target_image, "H W C -> C H W")

print(target_image.shape, target_image.dtype)
# %%
ws = projector.project(faces_model["G_ema"], num_steps=num_steps, target=target_image, device=device)
# %%
torch.save(ws, os.path.join(results_path, f"{target_image_file}_{faces_pkl}_ws_{num_steps}.pt"))

# %%
ws = torch.load(os.path.join(results_path, f"{target_image_file}_{faces_pkl}_ws_{num_steps}.pt")).to(device)
# %%
# swapped_generator = copy.deepcopy(cartoon_model["G_ema"])
swapped_generator = copy.deepcopy(faces_G)
# print(swapped_generator)
# swap_layer_res = [2**i for i in range(6, 11)]
swap_layer_res = [2**i for i in range(2, 5)]

for res in swap_layer_res:
    # setattr(swapped_generator.synthesis, f"b{res}", getattr(faces_model["G_ema"].synthesis, f"b{res}"))
    setattr(swapped_generator.synthesis, f"b{res}", getattr(cartoon_G.synthesis, f"b{res}"))

# %%
def tensor2img(img, normalized=True, drange=255.0):
    if normalized:
        img = ((img + 1.0) / 2)
    
    img = (img * drange).type(torch.uint8)
    img = img.cpu().permute(1, 2, 0)
    return img


print(ws.device)
s = 999
noise_mode = "const"
cartoon_out = cartoon_G.synthesis(ws[s:s+1], noise_mode=noise_mode)
face_out = faces_G.synthesis(ws[s:s+1], noise_mode=noise_mode)
swapped_out = swapped_generator.synthesis(ws[s:s+1], noise_mode=noise_mode)

cartoon_out = ((cartoon_out + 1.0) / 2.0)[0].cpu().permute(1, 2, 0)
face_out = ((face_out + 1.0) / 2.0).cpu()[0].permute(1, 2, 0)
swapped_out = ((swapped_out + 1.0) / 2.0)[0].cpu().permute(1, 2, 0)
target_img = target_image.permute(1, 2, 0) / 255.0

# %%
plt.figure(figsize=(5, 5))
plt.imshow(swapped_out)

plt.figure(figsize=(5, 5))
plt.imshow(cartoon_out)

plt.figure(figsize=(5, 5))
plt.imshow(face_out)

plt.figure(figsize=(5, 5))
plt.imshow(target_img)
# %%
print(ws.shape)
# %%
fakes_grid = Image.open(os.path.join(project_dir, "training_runs", "00000--mirror-paper1024-bgcfnc-resumeffhq1024", "fakes001000.png"))
H, W = fakes_grid.size
fakes_grid = fakes_grid.resize((H // 8, W // 8), Image.BICUBIC)
fakes_grid.save(os.path.join(project_dir, "training_runs", "00000--mirror-paper1024-bgcfnc-resumeffhq1024", "fakes001000_1024.png"))
# %%

