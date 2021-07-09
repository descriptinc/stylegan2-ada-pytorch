# pyright: reportMissingImports=false
# %%
from re import I
from IPython import get_ipython
import warnings

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
warnings.filterwarnings('ignore')
# %%
import sys
import os
import pathlib
import numpy as np
import copy
from PIL import Image
import torch
import stylegan2_ada.swapping_utils as swap_utils
from einops import rearrange
import matplotlib.pyplot as plt  # pyright: reportMissingModuleSource=false
from mpl_toolkits.axes_grid1 import ImageGrid
# %%
home = pathlib.Path.home()
project_dir = os.path.join(home, "Code", "git", "stylegan2_ada")
pretrained_dir = os.path.join(project_dir, "pretrained")
target_image_path = os.path.join(project_dir, "data")
results_path = os.path.join(project_dir, "results")

model2 = "ffhq-res1024-mirror-stylegan2-noaug"
# model1 = "cartoon"
# model1 = "ukiyoe_0400"
model1 = "cartoon_1000"
target_image_name = "susho_crop2"
image_ext = "jpg"

device = torch.device("cuda:0")
num_steps = 10000
swap_layer = 5
ws_idx = -1
noise_mode = "const"

model1_filename = os.path.join(pretrained_dir, model1 + ".pkl")
model2_filename = os.path.join(pretrained_dir, model2 + ".pkl")
target_image_filename = os.path.join(target_image_path, f"{target_image_name}.{image_ext}")
ws_filename = os.path.join(results_path, f"{target_image_name}_{model2}_ws_{num_steps}.pt")

# %%
model1 = swap_utils.load_pickle_model(model1_filename)["G_ema"].to(device)
model2 = swap_utils.load_pickle_model(model2_filename)["G_ema"].to(device)
swapped_generator = swap_utils.swap_layers(model1, model2, swap_layer)

# %%
target_img = Image.open(target_image_filename).resize((1024, 1024), Image.BICUBIC)
# ws = swap_utils.image2ws(target_image, model2, num_steps, device, ws_filename)
ws = torch.load(ws_filename)[ws_idx][None].to(device)

# %%
cartoon_img = swap_utils.ws2image(ws, model1, noise_mode)
face_img = swap_utils.ws2image(ws, model2, noise_mode)
swapped_img = swap_utils.ws2image(ws, swapped_generator, noise_mode)

# %%

figsize = (5, 20)
images = [target_img, face_img, swapped_img, cartoon_img]
fig = plt.figure(figsize=figsize)
# grid = iter(ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.1))
grid = iter(ImageGrid(fig, 111, nrows_ncols=(4, 1), axes_pad=0.1))

for ax, img in zip(grid, images):
    ax.imshow(img)

plt.show()
# %%
print(ws.shape)
# %%
fakes_grid = Image.open(os.path.join(project_dir, "training_runs", "00000--mirror-paper1024-bgcfnc-resumeffhq1024", "fakes001000.png"))
H, W = fakes_grid.size
fakes_grid = fakes_grid.resize((H // 8, W // 8), Image.BICUBIC)
fakes_grid.save(os.path.join(project_dir, "training_runs", "00000--mirror-paper1024-bgcfnc-resumeffhq1024", "fakes001000_1024.png"))
# %%
# %%
