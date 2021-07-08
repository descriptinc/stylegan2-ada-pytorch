# pyright: reportMissingImports=false
# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
# %%
import sys
import os
import pathlib
import stylegan2_ada.style_mixing as style_mixing
# %%

# network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
home = pathlib.Path.home()
project_dir = os.path.join(home, "Code", "git", "stylegan2_ada")
network_pkl = os.path.join(project_dir, "pretrained", "ffhq-res1024-mirror-stylegan2-noaug.pkl")

# mixing faces

# exp = 0
# row_seeds = [0, 1, 2, 3]
# col_seeds = [4, 5, 6, 7]
# col_styles = [4]*4

# exp = 1
# row_seeds = [1] * 4
# col_seeds = [2] * 4
# col_styles = [1, 2, 3, 4]

# exp = 2
# row_seeds = [0, 1, 2, 3]
# col_seeds = [0, 1, 2, 3]
# col_styles = [4]*4

# exp = 3
# row_seeds = [1]*4
# col_seeds = [2]*4
# # col_styles = [1, 2, 3, 4]
# # col_styles = [2, 4, 6, 8]
# # col_styles = [17, 17, 17, 17]
# col_styles = [4]*4

# exp = 4
# row_seeds = [1]*4
# col_seeds = [2]*4
# # col_styles = [2, 4, 6, 8]
# col_styles = [2, 4, 4, 4]

# exp = 5
# row_seeds = [1]*4
# col_seeds = [2]*4
# col_styles = [2, 4, 6, 8]
# # col_styles = [2, 4, 4, 4]

# exp = 6
# row_seeds = [1]*4
# col_seeds = [2]*4
# col_styles = list(range(9, 18))

# exp = 7
# seed1 = [21, 22, 23, 24]
# seed2 = [25, 26, 27, 28]
# seed2_layers = [4, 8, 12, 16]

# exp = 8
# seed1 = [21, 22, 23, 24]
# seed2 = [25, 30, 27, 28]
# seed2_layers = [2, 4, 6, 9, 12]

exp = 10
seed1 = [21, 22, 23, 24]
seed2 = [25, 30, 27, 28]
seed2_layers = [2, 4, 6, 9, 12]


output_dir = os.path.join(project_dir, "results", str(exp))
# %%
# style_mixing.generate_style_mix(network_pkl, row_seeds, col_seeds, col_styles, truncation_psi=1.0, noise_mode="const", outdir=output_dir)
style_mixing.generate_style_grid(network_pkl, seed1, seed2, seed2_layers, truncation_psi=1.0, noise_mode="const", outdir=output_dir)
# %%
