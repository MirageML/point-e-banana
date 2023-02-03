# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
import argparse

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh

from diffusers import StableDiffusionPipeline

import trimesh


state = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_state(s):
    print(s)
    global state
    state = s

def get_state():
    return state

def load_img2mesh_model(model_name):
    set_state(f'Creating img2mesh model {model_name}...')
    i2m_name = model_name
    i2m_model = model_from_config(MODEL_CONFIGS[i2m_name], device)
    i2m_model.eval()
    base_diffusion_i2m = diffusion_from_config(DIFFUSION_CONFIGS[i2m_name])

    set_state(f'Downloading img2mesh checkpoint {model_name}...')
    i2m_model.load_state_dict(load_checkpoint(i2m_name, device))

    return i2m_model, base_diffusion_i2m



def get_sampler(model_name, txt2obj, guidance_scale):
    if txt2obj:
        set_state('Creating txt2mesh model...')
        t2m_name = 'base40M-textvec'
        t2m_model = model_from_config(MODEL_CONFIGS[t2m_name], device)
        t2m_model.eval()
        base_diffusion_t2m = diffusion_from_config(DIFFUSION_CONFIGS[t2m_name])

        set_state('Downloading txt2mesh checkpoint...')
        t2m_model.load_state_dict(load_checkpoint(t2m_name, device))
    else:
        i2m_model, base_diffusion_i2m = load_img2mesh_model(model_name)

    set_state('Creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    set_state('Downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    return PointCloudSampler(
            device=device,
            models=[t2m_model if txt2obj else i2m_model, upsampler_model],
            diffusions=[base_diffusion_t2m if txt2obj else base_diffusion_i2m, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[guidance_scale, 0.0 if txt2obj else guidance_scale],
            model_kwargs_key_filter=('texts', '') if txt2obj else ("*",)
        )


def prepare_img(img):

    w, h = img.size
    if w > h:
        img = img.crop((w - h) / 2, 0, w - (w - h) / 2, h)
    else:
        img = img.crop((0, (h - w) / 2, w, h - (h - w) / 2))

    # resize to 256x256
    img = img.resize((256, 256))

    return img


def ply_to_glb(ply_file, glb_file):
    mesh = trimesh.load(ply_file)

    # Save the mesh as a glb file using Trimesh
    mesh.export(glb_file, file_type='glb')

    return glb_file

def save_ply(pc, file_name, grid_size):
    set_state('Creating SDF model...')
    sdf_name = 'sdf'
    sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
    sdf_model.eval()

    set_state('Loading SDF model...')
    sdf_model.load_state_dict(load_checkpoint(sdf_name, device))

    # Produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=grid_size, # increase to 128 for resolution used in evals
        progress=True,
    )

    # Write the mesh to a PLY file to import into some other program.
    with open(file_name, 'wb') as f:
        mesh.write_ply(f)

def create_gif(pc):
    fig = plt.figure(facecolor='black', figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75))

    # Create an empty list to store the frames
    frames = []

    # Create a loop to generate the frames for the GIF
    for angle in range(0, 360, 4):
        # Clear the plot and plot the point cloud
        ax.clear()
        color_args = np.stack(
            [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
        )
        c = pc.coords


        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=color_args)

        # Set the viewpoint for the plot
        ax.view_init(elev=10, azim=angle)

        # Turn off the axis labels and ticks
        ax.axis('off')
        ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
        ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
        ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])

        # Draw the figure to update the image data
        fig.canvas.draw()

        # Save the plot as a frame for the GIF
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        w, h = frame.shape[0], frame.shape[1]
        i = int(round((h - int(h*0.6)) / 2.))
        frame = frame[i:i + int(h*0.6),i:i + int(h*0.6)]
        frames.append(frame)

    # Save the GIF using imageio
    imageio.mimsave('/tmp/pointcloud.mp4', frames, fps=30)
    return '/tmp/pointcloud.mp4'

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def download_model():
    global text_model
    global img_model

    text_model = get_sampler("base40M-textvec", txt2obj=True, guidance_scale=0.0)
    img_model = get_sampler("base1B", txt2obj=False, guidance_scale=3.0)


if __name__ == "__main__":
    download_model()
