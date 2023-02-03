from PIL import Image
import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
import base64
import io
import requests

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh

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


def expand2square(img):
    width, height = img.size

    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), "white")
        result.paste(img, (0, (width - height) // 2))
    else:
        result = Image.new(img.mode, (height, height), "white")
        result.paste(img, ((height - width) // 2, 0))

    return img


def ply_to_glb(ply_file, glb_file):
    mesh = trimesh.load(ply_file)

    # Save the mesh as a glb file using Trimesh
    mesh.export(glb_file, file_type='glb')

    return glb_file

def save_ply(pc, file_name, grid_size):
    set_state('Creating SDF model...')
    global sdf_model

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
    return file_name

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global text_model
    global img_model
    global sdf_model

    text_model = get_sampler("base40M-textvec", txt2obj=True, guidance_scale=0.0)
    img_model = get_sampler("base1B", txt2obj=False, guidance_scale=3.0)

    sdf_name = 'sdf'
    sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
    sdf_model.eval()

    set_state('Loading SDF model...')
    sdf_model.load_state_dict(load_checkpoint(sdf_name, device))

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global text_model
    global img_model

    set_state('Entered generate function...')

    # Parse out your arguments
    input = model_inputs.get('prompt', None)
    image = model_inputs.get('image', None)

    if image != None:
        # Get the image from url
        input = Image.open(requests.get(image, stream=True).raw)

    if input == None:
        return {'message': "No prompt provided"}

    # if input is a string, it's a text prompt
    sampler = text_model if isinstance(input, str) else img_model

    if isinstance(input, Image.Image):
        input = expand2square(input)

    # Produce a sample from the model.
    set_state('Sampling...')
    samples = None
    kw_args = dict(texts=[input]) if isinstance(input, str) else dict(images=[input])
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=kw_args):
        samples = x

    set_state('Converting to point cloud...')
    pc = sampler.output_to_point_clouds(samples)[0]

    set_state('Converting to mesh...')
    ply_path = save_ply(pc, '/tmp/mesh.ply', 128)
    glb_path = ply_to_glb('/tmp/mesh.ply', '/tmp/mesh.glb')

    # Run the model
    result = {
        'ply': str(base64.b64encode(open(ply_path, 'rb').read()).decode('utf-8')),
        'glb': str(base64.b64encode(open(glb_path, 'rb').read()).decode('utf-8')),
    }

    # Return the results as a dictionary
    return result
