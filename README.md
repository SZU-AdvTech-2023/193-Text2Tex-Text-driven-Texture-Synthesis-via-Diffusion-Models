# Text2Tex: Text-driven Texture Synthesis via Diffusion Models

<p align="center"><img src="docs/static/teaser/teaser.jpg" width="100%"/></p>

## Introduction

Text2Tex, a novel method for generating high-quality textures for 3D meshes from the given text prompts. The method incorporates inpainting into a pre-trained depth-aware image diffusion model to progressively synthesize high resolution partial textures from multiple viewpoints. To avoid accumulating inconsistent and stretched artifacts across views, we dynamically segment the rendered view into a generation mask, which represents the generation status of each visible texel. This partitioned view representation guides the depth-aware inpainting model to generate and update partial textures for the corresponding regions. Furthermore, we propose an automatic view sequence generation scheme to determine the next best view for updating the partial texture. Extensive experiments demonstrate that our method significantly outperforms the existing text-driven approaches and GAN-based methods.

origin web:
https://github.com/daveredrum/Text2Tex

For additional detail, please see the Text2Tex paper:  
"[Text2Tex: Text-driven Texture Synthesis via Diffusion Models]()"  


## Setup

The code is tested on Ubuntu 20.04 LTS with PyTorch 1.12.1 CUDA 11.3 installed. Please follow the following steps to install PyTorch first. To run our method, you should at least have a NVIDIA GPU with 12 GB RAM (NVIDIA GeForce 2080 Ti works for us).

```shell
# create and activate the conda environment
conda create -n text2tex python=3.9.15
conda activate text2tex

# install PyTorch 1.12.1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Then, install PyTorch3D:

```shell
# install runtime dependencies for PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# install PyTorch3D
conda install pytorch3d -c pytorch3d
```

Install `xformers` to accelerate transformers:

```shell
# please don't use pip to install it, as it only supports PyTorch>=2.0
conda install xformers -c xformers
```

Install the necessary packages listed out in requirements.txt:

```shell
pip install -r requirements.txt
```

To use the ControlNet Depth2img model, please download `control_sd15_depth.pth` from the [hugging face page](https://huggingface.co/lllyasviel/ControlNet/tree/main/models), and put it under `models/ControlNet/models/`.

## Usage

### Try the demo / sanity check

To make sure everything is set up and configured correctly, you can run the following script to generate texture for a backpack.

```shell
./bash/run.sh
```

The generation and refinement should take around 500 sec and 360 sec, respectively. Once the synthesis completes, you should be able to see all generated assets under `outputs/backpack/42-p36-h20-1.0-0.3-0.1`. Load the final mesh `outputs/backpack/42-p36-h20-1.0-0.3-0.1/update/mesh/19.obj` in MeshLab, you should be able to see this (so something similar):

<p align="center"><img src="docs/static/img/backpack.png" width="50%"/></p>

### Try your own mesh

For the best quality of texture synthesis, there are some necessary pre-processing steps for running our method on your own mesh:

1) Y-axis is up.
2) The mesh should face towards +Z.
3) The mesh bounding box should be origin-aligned (note simply averaging the vertices coordinates could be problematic).
4) The max length of the mesh bounding box should be around 1.

We provide `scripts/normalize_mesh.py` and `scripts/rotate_mesh.py` to make the mesh preprocessing easy for you.

If you already have a normalized mesh but haven't parameterized it yet, please use `scripts/parameterize_mesh.py` to generate the UV map.

> NOTE: we expect the mesh to be triangulated.

A mesh ready for next steps should look like this:

<p align="center"><img src="docs/static/img/preprocessed.jpg" width="50%"/></p>

Then, you can generate your own texture via:

```shell
python scripts/generate_texture.py \
    --input_dir <path-to-mesh> \
    --output_dir outputs/<run-name> \
    --obj_name <mesh-name> \
    --obj_file <mesh-name>.obj \
    --prompt <your-prompt> \
    --add_view_to_prompt \
    --ddim_steps 50 \
    --new_strength 1 \
    --update_strength 0.3 \
    --view_threshold 0.1 \
    --blend 0 \
    --dist 1 \
    --num_viewpoints 36 \
    --viewpoint_mode predefined \
    --use_principle \
    --update_steps 20 \
    --update_mode heuristic \
    --seed 42 \
    --post_process \
    --device 2080 \
    --use_objaverse # assume the mesh is normalized with y-axis as up
```

If you want some high-res textures, you can set `--device` to `a6000` for 3k resolution. To play around other parameters, please check `scripts/generate_texture.py`, or simply run `python scripts/generate_texture.py -h`.

## Benchmark on Objaverse subset

To generate textures for the Objaverse objects we used in the paper, please run the following script to download and pre-process those meshes:

```shell
python scripts/download_objaverse_subset.py
```

All pre-processed meshes will be downloaded to `data/objaverse/`.


