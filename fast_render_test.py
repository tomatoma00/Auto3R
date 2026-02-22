import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,needgt = True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    psnr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "psnr.txt")
    ssim_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ssim.txt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    psnrs = []
    ssims = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        psnrs.append(psnr(rendering, gt).mean().double())
        ssims.append(ssim(rendering, gt).mean().double())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if needgt:
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    with open(psnr_path,'w') as f:
        for pp in psnrs:
            f.writelines(f"{pp}\n")
    with open(ssim_path,'w') as f:
        for ss in ssims:
            f.writelines(f"{ss}\n")
    

bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,needgt=True):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,needgt)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def render_multisets(dataset : ModelParams, iterations : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iterations[0], shuffle=False)
        for i in range(len(iterations)):
            render_set(dataset.model_path, "train", iterations[i], scene.getTrainCameras(), gaussians, pipeline, background,i==0)
            if i < len(iterations)-1:
                # gaussians = GaussianModel(dataset.sh_degree)
                gaussians.load_ply(os.path.join(dataset.model_path,"point_cloud",f"iteration_{iterations[i+1]}","point_cloud.ply"))
                

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", nargs="+", default=[-1], type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--repeat", type=int,default=1)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_multisets(model.extract(args), args.iteration, pipeline.extract(args))
