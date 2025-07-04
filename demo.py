import sys

sys.path.append("core")

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from raft_stereo import RAFTStereo
from tqdm import tqdm
from utils.frame_utils import readPFM, writePFM
from utils.utils import InputPadder

DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for imfile1, imfile2 in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            if args.mirror:
                image1, image2 = image2, image1

            padder = InputPadder(image1.shape, divis_by=32, mirror=args.mirror)
            image1, image2 = padder.pad(image1, image2)

            if args.profile:
                for _ in range(10):
                    torch.cuda.synchronize()
                    res = model(image1, image2, iters=args.valid_iters, test_mode=not args.train)
            else:
                flow_init = None
                if args.init_depth:
                    depth_init = np.load(args.init_depth)
                    depth_init = torch.tensor(depth_init.copy()).to(DEVICE)
                    flow_init = depth_init
                res = model(image1, image2, iters=args.valid_iters, flow_init=flow_init, test_mode=not args.train)
            flow_up = padder.unpad(res[-1]).squeeze()
            file_stem = Path(imfile1).stem
            if args.train:
                if args.train.suffix == ".pfm":
                    gt = readPFM(args.train)
                elif args.train.suffix == ".png":
                    gt = cv2.imread(args.train, cv2.IMREAD_UNCHANGED).astype(np.float32)
                for i, flow_i in enumerate(res[:-1]):
                    flow_i = padder.unpad(flow_i).squeeze().cpu().numpy()
                    assert gt.shape == flow_i.shape
                    mask = gt < 1e3
                    loss_i = np.abs(-flow_i[mask] - gt[mask]).mean()
                    print(f"Loss at iteration {i}: {loss_i.item()}")
                    if args.save_pfm:
                        writePFM(output_directory / f"{file_stem}_iter{i}.pfm", -flow_i)
                    plt.imsave(output_directory / f"{file_stem}_iter{i}.png", -flow_i, cmap="jet")
            if args.save_pfm:
                writePFM(output_directory / f"{file_stem}.pfm", -flow_up.cpu().numpy().squeeze())
            plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap="jet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", help="restore checkpoint", required=True)
    parser.add_argument("--save_pfm", action="store_true", help="save output as numpy arrays")
    parser.add_argument("-l", "--left_imgs", help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument("-r", "--right_imgs", help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument("--output_directory", help="directory to save output", default="demo_output")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--valid_iters", type=int, default=32, help="number of flow-field updates during forward pass")

    # Architecture choices
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3, help="hidden state and context dimensions")
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument("--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders")
    parser.add_argument("--corr_levels", type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument("--n_downsample", type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")

    # profile
    parser.add_argument("--cuda_graph", action="store_true", help="use CUDA graph")
    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("--train", type=Path, help="enable training mode and specify a label")
    parser.add_argument("--init_depth", type=Path, help="replace init depth with a specified pfm file")
    parser.add_argument("--mirror", action="store_true", help="mirror the input images and swap the left & right")

    args = parser.parse_args()

    demo(args)
