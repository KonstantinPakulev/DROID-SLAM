# noinspection PyInterpreter
import sys

sys.path.append('..')
sys.path.append('../droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse

import torch.nn.functional as F
from droid_slam.droid import Droid

import matplotlib.pyplot as plt


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def read_associations(path):
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    associations = []

    for i in lines:
        associations.append(i.split(' ')[1::2])

    return associations


def image_stream(associations, use_depth=False):
    """ image generator """
    cam_params = np.load(os.path.join(args.datapath, 'camera_params.npy')).astype(np.float32)
    fx, fy, cx, cy = cam_params[0, 0], cam_params[1, 1], cam_params[0, 2], cam_params[1, 2]

    for t, (image_path, depth_path) in enumerate(associations):
        image = cv2.imread(os.path.join(args.datapath, image_path))
        depth = cv2.imread(os.path.join(args.datapath, depth_path), cv2.IMREAD_ANYDEPTH) / 5000.0

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None, None], (h1, w1)).squeeze()
        depth = depth[:h1 - h1 % 8, :w1 - w1 % 8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if use_depth:
            yield t, image[None], depth, intrinsics

        else:
            yield t, image[None], intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--association_file")
    parser.add_argument("--output_file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=16)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=0)

    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--depth", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # this can usually be set to 2-3 except for "camera_shake" scenes
    # set to 2 for test scenes
    stride = 1

    associations = read_associations(os.path.join(args.datapath, args.association_file))[::stride]

    for (t, image, depth, intrinsics) in tqdm(image_stream(associations, use_depth=True)):
        if not args.disable_vis:
            show_image(image[0])

        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, depth, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(associations, use_depth=False))

    if args.output_file is not None:
        from evo.tools import file_interface
        from evo.core.trajectory import PoseTrajectory3D
        
        timestamps = [float(a[0].split('/')[-1][:-4]) for a in associations]
        pose_traj_3d = PoseTrajectory3D(positions_xyz=traj_est[:, :3],
                                        orientations_quat_wxyz=traj_est[:, 3:],
                                        timestamps=np.array(timestamps))

        file_interface.write_tum_trajectory_file(args.output_file, pose_traj_3d)


# image_list = sorted(glob.glob(os.path.join(datapath, 'color', '*.png')))[::stride]
# depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.png')))[::stride]

# print(len(image_list), len(depth_list))

# for t, (image_file, depth_file) in enumerate(zip(image_list, depth_list)):
#     print(image_file)

# fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
# print("#" * 20 + " Results...")
#
# import evo

# from evo.core import sync
# import evo.main_ape as main_ape
# from evo.core.metrics import PoseRelation
#
# traj_ref = file_interface.read_tum_trajectory_file(args.gt_path)
# traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
#
# result = main_ape.ape(traj_ref, traj_est, est_name='traj',
#                       pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
#
# print(result.stats)

