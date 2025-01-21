# coding: utf-8
# this code is modified from https://github.com/cleardusk/3DDFA_V2
__author__ = 'cleardusk'

import argparse
import imageio
from tqdm import tqdm
import yaml, cv2
import os, random
import sys
import numpy as np

#================================
path_3ddfav2_file = '.3ddfav2_path'
if os.path.exists(path_3ddfav2_file):
    with open(path_3ddfav2_file, 'r') as file:
        path_3ddfav2 = file.read().strip()
    
    if not os.path.exists(path_3ddfav2):
        raise FileNotFoundError("Please clone the 3DDFA_V2 environment and install it before proceeding.")
else:
    raise FileNotFoundError("Please clone the 3DDFA_V2 environment and install it before proceeding.")
#================================

# Add the path to sys.path
sys.path.append(path_3ddfav2)

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark, get_suffix
from utils.pose import draw_pose


def main(args):
    cfg = yaml.load(open(os.path.join(path_3ddfav2, 'configs/mb1_120x120.yml')), Loader=yaml.SafeLoader)

    cfg['bfm_fp'] = os.path.join(path_3ddfav2, cfg['bfm_fp'])
    cfg['checkpoint_fp'] = os.path.join(path_3ddfav2, cfg['checkpoint_fp'])
    print(cfg)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        print(tddfa)
        face_boxes = FaceBoxes()

    # Paths are now taken from args
    prefix = args.video_frames_dir
    visualization_dir = args.visualization_dir
    y_p_r_prefix = args.pose_data_dir
    fps = args.fps
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(y_p_r_prefix, exist_ok=True)
    
    clip_names = os.listdir(prefix)
    random.shuffle(clip_names) 
    
    for clip_name in tqdm(clip_names): 
        
        y_p_r_path = os.path.join(y_p_r_prefix, f"{clip_name}.npy")
        if os.path.exists(y_p_r_path):
            if args.check_mode:
                gaze_npy = np.load(y_p_r_path)
                if gaze_npy.shape[0] == 1:
                    import pdb;pdb.set_trace()
            continue

        video_wfp = os.path.join(visualization_dir, clip_name + ".mp4")
    
        # Sort the frames by filename before reading
        frame_filenames = sorted(os.listdir(os.path.join(prefix, clip_name)))
        frame_filepaths = [os.path.join(prefix, clip_name, frame) for frame in frame_filenames]
        # print(frame_filepaths)
        # reader = imageio.get_reader(frame_filepaths)

        
        writer = imageio.get_writer(video_wfp, fps=fps)

        dense_flag = False
        pre_ver = None
        
        pose_lists = []
        for i, frame_path in tqdm(enumerate(frame_filepaths)):
            frame_bgr = cv2.imread(frame_path)

            if frame_bgr is None:
                break

            if i == 0:
                # the first frame, detect face, here we only use the first face, you can change depending on your need
                boxes = face_boxes(frame_bgr)
                if len(boxes) == 0 :
                    print('f{clip_name} breaks at frame {i}: no face detected')
                    break
                boxes = [boxes[0]]
                param_lst, roi_box_lst, _ = tddfa(frame_bgr, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

                # refine
                param_lst, roi_box_lst , _= tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            else:
                try:
                    param_lst, roi_box_lst, _ = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
                except Exception as e:
                    print(f"Error processing frame {i} in clip {clip_name}: {e}")
                    break

                roi_box = roi_box_lst[0]
                # todo: add confidence threshold to judge the tracking is failed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame_bgr)
                    if len(boxes) == 0 :
                        print('f{clip_name} breaks at frame {i}: no face detected')
                        break
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst , _= tddfa(frame_bgr, boxes)

                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            pre_ver = ver  


            # NOTE: Add this function to 3ddfav2: utils/pose.py
            # def draw_pose(img, param_lst, ver_lst):
            #     P, pose = calc_pose(param_lst[0])
            #     img = plot_pose_box(img, P, ver_lst)
            #     return img, pose

            res, pose = draw_pose(frame_bgr, param_lst, ver)
            pose_lists.append(pose)
            
            writer.append_data(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

        writer.close()
        print(f'Dump to {video_wfp}')
        

        pose_array = np.array(pose_lists)
        
        np.save(os.path.join(y_p_r_prefix, f"{clip_name}.npy"), pose_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
    parser.add_argument('-v', '--video_frames_dir', type=str, default='data_processing/specified_formats/videos/video_frames/', help='Directory containing video frames')
    parser.add_argument('-d', '--visualization_dir', type=str, default='data_processing/specified_formats/videos/pose_orientations/visualization/', help='Directory for output visualizations')
    parser.add_argument('-p', '--pose_data_dir', type=str, default='data_processing/specified_formats/videos/pose_orientations/pose_data/', help='Directory for pose data output')
    parser.add_argument('-f', '--fps', type=int, default=25, help='Frames per second for output video')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='pose', choices=['2d_sparse', '3d', 'pose'])
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--check_mode', action='store_true',
                        help='Check missing gazes.')
    args = parser.parse_args()
    main(args)
