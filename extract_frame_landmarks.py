import argparse
import torchlm
import torch

import cv2
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet

from tqdm import tqdm
import os
import numpy as np

def save_lmds(dict_item, txt_path):
    with open(txt_path, 'w') as obj:
        for name, landmarks in dict_item.items():
            obj.write(name + " ")
            for x, y in landmarks:
                obj.write(f"{int(x)}_{int(y)} ")
            obj.write("\n")

def main(from_dir, lmd_output_dir, skip_existing):
    device = torch.device("cuda:0")
    torchlm.runtime.bind(faceboxesv2(device=device))  

    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True,
                num_nb=10, num_lms=68, net_stride=32, input_size=256,
                meanface_type="300w", map_location=device, checkpoint=None)
    ) 

    for clip_dir in os.listdir(from_dir):
        lmd_path = os.path.join(lmd_output_dir, f'{clip_dir}.txt')

        if skip_existing and os.path.exists(lmd_path):
            continue

        frames_path = os.path.join(from_dir, clip_dir)
        img_lists = sorted(os.listdir(frames_path))

        current_dict = {}
        for image_name in tqdm(img_lists):
            if not image_name.endswith('.png'):
                continue
            frame = cv2.imread(os.path.join(frames_path, image_name))
            landmarks, bboxes = torchlm.runtime.forward(frame)

            assert len(bboxes) != 0

            current_dict[image_name] = [(x, y) for x, y in landmarks[0][:68]]
        save_lmds(current_dict, lmd_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frame landmarks.')
    parser.add_argument('--from_dir', type=str, default='./data_processing/specified_formats/videos/video_frames/',
                        help='Directory where video frames are stored')
    parser.add_argument('--lmd_output_dir', type=str, default='./data_processing/specified_formats/videos/landmarks/',
                        help='Directory where landmarks will be saved')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if landmarks file already exists')
    args = parser.parse_args()

    main(args.from_dir, args.lmd_output_dir, args.skip_existing)
