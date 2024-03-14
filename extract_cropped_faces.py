import argparse
import torchlm
import torch
import cv2
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tqdm import tqdm
import os
import numpy as np

def main(from_dir_prefix, output_dir_prefix, expanded_ratio):
    device = torch.device("cuda:0")
    torchlm.runtime.bind(faceboxesv2(device=device))

    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True,
               num_nb=10, num_lms=68, net_stride=32, input_size=256,
               meanface_type="300w", map_location=device, checkpoint=None)
    )

    for mp4_name in tqdm(os.listdir(from_dir_prefix)):
        from_mp4_file_path = os.path.join(from_dir_prefix, mp4_name)
        to_mp4_file_path = os.path.join(output_dir_prefix, mp4_name)

        video = cv2.VideoCapture(from_mp4_file_path)
        index = 0
        bboxes_lists = []
        while video.isOpened():
            success = video.grab()

            if index % 25 == 0: # skip every one seconds
                success, frame = video.retrieve()
                if not success:
                    break
                landmarks, bboxes = torchlm.runtime.forward(frame)

                if bboxes.shape == (1, 5):
                    bboxes_lists.append(bboxes)
            index += 1

        x_center_lists, y_center_lists, width_lists, height_lists = [], [], [], []
        for i in range(len(bboxes_lists)):
            x1, y1, x2, y2 = bboxes_lists[i][0, :4]

            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_lists.append(x_center)
            y_center_lists.append(y_center)
            width_lists.append(x2 - x1)
            height_lists.append(y2 - y1)

        x_center_lists.sort(), y_center_lists.sort(), width_lists.sort(), height_lists.sort()
        x_center = x_center_lists[int(len(x_center_lists) / 2)]
        y_center = y_center_lists[int(len(y_center_lists) / 2)]

        new_width, new_height = width_lists[int(len(width_lists) / 2)], height_lists[int(len(height_lists) / 2)]
        new_width, new_height = int(new_width * (1 + expanded_ratio)), int(new_height * (1 + expanded_ratio))
        max_len = max(new_width, new_height)
        fixed_cropped_width = max_len

        x1, y1 = int(x_center - max_len / 2), int(y_center - max_len / 2)

        if x1 < 0:
            x1 = 0

        if y1 < 0:
            y1 = 0

        cmd = 'ffmpeg -i %s -filter:v "crop=%d:%d:%d:%d,pad=%d:%d:%d:%d" -c:a copy %s -y' % (
            from_mp4_file_path, fixed_cropped_width, max_len, x1, y1, fixed_cropped_width, fixed_cropped_width, x1 + fixed_cropped_width, y1 + fixed_cropped_width, to_mp4_file_path)
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--from_dir_prefix', type=str, default='data_processing/raw_data/',
                        help='input directory where raw videos are stored')
    parser.add_argument('--output_dir_prefix', type=str, default='data_processing/cropped_faces/',
                        help='output directory where cropped faces will be stored')
    parser.add_argument('--expanded_ratio', type=float, default=0.6,
                        help='ratio to expand the bounding box for cropping')
    args = parser.parse_args()

    main(args.from_dir_prefix, args.output_dir_prefix, args.expanded_ratio)

