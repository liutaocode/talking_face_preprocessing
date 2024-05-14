import argparse
import torchlm
import torch
import cv2
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tqdm import tqdm
import os

def main(from_dir_prefix, output_dir_prefix, expanded_ratio, skip_per_frame):
    os.makedirs(output_dir_prefix, exist_ok=True)
    
    device = torch.device("cuda:0")
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, 
            net_stride=32, input_size=256, meanface_type="300w", 
            map_location=device, checkpoint=None
        )
    )

    for mp4_name in tqdm(os.listdir(from_dir_prefix)):
        from_mp4_file_path = os.path.join(from_dir_prefix, mp4_name)
        to_mp4_file_path = os.path.join(output_dir_prefix, mp4_name)
        
        if os.path.exists(to_mp4_file_path):
            continue

        video = cv2.VideoCapture(from_mp4_file_path)
        index = 0
        bboxes_lists = []
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while video.isOpened():
            success = video.grab()
            if index % skip_per_frame == 0:
                success, frame = video.retrieve()
                if not success:
                    break
                landmarks, bboxes = torchlm.runtime.forward(frame)
                    
                if bboxes.shape == (1, 5):
                    bboxes_lists.append(bboxes[0])
                elif bboxes.shape[0] > 0:
                    # If multiple persons exist, select the one with the largest width
                    max_bboxes = max(bboxes, key=lambda bbox: bbox[2] - bbox[0])
                    bboxes_lists.append(max_bboxes)
            index += 1

        x_center_lists, y_center_lists, width_lists, height_lists = [], [], [], []
        for bbox in bboxes_lists:
            x1, y1, x2, y2 = bbox[:4]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_lists.append(x_center)
            y_center_lists.append(y_center)
            width_lists.append(x2 - x1)
            height_lists.append(y2 - y1)
            
        if not (x_center_lists and y_center_lists and width_lists and height_lists):
            print(f"Face may not exist, please check the video: {mp4_name}")
            exit(0)
            continue

        x_center = sorted(x_center_lists)[len(x_center_lists) // 2]
        y_center = sorted(y_center_lists)[len(y_center_lists) // 2]
        median_width = sorted(width_lists)[len(width_lists) // 2]
        median_height = sorted(height_lists)[len(height_lists) // 2]
        
        expanded_width = int(median_width * (1 + expanded_ratio))
        expanded_height = int(median_height * (1 + expanded_ratio))
        
        fixed_cropped_width = min(max(expanded_width, expanded_height), width, height)
        
        x1, y1 = int(x_center - fixed_cropped_width / 2), int(y_center - fixed_cropped_width / 2)

        if args.debug:
            print(cmd)
            cmd = (
                f'ffmpeg -i {from_mp4_file_path} -filter:v "crop={fixed_cropped_width}:{fixed_cropped_width}:{x1}:{y1},'
                f'pad={fixed_cropped_width}:{fixed_cropped_width}:{x1 + fixed_cropped_width}:{y1 + fixed_cropped_width}" '
                f'-c:a copy {to_mp4_file_path} -y'
            )
        else:
            cmd = (
                f'ffmpeg -i {from_mp4_file_path} -filter:v "crop={fixed_cropped_width}:{fixed_cropped_width}:{x1}:{y1},'
                f'pad={fixed_cropped_width}:{fixed_cropped_width}:{x1 + fixed_cropped_width}:{y1 + fixed_cropped_width}" '
                f'-c:a copy {to_mp4_file_path} -y -loglevel quiet'
            )
        
        if os.system(cmd) != 0 and args.debug:
            print(f"Error executing command: {cmd}, please check")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--from_dir_prefix', type=str, default='data_processing/raw_data/',
                        help='input directory where raw videos are stored')
    parser.add_argument('--output_dir_prefix', type=str, default='data_processing/cropped_faces/',
                        help='output directory where cropped faces will be stored')
    parser.add_argument('--expanded_ratio', type=float, default=0.6,
                        help='ratio to expand the bounding box for cropping')
    parser.add_argument('--skip_per_frame', type=int, default=25,
                        help='number of frames to skip during processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    main(args.from_dir_prefix, args.output_dir_prefix, args.expanded_ratio, args.skip_per_frame)
