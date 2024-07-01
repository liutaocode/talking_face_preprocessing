import os
import argparse
# from tqdm import tqdm 
# Set up argument parser
# the reason using FaceLandmarkImg is from : https://github.com/TadasBaltrusaitis/OpenFace/issues/149 (for resolving order ambiguity)
parser = argparse.ArgumentParser(description='Extract Facial Action Units using OpenFace.')
parser.add_argument('--openface_bin', type=str, default='/home/openface-build/build/bin/FaceLandmarkImg', help='Path to OpenFace FeatureExtraction binary.')
parser.add_argument('--from_dir_path', type=str, default='data_processing/specified_formats/videos/video_frames/', help='Directory path for input video frames.')
parser.add_argument('--to_dir_path', type=str, default='data_processing/specified_formats/videos/facial_action_units/', help='Directory path for output facial action units.')

# Parse arguments
args = parser.parse_args()

for filename in os.listdir(args.from_dir_path):
    # python 2.7 format (docker)
    print(filename)
    saved_to_dir = os.path.join(args.to_dir_path, filename)
    if os.path.exists(saved_to_dir):
        continue
    cmd = '{} -fdir {} -out_dir {} -aus'.format(args.openface_bin, os.path.join(args.from_dir_path, filename), saved_to_dir)
    os.system(cmd)