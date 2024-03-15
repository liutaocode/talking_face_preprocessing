from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from os.path import basename, join
import os, shutil
import argparse
from glob import glob

def split_video_into_scenes(video_path, output_directory, threshold):
    
    if any(fname.startswith(basename(video_path)) for fname in os.listdir(output_directory)):
        print(f'{basename(video_path)} exists, pass.')
        return
    
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    
    if len(scene_list) == 0:
        output_file = join(output_directory, basename(video_path))
        if os.path.exists(output_file):
            return
        shutil.copy(video_path, output_file)
    else:
        output_file = output_directory + '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4'
        if os.path.exists(output_file):
            return
        split_video_ffmpeg(video_path, scene_list, output_file_template=output_file, arg_override='-r 25 -map 0 -c:v libx264 -preset slow -b:v 6000k -c:a aac', show_progress=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split videos into scenes.')
    parser.add_argument('--from_directory', type=str, default='/path/to/before_scene_detected/', help='Directory where the original videos are located.')
    parser.add_argument('--output_directory', type=str, default='/path/to/after_scene_detected/', help='Directory where the split scenes will be saved.')
    parser.add_argument('--threshold', type=float, default=16.0, help='Threshold for scene detection.')

    args = parser.parse_args()

    video_paths = glob(os.path.join(args.from_directory, '*.mp4'))
    
    for video_path in video_paths:
        print(f'Processing {video_path}')
        split_video_into_scenes(video_path, args.output_directory, args.threshold)
