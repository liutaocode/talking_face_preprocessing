from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from os.path import basename, dirname, join
import os, shutil
import argparse
from glob import glob

def split_video_into_scenes(video_path, output_directory, threshold=16.0):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    if not os.path.exists(join(output_directory, "ss")):
        os.makedirs(join(output_directory, "ss"))
    
    output_file=output_directory + '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4'
    if len(scene_list)==0:
        shutil.copy(video_path, join(output_directory, basename(video_path)))
    else:
        split_video_ffmpeg(video_path, scene_list, output_file_template=output_file, arg_override='-r 25 -map 0 -c:v libx264 -preset slow -b:v 6000k -c:a aac',  show_progress=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split videos into scenes.')
    parser.add_argument('--from_directory', type=str, default='/path/to/before_scene_detected/', help='Directory where the original videos are located.')
    parser.add_argument('--output_directory', type=str, default='/path/to/after_scene_detected/', help='Directory where the split scenes will be saved.')

    args = parser.parse_args()

    video_paths = glob(os.path.join(args.from_directory, '*.mp4'))
    
    for video_path in video_paths:
        split_video_into_scenes(video_path, args.output_directory)
