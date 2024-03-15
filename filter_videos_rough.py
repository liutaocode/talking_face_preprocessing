import argparse
import os
import shutil
from moviepy.editor import VideoFileClip

def filter_by_duration(video_path, min_duration):
    try:
        clip = VideoFileClip(video_path)
        return clip.duration >= min_duration
    except:
        return False

def filter_by_filesize(video_path, min_size):
    size = os.path.getsize(video_path) / 1024  # Convert bytes to KB
    return size >= min_size

def filter_by_format(video_path):
    try:
        VideoFileClip(video_path)
        return True
    except:
        return False

def filter_videos(before_filtering_dir, after_filtering_dir, min_duration, min_size):
    if not os.path.exists(after_filtering_dir):
        os.makedirs(after_filtering_dir)
    
    for video_file in os.listdir(before_filtering_dir):
        video_path = os.path.join(before_filtering_dir, video_file)
        if (filter_by_duration(video_path, min_duration) and
            filter_by_filesize(video_path, min_size) and
            filter_by_format(video_path)):
            shutil.copy(video_path, os.path.join(after_filtering_dir, video_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter videos based on certain criteria.')
    parser.add_argument('--before_filtering_dir', type=str, required=True, help='Directory containing videos to filter.', default='/path/to/before_filtering/')
    parser.add_argument('--after_filtering_dir', type=str, required=True, help='Directory to save filtered videos.', default='/path/to/after_filtering/')
    parser.add_argument('--min_duration', type=float, required=True, help='Minimum duration of videos to keep (in seconds).', default=2.0)
    parser.add_argument('--min_size', type=float, required=True, help='Minimum file size of videos to keep (in KB).', default=10.0)
    args = parser.parse_args()
    
    filter_videos(args.before_filtering_dir, args.after_filtering_dir, args.min_duration, args.min_size)
