import subprocess
import os
import argparse

def convert_video_to_25fps(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video_file in os.listdir(source_folder):
        if video_file.endswith('.mp4'):
            source_path = os.path.join(source_folder, video_file)
            target_path = os.path.join(target_folder, video_file)
            subprocess.run(['ffmpeg', '-i', source_path, '-r', '25', target_path, '-y'])

def convert_audio_to_16k(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video_file in os.listdir(source_folder):
        if video_file.endswith('.mp4'):
            source_path = os.path.join(source_folder, video_file)
            audio_file = os.path.splitext(video_file)[0] + '.wav'
            target_path = os.path.join(target_folder, audio_file)
            subprocess.run(['ffmpeg', '-i', source_path, '-ar', '16000', target_path, '-y'])

def extract_frames_to_png(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video_file in os.listdir(source_folder):
        if video_file.endswith('.mp4'):
            source_path = os.path.join(source_folder, video_file)
            frame_folder = os.path.splitext(video_file)[0]
            frame_target_folder = os.path.join(target_folder, frame_folder)
            if not os.path.exists(frame_target_folder):
                os.makedirs(frame_target_folder)
            subprocess.run(['ffmpeg', '-i', source_path, '-vf', 'fps=25', os.path.join(frame_target_folder, '%06d.png'), '-y'])


def main():
    parser = argparse.ArgumentParser(description="Process videos and extract data.")
    parser.add_argument("--convert_video", type=bool, default=True, help="Enable video conversion to 25fps.")
    parser.add_argument("--convert_audio", type=bool, default=True, help="Enable audio conversion to 16kHz.")
    parser.add_argument("--extract_frames", type=bool, default=True, help="Enable frame extraction to PNG format.")
    parser.add_argument("--source_folder", type=str, default='data_processing/raw_data/', help="Source folder path.")
    parser.add_argument("--video_target_folder", type=str, default='data_processing/specified_formats/videos/videos_25fps/', help="Target folder path for videos.")
    parser.add_argument("--audio_target_folder", type=str, default='data_processing/specified_formats/audios/audios_16k/', help="Target folder path for audios.")
    parser.add_argument("--frames_target_folder", type=str, default='data_processing/specified_formats/videos/video_frames/', help="Target folder path for video frames.")
    
    args = parser.parse_args()

    if args.convert_video:
        convert_video_to_25fps(args.source_folder, args.video_target_folder)
    if args.convert_audio:
        convert_audio_to_16k(args.source_folder, args.audio_target_folder)
    if args.extract_frames:
        extract_frames_to_png(args.video_target_folder, args.frames_target_folder)

if __name__ == "__main__":
    main()
