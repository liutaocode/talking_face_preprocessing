import argparse
import os
import librosa
import torch
import whisper
import pdb
from tqdm import tqdm
import numpy as np

# pip install -U openai-whisper
def main(args):
    if not torch.cuda.is_available() and args.computed_device == 'cuda':
        print('CUDA is not available on this device. Switching to CPU.')
        args.computed_device = 'cpu'
    
    device = torch.device(args.computed_device)
    encoder = whisper.load_model(name=args.model_name, device=device, download_root=args.model_prefix).encoder

    if not os.path.exists(args.audio_feature_saved_path):
        os.makedirs(args.audio_feature_saved_path)
    
    for audio_name in tqdm(os.listdir(args.audio_dir_path), desc="Processing audio files"):
        audio_path = os.path.join(args.audio_dir_path, audio_name)
        output_path = os.path.join(args.audio_feature_saved_path, os.path.splitext(audio_name)[0] + '.npy')
        if os.path.exists(output_path):
            continue

        audio, sr = librosa.load(audio_path, sr=16000)

        # you can skip too long audio to avoid OOM
        # duration = librosa.get_duration(y=audio, sr=sr)
        # if duration > 60:
        #     print(f"Skipping {audio_name} as it is longer than 1 minute.")


        audio = whisper.pad_or_trim(audio.flatten()) # as least 30s. you can slide to your specific duration at the usage.
        mel = whisper.log_mel_spectrogram(audio)
        
        output_npy = encoder(mel.unsqueeze(0).cuda()).cpu().detach().numpy()
        
        np.save(output_path, output_npy) # e.g., torch.Size([1, 1500, 1280])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio features using a pre-trained HuBERT model.")
    parser.add_argument("--model_prefix", type=str, default='weights/whispermodel', help="Download large-v2.pt to this path")
    parser.add_argument("--model_name", type=str, default='large-v2', help=".")
    parser.add_argument("--audio_dir_path", type=str, default='./audio_samples/raw_audios/', help="Directory containing raw audio files.")
    parser.add_argument("--audio_feature_saved_path", type=str, default='./audio_samples/audio_features/', help="Directory where extracted audio features will be saved.")
    parser.add_argument("--computed_device", type=str, default='cpu', choices=['cuda', 'cpu'], help="Device to compute the audio features on. Use 'cuda' for GPU or 'cpu' for CPU.")

    args = parser.parse_args()
    main(args)

