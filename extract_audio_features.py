import argparse
import os
import numpy as np
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def main(args):
    if not torch.cuda.is_available() and args.computed_device == 'cuda':
        print('CUDA is not available on this device. Switching to CPU.')
        args.computed_device = 'cpu'
    
    device = torch.device(args.computed_device)
    model = HubertModel.from_pretrained(args.model_path).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    model.feature_extractor._freeze_parameters()
    model.eval()

    os.makedirs(args.audio_feature_saved_path, exist_ok=True)

    for wavfile in tqdm(os.listdir(args.audio_dir_path)):
        npy_save_path = os.path.join(args.audio_feature_saved_path, os.path.splitext(os.path.basename(wavfile))[0] + '.npy')
        
        if os.path.exists(npy_save_path):
            continue

        audio, sr = librosa.load(os.path.join(args.audio_dir_path, wavfile), sr=16000)
        input_values = feature_extractor(audio, sampling_rate=16000, padding=True, do_normalize=True, return_tensors="pt").input_values
        input_values = input_values.to(device)
        ws_feats = []
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
            for i in range(len(outputs.hidden_states)):
                ws_feats.append(outputs.hidden_states[i].detach().cpu().numpy())
            ws_feat_obj = np.array(ws_feats)
            ws_feat_obj = np.squeeze(ws_feat_obj, 1)

            if args.padding_to_align_audio:
                ws_feat_obj = np.pad(ws_feat_obj, ((0, 0), (0, 1), (0, 0)), 'edge')
            np.save(npy_save_path, ws_feat_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio features using a pre-trained HuBERT model.")
    parser.add_argument("--model_path", type=str, default='weights/chinese-hubert-large', help="Path to the pre-trained model weights.")
    parser.add_argument("--audio_dir_path", type=str, default='./audio_samples/raw_audios/', help="Directory containing raw audio files.")
    parser.add_argument("--audio_feature_saved_path", type=str, default='./audio_samples/audio_features/', help="Directory where extracted audio features will be saved.")
    parser.add_argument("--computed_device", type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to compute the audio features on. Use 'cuda' for GPU or 'cpu' for CPU.")
    parser.add_argument("--padding_to_align_audio", type=bool, default=True, help="Whether to pad the audio to align features.")
    args = parser.parse_args()
    main(args)

