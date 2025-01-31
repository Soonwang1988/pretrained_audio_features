import os
import torch
import glob
import numpy as np
from tqdm import tqdm
import librosa
import torchaudio
from transformers import (
    Wav2Vec2Model, WavLMModel, HubertModel, Data2VecAudioModel, WhisperForAudioClassification,
    AutoFeatureExtractor, WhisperFeatureExtractor
)
try:
    from .fairseq_hubert import FairseqHubert
except:
    from fairseq_hubert import FairseqHubert
import argparse
from colorama import Fore, Style, init

init(autoreset=True)


class AudioFeatureExtractor:
    def __init__(self, model_path="pretrained_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.audio_sampling_rate = 16000

        self.models_info = {
            "wavlm-base": {"source":"microsoft/wavlm-base","feature_dim":768},

            "wav2vec2-base": {"source":"facebook/wav2vec2-base","feature_dim":768},
            "wav2vec2-large": {"source":"facebook/wav2vec2-large","feature_din":1024},
            "wav2vec2-mms-1b-all":{"source":"facebook/mms-1b-all","feature_din":1280},

            "hubert-base": {"source":"https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt","feature_din":768},
            "hubert-large": {"source":"https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt","feature_dim":1024},

            "data2vec-base": {"source":"facebook/data2vec-audio-base","feature_dim":768},
            "data2vec-large": {"source":"facebook/data2vec-audio-large","feature_dim":1024},

            "whisper-large-v3": {"source":"openai/whisper-large-v3","feature_dim":1280},
            "whisper-tiny": {"source":"openai/whisper-tiny","feature_dim":384},
        }

    def get_info(self):
        return self.models_info.get(self.model_name, None)["source"]

    def load_model(self, model_name):
        self.model_name = model_name
        source = self.get_info()
        if not source:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if "whisper" in source:
            self.processor = WhisperFeatureExtractor.from_pretrained(source)
            self.model = WhisperForAudioClassification.from_pretrained(source, cache_dir=self.model_path)
            self.feature_func = self.extract_whisper_features
            self.model.to(self.device).eval()
        elif "hubert" in source:
            self.model = FairseqHubert(source, os.path.join(self.model_path, self.model_name + '.pt'), output_norm=False,
                                  freeze=True, freeze_feature_extractor=True)
            self.model.model.to(self.device).eval()
            self.feature_func = self.extract_fairseq_feature
        else:
            self.processor = AutoFeatureExtractor.from_pretrained(source, cache_dir=self.model_path)
            self.model = {
                "wav2vec2": Wav2Vec2Model,
                "wavlm": WavLMModel,
                "hubert": HubertModel,
                "data2vec": Data2VecAudioModel
            }.get(self.model_name.split("-")[0]).from_pretrained(source, cache_dir=self.model_path)
            self.feature_func = self.extract_huggingface_feature

            self.model.to(self.device).eval()

    def load_audio_file(self, file_path, db=0):
        waveform, sample_rate = librosa.load(file_path, sr=None, mono=False)
        waveform = librosa.effects.preemphasis(waveform * (10.0 ** (db / 20.0)))
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)
        if sample_rate != self.audio_sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.audio_sampling_rate)(
                torch.tensor(waveform))
        else:
            waveform = torch.Tensor(waveform)
        return waveform

    def extract_whisper_features(self, model, audio):
        input_features = self.processor(audio, sampling_rate=self.audio_sampling_rate,
                                        return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        with torch.no_grad():
            outputs = model(input_features, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def extract_fairseq_feature(self, model, audio):
        audio = audio.view(1, -1)
        audio = audio.to(self.device)
        out = model.extract_features(audio)
        return out

    def extract_huggingface_feature(self, model, audio):
        inputs = self.processor(audio, sampling_rate=self.audio_sampling_rate, return_tensors="pt").input_values.to(
            self.device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
        return outputs.last_hidden_state

    def process_audio_in_chunks(self, wav, chunk_size, buffer_size, db=0):
        """
        Process audio in chunks with buffer zones and combine the features.

        Parameters:
        wav (str): Path to the audio file.
        model (torch.nn.Module): Whisper model for feature extraction.
        processor: Whisper processor for audio preprocessing.
        chunk_size (int): Size of each chunk in samples.
        buffer_size (int): Size of the buffer to add before and after each chunk in samples.

        Returns:
        None: Saves the combined features as a .npy file.
        """

        source = self.load_audio_file(wav, db)  # Load the audio file as a tensor
        total_samples = source.shape[-1]
        feat_fps = 50

        combined_features = []
        chunk_size = chunk_size * self.audio_sampling_rate
        buffer_size = buffer_size * self.audio_sampling_rate

        for start in range(0, total_samples, chunk_size):
            # Define the range for the chunk with buffer
            end = start + chunk_size

            buffer_start = max(0, start - buffer_size)
            buffer_end = min(total_samples, end + buffer_size)
            actual_end_buffer = buffer_end - end

            # Extract the chunk with buffer
            chunk_with_buffer = source[buffer_start:buffer_end]

            # Add silence padding for start or end if necessary
            if start == 0:
                padding = torch.zeros(buffer_size)
                chunk_with_buffer = torch.cat((padding, chunk_with_buffer), dim=-1)
            if end >= total_samples:
                padding = torch.zeros(end + buffer_size - buffer_end)
                actual_end_buffer = end + buffer_size - buffer_end
                chunk_with_buffer = torch.cat((chunk_with_buffer, padding), dim=-1)

            # Preprocess the chunk and pass it through the model
            features = self.feature_func(self.model, chunk_with_buffer)

            # Determine valid feature range (exclude buffer regions)
            valid_start = int(buffer_size * feat_fps / self.audio_sampling_rate)
            valid_end = int((chunk_with_buffer.shape[0] - actual_end_buffer) * feat_fps / self.audio_sampling_rate)
            valid_features = features[:, valid_start:valid_end, :]

            combined_features.append(valid_features.detach().cpu().numpy())

        # Combine all features along the time dimension
        combined_features = np.concatenate(combined_features, axis=1)
        # print("Audio Length:", total_samples / self.audio_sampling_rate)
        # print("Feature Length:", combined_features.shape[1] / 50, combined_features.shape)
        return combined_features

    def save_features(self, features, output_path):
        np.save(output_path, features)

    def process_audio_directory(self, input_dir, db_gains=[0, 2, 4, 8, -2, -4, -8]):
        audio_files = glob.glob(os.path.join(input_dir, "**",  "*.wav")) + glob.glob(os.path.join(input_dir, "*.wav"))

        for file_path in tqdm(audio_files, desc="Processing audio files"):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dir_path = os.path.dirname(file_path)
            for gain_db in tqdm(db_gains, desc=f"Processing {base_name}", leave=False):
                output_file = os.path.join(dir_path,
                                           f"{base_name}_{self.model_name}_{gain_db}dB_features.npy") if gain_db != 0 else os.path.join(
                    dir_path, f"{base_name}--{self.model_name}.npy")

                if os.path.exists(output_file):
                    continue

                features = self.process_audio_in_chunks(file_path, 5, 1, gain_db)
                self.save_features(features, output_file)
                # print(f"Saved features with {gain_db} dB gain to {output_file}, shape:{features.shape}")
        print("All audio files processed.")


def main():
    """
    Audio Feature Extraction Script

    This script extracts audio features using a specified pre-trained model. 
    To run this script, execute the following command in the terminal:

    Example:
        python audio_feature_extractor.py \
            --model_name "wav2vec2-mms-1b-all" \
            --model_save_path "/home/ubuntu/pretrained_models/" \
            --input_dir "/home/Documents/audio"

    Arguments:
        --model_name: Name of the pre-trained model to use for feature extraction 
                      (e.g., "wav2vec2-mms-1b-all").
        --model_save_path: Path to save or load the pre-trained model 
                           (e.g., "/home/ubuntu/pretrained_models/").
        --input_dir: Directory containing the audio files to process 
                     (e.g., "/home/Documents/audio").

    Make sure to update the paths and arguments as per your environment before running the script.
    """
    parser = argparse.ArgumentParser(description="Extract Audio features with dB augmentation.")
    parser.add_argument("--model_name", required=True, help="Audio Model Name.")
    parser.add_argument("--model_save_path", default="./pretrained_models", help="Path to save the model.")
    parser.add_argument("--input_dir", required=True, help="Directory containing audio files.")
    parser.add_argument("--chunk_duration", type=int, default=5, help="Duration of audio chunks in seconds.")
    parser.add_argument("--buffer_duration", type=int, default=1, help="Buffer duration in seconds around each chunk.")
    parser.add_argument("--db_gains", type=int, nargs="+", default=[0, 2, 4, 8, -2, -4, -8],
                        help="List of dB gains for augmentation.")

    args = parser.parse_args()

    feature_extractor = AudioFeatureExtractor()

    feature_extractor.load_model(args.model_name)
    feature_extractor.process_audio_directory(
        input_dir=args.input_dir,
        db_gains=args.db_gains
    )


if __name__ == "__main__":
    
    main()
