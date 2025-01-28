import os
import shutil
import glob
import numpy as np
from funasr import AutoModel

class EmotionVectorExtractor:
    def __init__(self, model_id="iic/emotion2vec_plus_large", hub="ms"):
        """
        Initializes the EmotionVectorExtractor with the specified model.

        Args:
            model_id (str): The identifier for the emotion2vec model.
            hub (str): The hub to use for model loading ("ms" or "hf").
        """
        self.model = AutoModel(model=model_id, hub=hub)
        self.model_name = model_id.split("/")[-1]
        print(f"Loaded model: {model_id} from {hub}")

    def extract_emotion_vector(self, wav_file, output_dir="./temp_emotion_outputs", granularity="frame"):
        """
        Extracts emotion vectors from a given audio file.

        Args:
            wav_file (str): Path to the input WAV file.
            output_dir (str): Directory to store intermediate outputs.
            granularity (str): Granularity of extraction ("utterance" or "frame").

        Returns:
            np.ndarray: The extracted emotion features.
        """
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Audio file not found: {wav_file}")

        print(f"Processing file: {wav_file}")
        rec_result = self.model.generate(
            wav_file,
            output_dir=output_dir,
            granularity=granularity,
            extract_embedding=True
        )

        if rec_result and 'feats' in rec_result[0]:
            return rec_result[0]['feats']
        else:
            raise ValueError("Failed to extract emotion features from the audio file.")

    def save_to_npy(self, features, output_path):
        """
        Saves the extracted emotion features to a .npy file.

        Args:
            features (np.ndarray): The emotion features to save.
            output_path (str): Path to the output .npy file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        # print(f"Emotion features saved to {output_path}")

    def process_folder(self, wav_folder_path):
        """
        Processes an audio file to extract emotion features and save them to a .npy file.

        Args:
            wav_folder_path (str): Path to the input WAV folder.
        """
        audio_files = glob.glob(os.path.join(wav_folder_path, "**", "*.wav")) + glob.glob(os.path.join(wav_folder_path, "*.wav"))
        for file_path in audio_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dir_path = os.path.dirname(file_path)
            output_file = os.path.join(dir_path, f"{base_name}--{self.model_name}.npy")

            if os.path.exists(output_file):
                print(f"Skipping existing file: {output_file}")
                continue
            features = self.extract_emotion_vector(file_path)
            self.save_to_npy(features, output_file)
        if os.path.exists("./temp_emotion_outputs"):
            shutil.rmtree("./temp_emotion_outputs")
            print(f"Folder './temp_emotion_outputs' and its contents have been deleted.")
        else:
            print(f"Folder './temp_emotion_outputs' does not exist.")
        print("All audio files processed.")

# Example usage
if __name__ == "__main__":
    extractor = EmotionVectorExtractor()
    wav_folder_path = "/home/ubuntu/DATA/"
    extractor.process_folder(wav_folder_path)
