# Audio Feature Extractor

This repository provides a script to extract audio features using pre-trained models such as Wav2Vec2, WavLM, HuBERT, Data2Vec, and Whisper. The extracted features can be saved for further audio processing and analysis.

## Features
- Support for multiple pre-trained audio models.
- Audio feature extraction with dB gain augmentations.
- Processes audio files in chunks with buffer zones.
- Saves extracted features as `.npy` files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Soonwang1988/pretrained_audio_features.git
   cd pretrained_audio_features
   ```

2. Install the required dependencies:
   ```bash
   conda create --name aud_features python=3.10 -y
   source activate aud_features  # Use 'conda activate' if running manually
   # Install PyTorch with CUDA
   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

   # Install Fairseq
   conda install -c conda-forge fairseq==0.12.3 -y

   # Install additional dependencies
   pip install librosa==0.10.2 transformers==4.48.2 speechbrain==1.0.2
   ```

## Usage

### Command-Line Interface

Run the script using the following command:

```bash
# without db augmentation
python audio_feature_extractor.py \
    --model_name "wav2vec2-mms-1b-all" \
    --model_save_path "./pretrained_models" \
    --input_dir "./audio_files" \
    --chunk_duration 5 \
    --buffer_duration 1 \
    --db_gains 0

# if db aug required
python audio_feature_extractor.py \
    --model_name "wav2vec2-mms-1b-all" \
    --model_save_path "./pretrained_models" \
    --input_dir "./audio_files" \
    --chunk_duration 5 \
    --buffer_duration 1 \
    --db_gains 0 2 4 8 -2 -4 -8

```

### Arguments

| Argument            | Description                                                                                     | Default                 |
|---------------------|-------------------------------------------------------------------------------------------------|-------------------------|
| `--model_name`      | Name of the pre-trained model to use for feature extraction.                                    | Required                |
| `--model_save_path` | Path to save or load the pre-trained model.                                                    | `./pretrained_models`   |
| `--input_dir`       | Directory containing audio files.                                                              | Required                |
| `--chunk_duration`  | Duration of audio chunks in seconds.                                                           | `5`                     |
| `--buffer_duration` | Buffer duration in seconds around each chunk.                                                  | `1`                     |
| `--db_gains`        | List of dB gains for augmentation.                                                             | `[0, 2, 4, 8, -2, -4, -8]` |


## Models Supported

| Model Name               | Source                                    | Feature Dimension |
|--------------------------|-------------------------------------------|--------------------|
| `wavlm-base`             | `microsoft/wavlm-base`                   | 768                |
| `wav2vec2-base`          | `facebook/wav2vec2-base`                 | 768                |
| `wav2vec2-large`         | `facebook/wav2vec2-large`                | 1024               |
| `wav2vec2-mms-1b-all`    | `facebook/mms-1b-all`                    | 1280               |
| `hubert-base`            | `https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt` | 768 |
| `hubert-large`           | `https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt` | 1024 |
| `data2vec-base`          | `facebook/data2vec-audio-base`           | 768                |
| `data2vec-large`         | `facebook/data2vec-audio-large`          | 1024               |
| `whisper-large-v3`       | `openai/whisper-large-v3`                | 1280               |
| `whisper-tiny`           | `openai/whisper-tiny`                    | 384                |

## Processing Steps

1. **Load Pre-trained Model**
   The script downloads the selected model if not already present in the specified path.

2. **Audio Preprocessing**
   - Resamples audio to 16 kHz.
   - Converts stereo to mono if needed.
   - Applies dB gain augmentations if needed.

3. **Feature Extraction**
   - Processes audio in chunks with buffer zones.
   - Extracts features using the selected model.

4. **Save Features**
   - Saves extracted features as `.npy` files in the same directory as the input audio files.


## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to raise issues or submit pull requests for improvements or additional features.
