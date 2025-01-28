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
   cd <pretrained_audio_features
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface

Run the script using the following command:

```bash
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

### Example

```bash
python audio_feature_extractor.py \
    --model_name "wav2vec2-base" \
    --model_save_path "./pretrained_models" \
    --input_dir "./audio_samples" \
    --chunk_duration 5 \
    --buffer_duration 1 \
    --db_gains 0
```

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
   - Applies dB gain augmentations.

3. **Feature Extraction**
   - Processes audio in chunks with buffer zones.
   - Extracts features using the selected model.

4. **Save Features**
   - Saves extracted features as `.npy` files in the same directory as the input audio files.

## Class Structure

### `AudioFeatureExtractor`

#### Methods

- `__init__(model_path)`: Initializes the extractor with model path and default sampling rate.
- `load_model(model_name)`: Loads the specified pre-trained model.
- `load_audio_file(file_path, db)`: Loads and preprocesses audio files.
- `process_audio_in_chunks(wav, chunk_size, buffer_size, db)`: Processes audio in chunks with buffer zones.
- `save_features(features, output_path)`: Saves extracted features to a `.npy` file.
- `process_audio_directory(input_dir, db_gains)`: Processes all audio files in the specified directory.

### `main()`
The main entry point for command-line usage.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to raise issues or submit pull requests for improvements or additional features.

