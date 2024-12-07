#!/usr/bin/python3

import os
import sys
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your SpeechBrain modules
import speechbrain as sb

# Path to the hyperparameter file
HPARAMS_FILE = "hparams/train_ecapa_tdnn_cs541.yaml"  # Update this path as needed

# Path to a sample WAV file for testing
TEST_WAV_PATH = "C:\\OldComputerFiles\\DeepLearningCode\\voxceleb\\vox1\\vox1_dev_wav\\wav\\id10001\\1zcIwhmdeo4\\00001.wav"  # Replace with an actual file

# Output folder for saving results
OUTPUT_FOLDER = "./distortion_tests/"


def apply_distortion(wav_tensor, lens, distortion, name, output_folder):
    """Applies a distortion to the audio and saves the result."""
    distorted_wav, _ = distortion(wav_tensor, lens)
    output_path = os.path.join(output_folder, f"{name}_distorted.wav")
    torchaudio.save(output_path, distorted_wav.cpu(), sample_rate=16000)
    print(f"Saved {name} distortion to {output_path}")


def main():
    # Load hyperparameters
    with open(HPARAMS_FILE, "r") as f:
        hparams = load_hyperpyyaml(f)

    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load the sample WAV file
    wav_tensor, sample_rate = torchaudio.load(TEST_WAV_PATH)
    wav_tensor = wav_tensor.to(torch.float32)
    lens = torch.tensor([1.0])  # Full length of the audio

    print(f"Loaded WAV file {TEST_WAV_PATH}, shape: {wav_tensor.shape}")

    # Initialize distortions
    distortions = {
        "add_noise": hparams["add_noise"],
        "add_reverb": hparams["add_reverb"],
        "drop_freq": hparams["drop_freq"],
        "drop_chunk": hparams["drop_chunk"],
        "multi_pitch_shift": hparams["multi_pitch_shift"],
        "vary_pitch_shift": hparams["vary_pitch_shift"],
    }

    # Apply and save each distortion
    for name, distortion in distortions.items():
        print(f"Applying distortion: {name}")
        apply_distortion(wav_tensor, lens, distortion, name, OUTPUT_FOLDER)

    print("Distortion testing complete!")


if __name__ == "__main__":
    main()
