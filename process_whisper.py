import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
import librosa
from torch import Tensor
from pathlib import Path
import random
import gzip

from audiomentations import (
    Compose,
    AddColorNoise,
    ApplyImpulseResponse,
    AddBackgroundNoise,
    Mp3Compression,
)
from tqdm import tqdm
import whisper
import os
import argparse

model = whisper.load_model("tiny")

audioment1 = AddColorNoise(
    max_snr_db=40,
    min_f_decay=-2.0,
    max_f_decay=2,
    p=0.3,
)

audioment2 = ApplyImpulseResponse(
    p=0.3,
    ir_path="./archive/rirs_noises/real_rirs_isotropic_noises",
    leave_length_unchanged=True,
)

# audioment3 = AddBackgroundNoise(
#     sounds_path="./archive/musan/merged",
#     max_snr_in_db=40,
#     p=0.1,
# )

mp3_compressor = Mp3Compression(min_bitrate=16, max_bitrate=256, backend="lameenc", p=1)


def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def save_tensor_compressed(tensor, saved_dir):
    np_array = tensor.numpy().astype(np.float16)
    with gzip.open(f"{saved_dir}.npy.gz", "wb") as f:
        np.save(f, np_array)


def process_single_file(args):
    feature_path, file_name, saved_dir, partition, max_len, compress = args
    try:
        feature, sr = librosa.load(feature_path, sr=16000)
        # if partition != "eval" and random.random() < 0.5: first
        if partition != "eval":
            feature = random.choice([audioment1, audioment2])(
                samples=feature, sample_rate=sr
            )

            # if compress == "mp3":
            #     feature = mp3_compressor(samples=feature, sample_rate=sr)
        feature = pad(feature, max_len)
        mel = whisper.log_mel_spectrogram(feature).to(model.device)
        mel = mel.unsqueeze(0)

        with torch.no_grad():
            embeddings = model.embed_audio(mel).squeeze().T[..., :400]
        save_tensor_compressed(embeddings.cpu(), f"{saved_dir}/{file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")


def process_feature(
    protocol_dir, saved_dir, audio_path, partition, dataset, max_len, compress=None
):
    protocol_path = Path(protocol_dir)
    protocol_lines = (
        protocol_path.read_text().splitlines()[1:]
        if dataset != "asv"
        else protocol_path.read_text().splitlines()
    )

    tasks = []
    for protocol_line in protocol_lines:
        if dataset == "asv":
            tokens = protocol_line.strip().split(" ")
            feature_path = audio_path + tokens[1] + ".flac"
            file_name = tokens[1]
        else:
            tokens = protocol_line.strip().split(",")
            feature_path = audio_path + tokens[0][:-4] + ".flac"
            file_name = tokens[0][:-4]
        tasks.append(
            [str(feature_path), file_name, saved_dir, partition, max_len, compress]
        )

    num_processes = 5
    with Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.imap(process_single_file, tasks),
                total=len(tasks),
                desc="Processing Audio Files",
            )
        )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    max_len = 480000

    # ========Train flac
    # partition = 'train'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2019_LA_train/flac/'
    # protocol_dir = './datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    # saved_dir = './datasets/LA/ASVspoof2019_LA_train/npyflac/'
    # process_feature(protocol_dir, saved_dir, audio_path, partition, dataset, max_len)

    # ========Train mp3
    # partition = 'train'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2019_LA_train/_MP3/16/'
    # protocol_dir = './datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    # saved_dir = './datasets/LA/ASVspoof2019_LA_train/npymp3/'
    # process_feature(protocol_dir, saved_dir, audio_path, partition, dataset, max_len)

    # #========Train m4a
    # partition = 'train'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2019_LA_train/_M4A/64/'
    # protocol_dir = './datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    # saved_dir = './datasets/LA/ASVspoof2019_LA_train/npym4a/'
    # process_feature(protocol_dir, saved_dir, audio_path, partition, dataset, max_len)

    # # #========Dev
    # partition = 'dev'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2019_LA_dev/flac/'
    # protocol_dir = './datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    # saved_dir = './datasets/LA/ASVspoof2019_LA_dev/npyflac/'
    # process_feature(protocol_dir, saved_dir, audio_path, partition, dataset, max_len)

    # # #========Dev mp3
    # partition = 'dev'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2019_LA_dev/_MP3/16/'
    # protocol_dir = './datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    # saved_dir = './datasets/LA/ASVspoof2019_LA_dev/npymp3/'
    # process_feature(protocol_dir, saved_dir, audio_path, partition, dataset, max_len)

    # # #========Dev m4a
    # partition = 'dev'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2019_LA_dev/_M4A/64/'
    # protocol_dir = './datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    # saved_dir = './datasets/LA/ASVspoof2019_LA_dev/npym4a/'
    # process_feature(protocol_dir, saved_dir, audio_path, partition, dataset, max_len)

    # #========eval
    # partition = 'eval'
    # dataset = 'asv'
    # audio_path = './datasets/LA/ASVspoof2021_DF_eval/flac/'
    # protocol_dir = './datasets/LA/ASVspoof2021_DF_cm_protocols/trial_metadata.txt'
    # saved_dir = './datasets/LA/ASVspoof2021_DF_eval/npy/'

    # #========eval
    # partition = 'eval'
    # dataset = 'itw'
    # audio_path = "D:/Experiment/Audio Dataset/In The Wild Dataset/flac/"
    # protocol_dir = "D:/Experiment/Audio Dataset/In The Wild Dataset/flac/meta.csv"
    # saved_dir = "D:/Experiment/Audio Dataset/In The Wild Dataset/npy/"
