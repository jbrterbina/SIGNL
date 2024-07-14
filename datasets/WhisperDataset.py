import numpy as np
import torch.utils.data as data
import torch
import pandas as pd
import gzip
from collections import defaultdict


class WhisperDataset(data.Dataset):
    def __init__(self, partition, dataset="asv", sample_ratio=1):
        super(WhisperDataset, self).__init__()

        self.partition = partition
        self.features = []

        if self.partition == "train":
            protocol_dir = "./datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
            feature_dir = "./datasets/LA/ASVspoof2019_LA_train/npy"
            codecs = ["flac", "mp3", "m4a"]
        elif self.partition == "dev":
            protocol_dir = "./datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
            feature_dir = "./datasets/LA/ASVspoof2019_LA_dev/npy"
            codecs = ["flac", "mp3", "m4a"]
        elif self.partition == "eval":
            if dataset == "asv21":
                protocol_dir = (
                    "./datasets/LA/ASVspoof2021_DF_cm_protocols/trial_metadata.txt"
                )
                feature_dir = "./datasets/LA/ASVspoof2021_DF_eval/npy"
            else:
                protocol_dir = (
                    "D:/Experiment/Audio Dataset/In The Wild Dataset/flac/meta.csv"
                )
                feature_dir = "D:/Experiment/Audio Dataset/In The Wild Dataset/npy"
            codecs = ["flac"]

        self.sysid_dict_asv = {
            "bonafide": 1,  # Bonafide speech
            "spoof": 0,  # Spoofed signal
        }

        self.sysid_dict_itw = {
            "bona-fide": 1,  # Bonafide speech
            "spoof": 0,  # Spoofed signal
        }

        print("Reading ", protocol_dir)
        if dataset == "asv":
            protocol_lines = open(protocol_dir).readlines()
        else:
            protocol_lines = open(protocol_dir).readlines()[1:]

        # data_by_label = defaultdict(list)
        for codec in codecs:
            for protocol_line in protocol_lines:
                if dataset == "asv":
                    tokens = protocol_line.strip().split(" ")
                    file_name = tokens[1]
                    attack_id = tokens[3]
                    label = self.sysid_dict_asv[tokens[4]]
                elif dataset == "asv21":
                    tokens = protocol_line.strip().split(" ")
                    file_name = tokens[1]  # Assuming the file name is the first token
                    attack_id = tokens[4]
                    label = self.sysid_dict_asv[tokens[5]]
                else:
                    tokens = protocol_line.strip().split(",")
                    file_name = tokens[0][:-4]
                    attack_id = tokens[1]
                    label = self.sysid_dict_itw[tokens[2]]

                    # The protocols look like this (ASV):
                    #  [0]      [1]       [2][3]  [4]
                    # LA_0070 LA_D_7622198 - - bonafide

                    # The protocols look like this (ITW):
                    #  [0]         [1]        [2]
                    # 0.wav    Alec Guiness  spoof

                feature_path = f"{feature_dir}{codec}/{file_name}.npy.gz"
                # data_by_label[label].append(
                #         (feature_path, file_name, attack_id, label)
                #     )

                self.features.append((feature_path, file_name, attack_id, label))

        if partition != "eval":
            df = pd.DataFrame(
                self.features,
                columns=["feature_path", "file_name", "attack_id", "label"],
            )

            # Determine the minimum count among all labels
            min_count = df["label"].value_counts().min()

            # Sample equal number of items from each label
            sampled_df = (
                df.groupby("label")
                .apply(lambda x: x.sample(min_count))
                .reset_index(drop=True)
            )

            # Convert back to list of tuples
            sampled_data = [tuple(row) for row in sampled_df.to_records(index=False)]

            self.features = sampled_data

    def load_tensor_compressed(self, feature_path):
        with gzip.open(feature_path, "rb") as f:
            np_array = np.load(f)
        return torch.from_numpy(np_array.astype(np.float32))

    def __getitem__(self, index):
        feature_path, file_name, attack_id, label = self.features[index]
        feature = self.load_tensor_compressed(feature_path)
        return feature, file_name, attack_id, label

    def __len__(self):
        return len(self.features)
