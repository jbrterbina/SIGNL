import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.utils import dropout_edge

import numpy as np
import random

import wandb


class GCN(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, p=0.0):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gconv = nn.ModuleList()
        self.conv1d = nn.ModuleList()
        self.batch_norms_gconv = nn.ModuleList()
        self.layer_norms_gconv = nn.ModuleList()
        self.batch_norms_conv1d = nn.ModuleList()
        self.layer_norms_conv1d = nn.ModuleList()
        self.do = p
        self.fcn = nn.ModuleList()
        self.fc_res = nn.ModuleList()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        for i in range(num_layers):
            in_features = hidden_dim // (2**i)
            out_features = hidden_dim // (2 ** (i + 1))
            self.gconv.append(GCNConv(in_features, out_features))
            self.batch_norms_gconv.append(nn.BatchNorm1d(out_features))
            self.layer_norms_gconv.append(nn.LayerNorm(out_features))
            self.conv1d.append(
                nn.Conv1d(
                    out_features,
                    out_features,
                    1,
                    bias=True,
                    groups=out_features // num_heads,
                )
            )
            self.batch_norms_conv1d.append(nn.BatchNorm1d(out_features))
            self.layer_norms_conv1d.append(nn.LayerNorm(out_features))
            self.fc_res.append(nn.Linear(in_features, out_features))
            self.fcn.append(
                nn.Sequential(
                    nn.Conv1d(out_features, out_features, 1, stride=1, padding=0),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Conv1d(out_features, out_features, 1, stride=1, padding=0),
                    nn.BatchNorm1d(out_features),
                )
            )

    def forward(self, x, edge_index):
        residual = x
        for i in range(self.num_layers):
            x = self.gconv[i](x, edge_index)
            x = F.relu(x)
            x = x.unsqueeze(2)
            x = self.conv1d[i](x)
            x = self.batch_norms_conv1d[i](x)
            x = x.squeeze(2)
            residual = self.fc_res[i](residual)
            x = x + residual
            residual = x
            x = self.fcn[i](x.unsqueeze(2)).squeeze(2) + residual
            residual = x
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        p=0.1,
        num_layers=4,
        k=2,
        num_channel=16,
        spec_size=(20, 20),
        drop_edge=False,
        gaussian_noise=False,
        node_feature_masking=False,
    ):
        super(Encoder, self).__init__()
        h, w = spec_size
        self.num_nodes = h
        self.to_node_patch = nn.Sequential(
            Rearrange("b c h w -> (b h) (c w)", c=num_channel, w=w),
        )
        self.k = k
        self.gnn = GCN(hidden_dim=num_channel * w, num_layers=num_layers, p=p)
        self.drop_edge = drop_edge
        self.gaussian_noise = gaussian_noise
        self.node_feature_masking = node_feature_masking

    def add_gaussian_noise(self, x, std=0.3):
        noise = torch.randn_like(x) * std
        return x + noise

    def mask_node_features(self, x, mask_rate=0.5):
        mask = torch.rand(x.size()) > mask_rate
        return x * mask.to(x.device)

    def forward(self, x):
        bs, _, _, _ = x.size()
        x = self.to_node_patch(x)
        batch = torch.arange(bs).repeat_interleave(self.num_nodes).to(x.device)
        edge_index = knn_graph(
            x.float(), k=self.k, batch=batch, loop=False, cosine=True
        )
        if self.drop_edge:
            edge_index, edge_mask = dropout_edge(
                edge_index, p=0.5, force_undirected=True
            )
        if self.gaussian_noise:
            x = self.add_gaussian_noise(x)
        if self.node_feature_masking:
            x = self.mask_node_features(x)
        x = self.gnn(x, edge_index)
        x = rearrange(x, "(b n) d -> b (n d)", b=bs)
        return x


class Stem(nn.Module):
    def __init__(self, index_factor=4, in_dim=1, out_dim=16):
        super().__init__()
        factors = [
            [[2, 6, 8], [1, 6, 9]],
            [[2, 4, 8], [2, 3, 6]],
            [[2, 4, 6], [1, 3, 9]],
            [[2, 4, 4], [1, 3, 6]],
            [[2, 2, 4], [1, 3, 3]],
        ]
        spatial_factors = factors[index_factor][0]
        temporal_factors = factors[index_factor][1]
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 8, 3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_dim // 8),
            nn.ReLU(),
            nn.Conv2d(
                out_dim // 8,
                out_dim // 4,
                3,
                stride=(spatial_factors[0], temporal_factors[0]),
                padding=1,
            ),
            nn.BatchNorm2d(out_dim // 4),
            nn.ReLU(),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            nn.ReLU(),
            nn.Conv2d(
                out_dim // 2,
                out_dim,
                3,
                stride=(spatial_factors[1], temporal_factors[1]),
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(
                out_dim,
                out_dim,
                3,
                stride=(spatial_factors[2], temporal_factors[2]),
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class GCS_enc_trn(pl.LightningModule):
    def __init__(self, config, sweep=False):
        super().__init__()

        num_channel = 32
        num_layers = 5
        self.split_size = 216

        possible_node = [4, 6, 8, 12, 24]
        num_node = possible_node[config["model"]["num_patches_id"]]
        k = config["model"]["num_k"]
        self.p1 = config["model"]["dropout"]

        hidden_dim = num_channel * num_node
        flat_dim = (hidden_dim // (2**num_layers)) * num_node

        self.p1 = config["model"]["dropout"]
        self.encoder_t = Encoder(
            num_channel=num_channel,
            num_layers=num_layers,
            k=k,
            spec_size=(num_node, num_node),
            drop_edge=config["model"]["de"],
            gaussian_noise=config["model"]["gn"],
            node_feature_masking=config["model"]["fm"],
        )
        self.encoder_f = Encoder(
            num_channel=num_channel,
            num_layers=num_layers,
            k=k,
            spec_size=(num_node, num_node),
            drop_edge=config["model"]["de"],
            gaussian_noise=config["model"]["gn"],
            node_feature_masking=config["model"]["fm"],
        )
        self.stem = Stem(out_dim=num_channel)
        self.g = nn.Sequential(
            nn.Linear(flat_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 80),
        )

        self.sweep = sweep
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def alignment_loss(self, u, v, temperature=1.0):
        dot_product = torch.sum(u * v, dim=1)
        norm_u = torch.norm(u, dim=1)
        norm_v = torch.norm(v, dim=1)
        cosine_similarity = dot_product / (norm_u * norm_v)
        scaled_similarity = cosine_similarity / temperature
        align_loss = -scaled_similarity
        return align_loss.mean()

    def forward(self, x):
        with torch.no_grad():
            if self.training:
                rand1 = random.randint(0, 184)
                rand2 = random.randint(0, 184)
                x1 = x[..., rand1 : rand1 + self.split_size]
                x2 = x[..., rand2 : rand2 + self.split_size]
            else:
                x1 = x[..., : self.split_size]
                x2 = x[..., 184:]

        x1 = x1.unsqueeze(1)
        x1 = self.stem(x1)
        x2 = x2.unsqueeze(1)
        x2 = self.stem(x2)

        x1_t = F.dropout(self.encoder_t(x1), p=self.p1, training=self.training)
        x1_f = F.dropout(
            self.encoder_f(rearrange(x1, "b c t f -> b c f t")),
            p=self.p1,
            training=self.training,
        )
        x1 = torch.cat((x1_t, x1_f), dim=1)
        x1_g = self.g(x1)

        x2_t = F.dropout(self.encoder_t(x2), p=self.p1, training=self.training)
        x2_f = F.dropout(
            self.encoder_f(rearrange(x2, "b c t f -> b c f t")),
            p=self.p1,
            training=self.training,
        )
        x2 = torch.cat((x2_t, x2_f), dim=1)
        x2_g = self.g(x2)

        return x1_g, x2_g, x1, x2

    def training_step(self, batch):
        x, _, _, _ = batch
        x1_g, x2_g, x1, x2 = self(x)
        enc_sim = self.alignment_loss(x1, x2)
        loss = self.alignment_loss(x1_g, x2_g)
        self.train_step_outputs.append(
            {"loss": loss.detach().cpu(), "enc_sim": enc_sim.detach().cpu()}
        )
        return loss

    def validation_step(self, batch):
        x, _, _, _ = batch
        x1_g, x2_g, x1, x2 = self(x)
        enc_sim = self.alignment_loss(x1, x2)
        loss = self.alignment_loss(x1_g, x2_g)
        self.validation_step_outputs.append(
            {"loss": loss.detach().cpu(), "enc_sim": enc_sim.detach().cpu()}
        )

    def test_step(self, batch):
        x, _, _, _ = batch
        x1_g, x2_g, x1, x2 = self(x)
        enc_sim = self.alignment_loss(x1, x2)
        loss = self.alignment_loss(x1_g, x2_g)
        self.test_step_outputs.append(
            {"loss": loss.detach().cpu(), "enc_sim": enc_sim.detach().cpu()}
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001
        )
        return [optimizer]

    def on_train_epoch_end(self):
        all_loss = [output["loss"] for output in self.train_step_outputs]
        enc_sim = [output["enc_sim"] for output in self.train_step_outputs]
        avg_loss = np.mean(all_loss)
        enc_sim = np.mean(enc_sim)
        self.log("train_alignment_loss", avg_loss)
        if self.sweep:
            wandb.log({"train_alignment_loss": avg_loss, "train_enc_sim": enc_sim})
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        all_loss = [output["loss"] for output in self.validation_step_outputs]
        enc_sim = [output["enc_sim"] for output in self.validation_step_outputs]
        avg_loss = np.mean(all_loss)
        enc_sim = np.mean(enc_sim)
        self.log("val_alignment_loss", avg_loss)
        if self.sweep:
            wandb.log({"val_alignment_loss": avg_loss, "val_enc_sim": enc_sim})
        self.validation_step_outputs.clear()

    def on_test_end(self):
        all_loss = [output["loss"] for output in self.test_step_outputs]
        enc_sim = [output["enc_sim"] for output in self.test_step_outputs]
        avg_loss = np.mean(all_loss)
        enc_sim = np.mean(enc_sim)
        if self.sweep:
            wandb.log({"test_alignment_loss": avg_loss, "test_enc_sim": enc_sim})
        self.test_step_outputs.clear()


class GCS_cls(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_channel = 32
        num_layers = 5
        self.split_size = 216

        possible_node = [4, 6, 8, 12, 24]
        num_node = possible_node[config["model"]["num_patches_id"]]
        k = config["model"]["num_k"]
        self.p1 = config["model"]["dropout"]

        hidden_dim = num_channel * num_node
        flat_dim = (hidden_dim // (2**num_layers)) * num_node
        self.encoder_t = Encoder(
            num_channel=num_channel,
            num_layers=num_layers,
            k=k,
            spec_size=(num_node, num_node),
        )
        self.encoder_f = Encoder(
            num_channel=num_channel,
            num_layers=num_layers,
            k=k,
            spec_size=(num_node, num_node),
        )
        self.stem = Stem(
            index_factor=config["model"]["num_patches_id"], out_dim=num_channel
        )
        self.fcn = nn.Sequential(
            nn.Linear(flat_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 80),
        )

    def forward(self, x):
        with torch.no_grad():
            if self.training:
                rand1 = random.randint(0, 184)
                rand2 = random.randint(0, 184)
                x1 = x[..., rand1 : rand1 + self.split_size]
                x2 = x[..., rand2 : rand2 + self.split_size]
            else:
                x1 = x[..., : self.split_size]
                x2 = x[..., 184:]

        x1 = x1.unsqueeze(1)
        x1 = self.stem(x1)

        x1_t = F.dropout(self.encoder_t(x1), p=self.p1, training=self.training)
        x1_f = F.dropout(
            self.encoder_f(rearrange(x1, "b c t f -> b c f t")),
            p=self.p1,
            training=self.training,
        )

        y_hat = torch.cat((x1_t, x1_f), dim=1)
        y_hat = self.fcn(y_hat)

        return y_hat, 0


class Spectrogram(nn.Module):
    def __init__(
        self,
        spec="mel",
        emphasis=True,
        band_trimming=False,
        normalise="n",
        specaugment=False,
        filter_type="n",
        log_transform="n",
        noise=False,
        sample_rate=16000,
    ):
        super().__init__()
        self.spec = spec
        self.with_emphasis = emphasis
        self.band_trimming = band_trimming
        self.normalise = normalise
        self.specaugment = specaugment
        self.sample_rate = sample_rate
        self.frame_length = int(0.032 * 16000)
        self.frame_step = int(0.016 * 16000)
        self.window_fn = torch.hamming_window(window_length=self.frame_length)

    def forward(self, x):
        return x[:, :, :400], x[:, :, :400]
