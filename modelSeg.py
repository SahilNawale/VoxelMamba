from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from block import Block
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.InstanceNorm1d(64)
      self.bn2 = nn.InstanceNorm1d(128)
      self.bn3 = nn.InstanceNorm1d(1024)
      self.bn4 = nn.InstanceNorm1d(512)
      self.bn5 = nn.InstanceNorm1d(256)


   def forward(self, input):
      # input.shape == (bs,3,n)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))

      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init

      final = torch.bmm(torch.transpose(input,1,2), matrix).transpose(1,2)
      return final

# transform = Tnet(k=3)
# x = torch.randn(1,3,256)
# y = transform(x)
# y.shape

def embed(pt_cloud, H):
    """
    Embeds a batch of point clouds.

    Args:
        pt_cloud (torch.Tensor): A batch of point clouds of shape [B, N, 3], where B is the batch size and N is the number of points.
        H (int): The desired number of bins or voxels.

    Returns:
        torch.Tensor: A tensor of shape [B, N, 4], where the first 3 dimensions are the normalized coordinates and the last dimension is the count.
    """
    B, N, _ = pt_cloud.shape

    # Find the min and max values for each dimension
    x_min, x_max = pt_cloud[:, :, 0].min(), pt_cloud[:, :, 0].max()
    y_min, y_max = pt_cloud[:, :, 1].min(), pt_cloud[:, :, 1].max()
    z_min, z_max = pt_cloud[:, :, 2].min(), pt_cloud[:, :, 2].max()

    # Apply the normalization formula
    pts = pt_cloud.clone()
    pts[:, :, 0] = (pts[:, :, 0] - x_min) // ((x_max - x_min) / H)
    pts[:, :, 1] = (pts[:, :, 1] - y_min) // ((y_max - y_min) / H)
    pts[:, :, 2] = (pts[:, :, 2] - z_min) // ((z_max - z_min) / H)

    # Find the unique points and their counts
    unique_pts, counts = torch.unique(pts, dim=1, return_counts=True)


    # Combine the unique points and their counts
    result = torch.cat((unique_pts, counts.unsqueeze(dim=0).unsqueeze(dim=2)), dim=-1)
    print(result.shape)
    return result

class PointCloudEmbedding(nn.Module):
    def __init__(self, H):
        super(PointCloudEmbedding, self).__init__()
        self.H = H
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU()

    def forward(self, pt_cloud):
        B, N, _ = pt_cloud.shape
        embedded_pts = embed(pt_cloud, self.H)
        x = embedded_pts.permute(0, 2, 1)  # [B, 4, N]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x.permute(0, 2, 1)  # [B, N, 512]

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            grid_size:int,
            output_classes:int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.transform = Tnet(k=3)
        self.embedding = PointCloudEmbedding(grid_size)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_classes)
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, pts,inference_params=None):

        pts = pts.transpose(1, 2)
        pts = self.transform(pts).transpose(1,2)
        print(pts.shape)
        pts = self.embedding(pts)
        skip = pts

        print(pts.shape)
        residual = None
        hidden_states = pts

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        hidden_states = torch.cat((hidden_states,skip),dim=-1)
        print(hidden_states.shape)
        hidden_states = self.mlp(hidden_states)
        output = F.softmax(hidden_states, dim=-1)
        _, labels = output.max(dim=-1)
        return labels


model = MixerModel(1024,16,10).cuda()
summary = torchinfo.summary(model, input_size=(1, 10000,3))
print(summary.total_mult_adds/(10**9))
print(summary)
