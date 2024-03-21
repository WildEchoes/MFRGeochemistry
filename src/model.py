"""
Multi-Modal fusion Model for the Geochemistric Data Regression.
"""

import torch
import torch.nn as nn
from functools import partial
try:
    from model_swintf import SwinTransformer
except ImportError:
    print("\n--Can't find swin transformer--\n")


class MHCA(nn.Module):
    def __init__(self, n_feats: int = 66, ratio: float = 0.5):
        """
        MHCA spatial-channel attention module.
        :param n_feats: The number of filter of the input.
        :param ratio: Channel reduction ratio.
        """
        super(MHCA, self).__init__()

        out_channels = int(n_feats // ratio)

        head_1 = [
            nn.Conv2d(
                in_channels=n_feats,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=n_feats,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        ]

        kernel_size_sam = 3
        head_2 = [
            nn.Conv2d(
                in_channels=n_feats,
                out_channels=out_channels,
                kernel_size=kernel_size_sam,
                padding=0,
                bias=True,
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=n_feats,
                kernel_size=kernel_size_sam,
                padding=0,
                bias=True,
            ),
        ]

        kernel_size_sam_2 = 5
        head_3 = [
            nn.Conv2d(
                in_channels=n_feats,
                out_channels=out_channels,
                kernel_size=kernel_size_sam_2,
                padding=0,
                bias=True,
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=n_feats,
                kernel_size=kernel_size_sam_2,
                padding=0,
                bias=True,
            ),
        ]

        self.head_1 = nn.Sequential(*head_1)
        self.head_2 = nn.Sequential(*head_2)
        self.head_3 = nn.Sequential(*head_3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_h1 = self.head_1(x)
        res_h2 = self.head_2(x)
        res_h3 = self.head_3(x)
        m_c = self.sigmoid(res_h1 + res_h2 + res_h3)
        res = x * m_c
        return res


class SpectrumHead(nn.Module):
    """SpectrumHead : Using 1D-Conv for extracting the spectrum feature from the input pixel data"""

    def __init__(
        self,
        in_chans: int = 1,
        hidden_chans: int = 16,
        pooling=nn.MaxPool1d,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_chans, hidden_chans, 3)
        self.conv2 = nn.Conv1d(hidden_chans, hidden_chans * 2, 3)
        self.conv3 = nn.Conv1d(hidden_chans * 2, hidden_chans * 4, 3)

        self.bn1 = nn.BatchNorm1d(hidden_chans)
        self.bn2 = nn.BatchNorm1d(hidden_chans * 2)
        self.bn3 = nn.BatchNorm1d(hidden_chans * 4)

        self.pool = pooling(3) or nn.AvgPool1d(3)
        self.relu = act_layer() or nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape the input tensor fit for the conv1d layer
        B, C, H, W = x.size()
        # B, C, H, W -> B, H, W, C -> B*H*W, 1, C
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 1, C)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)

        C = x.shape[1]  # 新的通道数
        # B*H*W, C, 1 -> B*H*W, C -> B, H, W, C -> B, C, H, W
        x = x.squeeze_(-1).view(B, H, W, C).permute(0, 3, 1, 2)
        return x


class MultiModalRegression(nn.Module):
    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x


class MFRGeoChemistry(nn.Module):
    def __init__(
        self,
        img_s: int = 50,
        patch_s: int = 5,
        in_c1: int = 9,
        in_c2: int = 1,
        embed_dim1: int = 225,
        embed_dim2: int = 25,
        depths: int = 5,
        num_heads: int = 3,
        window_size: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        """
        Args:
        img_s: Input image size.
        patch_s: Patch size.
        in_c1: Number of input Remote sensing image channels.
        in_c2: Number of input Ohter image channels.
        embed_dim1: Patch embedding dimension.(RS)
        embed_dim2: Patch embedding dimension.(Other)
        depths: Depth of each Swin Transformer layer.
        num_heads: Number of attention heads in different layers.
        head_dim: Dimension of self-attention heads.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        drop_rate: Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        """
        super().__init__()
        self.img_size = img_s
        self.patch_size = patch_s
        self.in_c1 = in_c1
        self.in_c2 = in_c2

        self.spectrum_head = SpectrumHead()

        common_parm = partial(
            SwinTransformer,
            img_size=img_s,
            patch_size=patch_s,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.rs_feature = common_parm(in_chans=in_c1, embed_dim=embed_dim1)
        self.ndvi_feature = common_parm(in_chans=in_c2, embed_dim=embed_dim2)
        self.magnetic_feature = common_parm(in_chans=in_c2, embed_dim=embed_dim2)
        self.dem_feature = common_parm(in_chans=in_c2, embed_dim=embed_dim2)

        self.fusion = MHCA(n_feats=in_c1 + in_c2 * 3, ratio=0.5)
        self.regression = MultiModalRegression(in_c=in_c1 + in_c2 * 3 + 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectrum_head(x[:, 0:9, :, :])

        x2 = self.reshape_out_feature(
            self.rs_feature(x[:, 0:9, :, :]), in_chan=self.in_c1
        )
        x3 = self.reshape_out_feature(
            self.ndvi_feature(x[:, 9:10, :, :]), in_chan=self.in_c2
        )
        x4 = self.reshape_out_feature(
            self.magnetic_feature(x[:, 10:11, :, :]), in_chan=self.in_c2
        )
        x5 = self.reshape_out_feature(
            self.dem_feature(x[:, 11:12, :, :]), in_chan=self.in_c2
        )

        x = torch.cat([x2, x3, x4, x5], dim=1)
        
        del x2, x3, x4, x5
        torch.cuda.empty_cache()
        
        x = self.fusion(x)

        x = torch.cat([x, x1], dim=1)
        x = self.regression(x)

        return x

    def reshape_out_feature(self, x: torch.Tensor, in_chan: int = 9) -> torch.Tensor:
        B, PX, PY, L = x.size()
        assert (
            L == (self.patch_size**2) * in_chan
        ), f"The feature dimension doesn't match with expected patch size and channel."
        x = (
            x.permute(0, 3, 1, 2)  # B, L, PX, PY
            .contiguous()
            .reshape(
                B, in_chan, self.patch_size, self.patch_size, PX, PY
            )  # B, C, H, W, PX, PY
        )
        x = (
            x.permute(0, 1, 4, 2, 5, 3)
            .contiguous()
            .reshape(B, in_chan, self.img_size, self.img_size)
        )  # B, C, PX, H, PY, W
        return x


if __name__ == "__main__":
    flag: list = [2]
    x = torch.randn(320, 12, 50, 50)

    if 1 in flag:
        """Test SpectrumHead"""
        print("---Test SpectrumHead---")
        model = SpectrumHead()

        print(f"\n{x.shape=}")
        reslut = model(x)
        print(f"{reslut.shape=}\n")

    if 2 in flag:
        """Test MFRGeoChemistry"""
        print("---Test MFRGeoChemistry---")
        model = MFRGeoChemistry()
        print(f"\n{x.shape=}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params=}")
        model = model.cuda()
        x = x.cuda()
        model.train()
        reslut = model(x)
        print(f"{reslut.shape=}\n")
