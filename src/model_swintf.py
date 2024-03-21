import torch
import torch.nn as nn

from typing import Callable, Optional, Union
from functools import partial

try:
    from timm.models.swin_transformer import SwinTransformerBlock
    from timm.layers import PatchEmbed
    from timm.models.layers import trunc_normal_
except ImportError:
    print("\n--Need install timm package!--\n")


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        global_pool: str = "avg",
        embed_dim: int = 96,
        depths: int = 5,
        num_heads: int = 3,
        head_dim: Optional[int] = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Union[str, Callable] = nn.LayerNorm,
        weight_init: str = '',
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg")
        self.output_fmt = "NHWC"
        self.num_features = embed_dim
        # self.num_layers = len(depths)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            output_fmt="NHWC",
        )
        self.patch_grid = self.patch_embed.grid_size

        # swin transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        num_heads = [num_heads * (2 ** (i // 2)) for i in range(depths)]

        self.blocks = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=(self.patch_grid[0], self.patch_grid[1]),
                    num_heads=num_heads[i],
                    head_dim=head_dim,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depths)
            ]
        )

        self.norm = norm_layer(self.num_features)
    
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return x


if __name__ == "__main__":
    img = torch.randn(8, 9, 50, 50)
    model = SwinTransformer(
        img_size=50,
        patch_size=5,
        in_chans=9,
        embed_dim=225,
        depths=5,
        num_heads=3,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )
    out = model(img)
    print(out.shape)
