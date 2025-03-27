import torch


class CreatePatchesLayer(torch.nn.Module):
    def __init__(
            self,
            patch_size: int,
            strides: int
    ) -> None:
        super().__init__()
        self.unfold_layer = torch.nn.Unfold(
            kernel_size=patch_size, stride=strides
    )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patched_images = self.unfold_layer(images)
        return patched_images.permute((0, 2, 1))


class PatchEmbeddingLayer(torch.nn.Module):
    def __init__(
            self,
            num_patches: int,
            patch_size: int,
            embed_dim: int,
            device: torch.device
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.position_embeddings = torch.nn.Embedding(
            num_embeddings=num_patches + 1,
            embedding_dim=embed_dim
        )
        self.proj_layer = torch.nn.Linear(
            in_features=patch_size * patch_size * 3,
            out_features=embed_dim
        )
        self.cls_token = torch.nn.Parameter(
            torch.rand(1, 1, embed_dim).to(device),
            requires_grad=True
        )
        self.device = device

    def forward(
            self,
            patches: torch.Tensor
    ) -> torch.Tensor:
        batch_size = patches.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = (
            torch.arange(self.num_patches + 1)
            .to(self.device)
            .unsqueeze(0)
        )
        patches = self.proj_layer(patches)
        encoded_patches = torch.cat(
            tensors=(cls_tokens, patches),
            dim=1
        ) + self.position_embeddings(positions)
        return encoded_patches

class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            num_heads: int,
            key_dim: int,
            embed_dim: int,
            mlp_dim: int,
            dropout: int = 0.1
    ) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )
        self.multi_head_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm_2 = torch.nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embed_dim,
                out_features=mlp_dim,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                in_features=mlp_dim,
                out_features=embed_dim
            ),
            torch.nn.Dropout(dropout)
        )

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        out = self.layer_norm(inputs)
        attn_output, _ = self.multi_head_attn(
            query=out,
            key=out,
            value=out
        )
        inputs = inputs + attn_output
        out = self.layer_norm_2(inputs)
        mlp_output = self.mlp(out)
        inputs = inputs + mlp_output
        return inputs


class ViTClassifier(torch.nn.Module):
    def __init__(
            self,
            num_transformer_layers: int,
            embed_dim: int,
            num_heads: int,
            patch_size: int,
            num_patches: int,
            mlp_dim: int,
            num_classes: int,
            device: torch.device
    ) -> None:
        super().__init__()
        self.patch_layer = CreatePatchesLayer(
            patch_size=patch_size,
            strides=patch_size
        )
        self.patch_embedding_layer = PatchEmbeddingLayer(
            num_patches=num_patches,
            patch_size=patch_size,
            embed_dim=embed_dim,
            device=device
        )
        self.transformer_layers = torch.nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerBlock(
                    num_heads=num_heads,
                    key_dim=embed_dim,
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim
                )
            )
        self.mlp_block = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=embed_dim,
                out_features=mlp_dim,
                device=device
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                in_features=mlp_dim,
                out_features=embed_dim,
                device=device
            ),
            torch.nn.Dropout(
                p=0.5
            )
        )

        self.logits = torch.nn.Linear(
            in_features=embed_dim,
            out_features=num_classes
        )

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        out = self.patch_layer(inputs)
        out = self.patch_embedding_layer(out)
        for transformer_layer in self.transformer_layers:
            out = transformer_layer(out)
        out = out[:, 0, :]
        out = self.mlp_block(out)
        out = self.logits(out)
        return out

