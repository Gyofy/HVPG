import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SwinUNETR import SwinUNETR


class SwinUNETRWithEnhancedText(SwinUNETR):
    def __init__(self, *args, **kwargs):
        self.fusion_mode = kwargs.pop("fusion_mode", "concat")
        self.num_classes = kwargs.pop("num_classes", 1)
        super().__init__(*args, **kwargs)

        feature_size_local = kwargs.get("feature_size", 48)
        in_channels_per_stage = [
            feature_size_local,
            feature_size_local * 2,
            feature_size_local * 4,
            feature_size_local * 8,
            feature_size_local * 16,
        ]
        self.stage_projs = nn.ModuleList(
            [nn.Conv3d(channels, 768, kernel_size=1) for channels in in_channels_per_stage]
        )

        self.text_projection = nn.Linear(768, 768)
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536 * len(self.stage_projs), self.num_classes),
        )

    def forward(self, x, text_embeddings):
        hidden_states_out = self.swinViT(x, self.normalize)
        text_proj = self.text_projection(text_embeddings)
        text_proj_flat = text_proj.mean(dim=1, keepdim=True)

        fused_stage_vectors = []
        for proj_layer, stage_idx in zip(self.stage_projs, range(len(self.stage_projs))):
            stage_feat = hidden_states_out[stage_idx]
            stage_feat_proj = proj_layer(stage_feat)
            stage_flat = stage_feat_proj.flatten(start_dim=2).permute(0, 2, 1)

            if self.fusion_mode == "concat":
                if stage_flat.size(1) != 1:
                    stage_flat_pooled = F.adaptive_avg_pool1d(
                        stage_flat.permute(0, 2, 1), output_size=1
                    ).permute(0, 2, 1)
                else:
                    stage_flat_pooled = stage_flat
                combined = torch.cat([stage_flat_pooled, text_proj_flat], dim=1)
            else:
                q = self.q_proj(text_proj)
                k = self.k_proj(stage_flat)
                v = self.v_proj(stage_flat)
                scale = 768**0.5
                attn_logits = torch.bmm(q, k.transpose(1, 2)) / scale
                attn_weights = F.softmax(attn_logits, dim=-1)
                attended = torch.bmm(attn_weights, v)
                if attended.size(1) != 1:
                    attended = attended.mean(dim=1, keepdim=True)
                combined = torch.cat([attended, text_proj_flat], dim=1)

            fused_stage_vectors.append(combined.flatten(start_dim=1))

        multi_stage_features = torch.cat(fused_stage_vectors, dim=1)
        logits = self.out(multi_stage_features)
        return logits, None

