# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: delphi
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn

class TypeAwareEmbeddingProjector(nn.Module):

    def __init__(self, d_ext, d_model, type_to_ids: dict):
        super().__init__()
        self.projectors = nn.ModuleDict({
            type_name: nn.Linear(d_ext, d_model)
            for type_name in type_to_ids
        })
        self.token_to_type = {} 
        for type_name, ids in type_to_ids.items():
            for token_id in ids:
                self.token_to_type[token_id] = type_name

    def forward(self, input_ids, ext_embeddings):
        """
        input_ids: (seq_len,)
        ext_embeddings: dict {token_id: embedding tensor of dim d_ext}
        """
        projected = []
        for token_id in input_ids:
            token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
            ext_embed = ext_embeddings[token_id]  # tensor (d_ext,)
            type_name = self.token_to_type[token_id]
            projector = self.projectors[type_name]
            projected.append(projector(ext_embed))
        return torch.stack(projected, dim=0)  # (seq_len, d_model)


# %%
TypeAwareEmbeddingProjector()
