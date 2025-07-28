import numpy as np
import torch
import re


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def shap_model_creator(model, disease_ids, person_tokens_ids, person_ages, device):
    """
    Creates a pseudo model that returns only logits for specified tokens.
    Needed for SHAP values, otherwise the SHAP visualisation is too huge.
    """
    def f(ps):
        xs = []
        as_ = []

        for p in ps:
            if len(p) == 0:
                print('No tokens found??')
                raise
            p = list(map(int, p))
            new_tokens = []
            new_ages = []
            for num, (masked, value, age) in enumerate(zip(p, person_tokens_ids, person_ages)):
                if num == 0:
                    new_ages.append(age)
                    if masked == 10000:
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_tokens.append(value)

            x = (torch.tensor(new_tokens, device=device)[None, ...])
            a = (torch.tensor(new_ages, device=device)[None, ...])

            xs.append(x)
            as_.append(a)

        max_length = max([x.shape[-1] for x in xs])

        xs = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs]
        as_ = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in as_]

        x = torch.cat(xs)
        a = torch.cat(as_)

        with torch.no_grad():
            probs = model(x, a)[0][:, -1, disease_ids].detach().cpu().numpy()
        return probs

    return f