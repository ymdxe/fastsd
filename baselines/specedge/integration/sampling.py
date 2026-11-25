"""Target logit selection for the adapted verifier only."""

import torch


def probabilities(logits, temperature, top_p, top_k):
    if temperature == 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1)
    scores = logits.float() / temperature
    width = scores.shape[-1]
    if 0 < top_k < width:
        scores, indices = scores.topk(top_k, dim=-1)
    elif top_p < 1:
        scores, indices = scores.sort(descending=True, dim=-1)
    else:
        indices = None
    probabilities = scores.softmax(dim=-1)
    if top_p < 1:
        probabilities.masked_fill_(probabilities.cumsum(dim=-1) - probabilities >= top_p, 0)
    if indices is not None:
        probabilities = torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, indices, probabilities)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def select_tokens(logits, temperature, top_p, top_k):
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = probabilities(logits, temperature, top_p, top_k)
    return torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(logits.shape[:-1])


def rejection_selection(target, draft, proposed, generator):
    """Encode accept/reject choices for the official linear-tree commit logic."""
    target_mass = target[:-1].gather(-1, proposed[:, None]).squeeze(-1)
    draft_mass = draft.gather(-1, proposed[:, None]).squeeze(-1)
    if not (torch.isfinite(draft).all() and (draft >= 0).all() and (draft_mass > 0).all()
            and torch.allclose(draft.sum(-1), torch.ones_like(draft_mass), atol=1e-5)):
        raise ValueError("invalid draft probability rows")
    accept = torch.rand(proposed.shape, device=target.device, generator=generator) < (target_mass / draft_mass)
    residual = (target[:-1] - draft).clamp_min(0)
    residual = torch.where(accept[:, None], target[:-1], residual)
    replacements = torch.multinomial(residual, 1, generator=generator).squeeze(-1)
    bonus = torch.multinomial(target[-1], 1, generator=generator)
    return torch.cat((torch.where(accept, proposed, replacements), bonus))
