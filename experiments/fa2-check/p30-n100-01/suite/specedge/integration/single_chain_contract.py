"""Model-free contracts shared by the single-chain client and target."""

SINGLE_CHAIN_ID = "single_chain_gamma4_gamma8_v1"
ALLOWED_GAMMAS = (4, 8)


def validate_chain_layout(input_ids, position_ids, cache_seq_indices, parent_indices,
                          *, gamma, max_len):
    """Require one root followed by exactly gamma contiguous chain candidates."""
    gamma = int(gamma)
    if gamma not in ALLOWED_GAMMAS:
        raise ValueError("Single-chain gamma must be 4 or 8")
    inputs = list(input_ids)
    positions = list(position_ids)
    slots = list(cache_seq_indices)
    parents = list(parent_indices)
    if len(inputs) != gamma + 1 or len(positions) != gamma + 1 or len(slots) != gamma + 1:
        raise ValueError("Actual wire inputs must equal gamma + 1")
    if len(parents) != gamma:
        raise ValueError("Actual candidates must equal gamma")
    root = int(slots[0])
    expected = list(range(root, root + gamma + 1))
    if root < 0 or root + gamma >= int(max_len):
        raise ValueError("Single chain exceeds context capacity")
    if positions != expected or slots != expected or parents != expected[:-1]:
        raise ValueError("Single-chain positions, KV slots and parents must be contiguous")
    return root


def scratch_slots(*, root, logical_width, physical_width, max_len):
    """Reserve unique padding slots above all live prefix and candidate slots."""
    count = int(physical_width) - int(logical_width)
    if count < 0 or logical_width <= 0:
        raise ValueError("Invalid logical/physical graph widths")
    slots = list(range(int(max_len) - count, int(max_len)))
    if slots and slots[0] < int(root) + int(logical_width):
        raise ValueError("No disjoint scratch capacity for graph padding")
    return slots


def accepted_chain_length(proposals, predictions):
    """Only the matching prefix contributes KV; matches after a miss do not."""
    proposals, predictions = list(proposals), list(predictions)
    if len(predictions) != len(proposals) + 1:
        raise ValueError("One prediction is required for every real input slot")
    accepted = 0
    for proposed, predicted in zip(proposals, predictions):
        if int(proposed) != int(predicted):
            break
        accepted += 1
    return accepted


def clipped_chunk_length(tokens, remaining, eos_token_ids):
    """Clip at the output cap and the first canonical EOS, including that EOS."""
    if int(remaining) <= 0:
        return 0, False
    eos = set(map(int, eos_token_ids))
    clipped = list(tokens)[:int(remaining)]
    for index, token in enumerate(clipped):
        if int(token) in eos:
            return index + 1, True
    return len(clipped), False
