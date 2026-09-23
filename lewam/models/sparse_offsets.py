import torch


def default_sparse_offsets(action_block_size, num_transition, *, device=None) -> torch.Tensor:
    action_block_size = int(action_block_size)
    num_transition = int(num_transition)
    return torch.arange(
        0,
        action_block_size * num_transition + 1,
        action_block_size,
        device=device,
        dtype=torch.long,
    )


def sample_random_sparse_offsets(action_block_size, num_transition, *, device=None) -> torch.Tensor:
    action_block_size = int(action_block_size)
    num_transition = int(num_transition)
    action_horizon = action_block_size * num_transition
    future_offsets = torch.randperm(action_horizon, device=device, dtype=torch.long)[:num_transition] + 1
    future_offsets = future_offsets.sort().values
    return torch.cat([torch.zeros(1, device=device, dtype=torch.long), future_offsets], dim=0)


def normalize_sparse_offsets(
    sparse_offsets,
    *,
    action_block_size,
    num_transition,
    batch_size=None,
    device=None,
) -> torch.Tensor:
    action_block_size = int(action_block_size)
    num_transition = int(num_transition)
    num_sparse_tokens = num_transition + 1
    action_horizon = action_block_size * num_transition

    if sparse_offsets is None:
        sparse_offsets = default_sparse_offsets(
            action_block_size,
            num_transition,
            device=device,
        )
    else:
        sparse_offsets = torch.as_tensor(sparse_offsets, device=device, dtype=torch.long)
        if sparse_offsets.ndim == 2:
            if batch_size is not None and sparse_offsets.size(0) != int(batch_size):
                raise ValueError(
                    f"Expected sparse_offsets batch size {int(batch_size)}, got {sparse_offsets.size(0)}"
                )
            if sparse_offsets.size(1) != num_sparse_tokens:
                raise ValueError(
                    f"Expected sparse_offsets shape (B, {num_sparse_tokens}), got {tuple(sparse_offsets.shape)}"
                )
            if bool((sparse_offsets != sparse_offsets[:1]).any().item()):
                raise ValueError("Expected batch-shared sparse_offsets, got different offsets within the batch")
            sparse_offsets = sparse_offsets[0]
        elif sparse_offsets.ndim != 1:
            raise ValueError(f"Expected sparse_offsets with shape (T,) or (B,T), got {tuple(sparse_offsets.shape)}")

    if sparse_offsets.shape != (num_sparse_tokens,):
        raise ValueError(f"Expected sparse_offsets shape ({num_sparse_tokens},), got {tuple(sparse_offsets.shape)}")
    if sparse_offsets[0].item() != 0:
        raise ValueError(f"Expected sparse_offsets to start at 0, got {sparse_offsets.tolist()}")
    if bool((sparse_offsets[1:] <= sparse_offsets[:-1]).any().item()):
        raise ValueError(f"Expected strictly increasing sparse_offsets, got {sparse_offsets.tolist()}")
    if sparse_offsets[-1].item() > action_horizon:
        raise ValueError(f"Expected sparse_offsets <= horizon {action_horizon}, got {sparse_offsets.tolist()}")
    return sparse_offsets
