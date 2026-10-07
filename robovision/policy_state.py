"""State fingerprints used to prove evaluation leaves policies frozen."""

import hashlib
import numpy as np
import torch


def tree_hash(value):
    """Hash tensor bytes and scalar optimizer metadata without changing state."""
    import hashlib

    digest = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                digest.update(repr(k).encode())
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            digest.update(type(x).__name__.encode())
            for v in x:
                visit(v)
        else:
            digest.update(repr(x).encode())

    visit(value)
    return digest.hexdigest()


def policy_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _counters(expert):
    result = {
        name: _json(getattr(expert, name))
        for name in ("num_timesteps", "_n_updates", "_episode_num")
        if hasattr(expert, name)
    }
    buffer = getattr(expert, "rollout_buffer", None)
    if buffer is not None:
        result["rollout_buffer"] = {
            name: _json(getattr(buffer, name)) for name in ("pos", "full")
        }
    return result


def _optimizer_hash(model):
    optimizers = {}
    for prefix, owner in (("model", model), ("policy", model.policy)):
        for name, value in vars(owner).items():
            if isinstance(value, torch.optim.Optimizer):
                optimizers[f"{prefix}.{name}"] = value.state_dict()
    return tree_hash(optimizers)
