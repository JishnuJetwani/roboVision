"""Commit-backed, bounded-storage checkpoints for trusted remote PPO training.

The caller owns the experiment directory and serializes training/checkpoint calls.
No source/parent file is modified. A generation becomes resumable only after its
files were verified and committed, then its pointer was committed. Resume loads
the algorithm directly, including optimizer states: never call prepare_model.

Environment state is intentionally not restored. Resume starts fresh episodes and
an empty rollout buffer. RNG and learned exploration tensors are restored, but
this does not claim a bit-exact continuation of the lost simulator trajectory.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch

VERSION = "training-checkpoint-store-v1"
STORE = "checkpoint-store-v1"
FILES = ("model.zip", "metadata.json", "runtime.pt")


class CheckpointError(ValueError):
    """Checkpoint integrity, provenance, or accounting validation failed."""


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(identity):
    if not isinstance(identity, dict) or not all(
        (identity.get(k) for k in ("recipe", "source", "config_sha256"))
    ):
        raise CheckpointError("identity requires recipe, source, and config_sha256")
    digest = identity["config_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((c not in "0123456789abcdef" for c in digest))
    ):
        raise CheckpointError("config_sha256 must be a lowercase SHA256 digest")
    return hashlib.sha256(_json(identity)).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise CheckpointError(f"Cannot read checkpoint JSON: {path.name}") from error


def _store(out):
    path = Path(out) / STORE
    if path.is_symlink():
        raise CheckpointError("Checkpoint store may not be a symlink")
    return path


def _generation(store, name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or (not name.startswith("gen-"))
    ):
        raise CheckpointError("Invalid generation basename")
    path = store / name
    if path.is_symlink():
        raise CheckpointError("Generation may not be a symlink")
    return path


def _pointer(store):
    path = store / "CURRENT.json"
    if not path.exists():
        return None
    data = _read(path)
    if (
        data.get("version") != VERSION
        or not isinstance(data.get("entries"), list)
        or (not data["entries"])
        or (len({e["generation"] for e in data["entries"]}) != len(data["entries"]))
    ):
        raise CheckpointError("Invalid checkpoint pointer")
    return data


def _parameter_digest(parameters):
    """Stable digest of all saved policy/optimizer tensors and scalar settings."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda x: (type(x).__name__, repr(x))):
                add(key)
                add(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for item in value:
                add(item)
        else:
            digest.update((type(value).__name__ + ":" + repr(value)).encode())
        digest.update(b"\x00")

    add(parameters)
    return digest.hexdigest()


def _settings(model):
    keys = (
        "n_steps",
        "n_envs",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "normalize_advantage",
        "use_sde",
        "sde_sample_freq",
        "critic_learning_rate",
        "actor_update_scope",
    )
    settings = {key: getattr(model, key) for key in keys if hasattr(model, key)}
    for name in ("lr_schedule", "clip_range", "clip_range_vf"):
        value = getattr(model, name, None)
        settings[name] = (
            [float(value(progress)) for progress in (0.0, 0.5, 1.0)]
            if callable(value)
            else value
        )
    return settings


def _runtime(model):
    numpy_state = np.random.get_state()
    distribution = getattr(model.policy, "action_dist", None)
    return dict(
        python=random.getstate(),
        numpy=(
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        exploration={
            name: getattr(distribution, name).detach().cpu().clone()
            for name in ("exploration_mat", "exploration_matrices")
            if isinstance(getattr(distribution, name, None), torch.Tensor)
        },
    )


def _restore_runtime(model, runtime):
    random.setstate(runtime["python"])
    state = runtime["numpy"]
    np.random.set_state((state[0], state[1].numpy(), state[2], state[3], state[4]))
    torch.set_rng_state(runtime["torch"])
    if runtime["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(runtime["cuda"]) != torch.cuda.device_count()
        ):
            raise CheckpointError(
                "Exact CUDA RNG restore requires the saved CUDA device count"
            )
        torch.cuda.set_rng_state_all(runtime["cuda"])
    distribution = getattr(model.policy, "action_dist", None)
    for name, value in runtime["exploration"].items():
        setattr(distribution, name, value.to(model.device))


def _verify(store, entry, identity_hash):
    directory = _generation(store, entry["generation"])
    manifest_path = directory / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or _hash(manifest_path) != entry["manifest_sha256"]
    ):
        raise CheckpointError("Manifest hash mismatch")
    manifest = _read(manifest_path)
    if (
        manifest.get("version") != VERSION
        or manifest.get("generation") != entry["generation"]
        or manifest.get("identity_sha256") != identity_hash
        or (set(manifest.get("files", {})) != set(FILES))
        or (manifest.get("num_timesteps") != entry["num_timesteps"])
    ):
        raise CheckpointError("Manifest provenance or counter mismatch")
    for name, digest in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _hash(path) != digest:
            raise CheckpointError(f"Checkpoint bytes corrupt: {name}")
    metadata = _read(directory / "metadata.json")
    if _identity(metadata["identity"]) != identity_hash:
        raise CheckpointError("Metadata identity mismatch")
    return (directory, manifest, metadata)


def _garbage_collect(store, pointer):
    """Only delete complete, verified, untagged generations owned by this store."""
    keep = {entry["generation"] for entry in pointer["entries"]}
    removed = []
    for directory in store.glob("gen-*"):
        if directory.name in keep or directory.is_symlink() or (not directory.is_dir()):
            continue
        try:
            manifest = _read(directory / "manifest.json")
            entry = dict(
                generation=directory.name,
                manifest_sha256=_hash(directory / "manifest.json"),
                num_timesteps=manifest["num_timesteps"],
            )
            _, verified, _ = _verify(store, entry, pointer["identity_sha256"])
            if verified.get("tagged_gate") or {p.name for p in directory.iterdir()} != {
                *FILES,
                "manifest.json",
            }:
                continue
        except (CheckpointError, KeyError, OSError):
            continue
        shutil.rmtree(directory)
        removed.append(directory.name)
    return removed


def save_checkpoint(
    model, out, metadata, commit=lambda: None, *, tagged_gate=False, retain=2
):
    """Save full algorithm state plus caller audit in an immutable generation.

    metadata requires ``identity`` and JSON-serializable ``audit``. Other fields
    (cumulative training_seconds/new_steps, original initial counter, recovery
    expenditure) remain caller-defined and are returned unchanged on load.
    ``commit`` must synchronously make filesystem writes durable remotely.
    Latest ``retain`` generations and every tagged gate survive garbage collection.
    """
    if type(retain) is not int or retain < 2:
        raise CheckpointError("Retain at least two generations")
    if not isinstance(metadata, dict) or "audit" not in metadata:
        raise CheckpointError("Metadata requires identity and audit history")
    identity_hash = _identity(metadata.get("identity"))
    _json(metadata)
    store = _store(out)
    store.mkdir(parents=True, exist_ok=True)
    previous = _pointer(store)
    if previous and previous["identity_sha256"] != identity_hash:
        raise CheckpointError(
            "Cannot replace another recipe/source/config checkpoint store"
        )
    previous_valid = []
    for old in previous["entries"] if previous else []:
        try:
            _verify(store, old, identity_hash)
            previous_valid.append(old)
        except (CheckpointError, KeyError, OSError):
            continue
    name = f"gen-{int(model.num_timesteps):012d}-{uuid.uuid4().hex}"
    directory = store / name
    directory.mkdir(exist_ok=False)
    runtime = _runtime(model)
    model.save(directory / "model.zip")
    _atomic_json(directory / "metadata.json", metadata)
    torch.save(runtime, directory / "runtime.pt")
    manifest = dict(
        version=VERSION,
        generation=name,
        identity_sha256=identity_hash,
        num_timesteps=int(model.num_timesteps),
        n_updates=int(getattr(model, "_n_updates", 0)),
        algorithm_class=type(model).__module__ + "." + type(model).__qualname__,
        parameter_sha256=_parameter_digest(model.get_parameters()),
        settings=_settings(model),
        files={filename: _hash(directory / filename) for filename in FILES},
        tagged_gate=bool(tagged_gate),
    )
    _atomic_json(directory / "manifest.json", manifest)
    entry = dict(
        generation=name,
        manifest_sha256=_hash(directory / "manifest.json"),
        num_timesteps=int(model.num_timesteps),
    )
    _verify(store, entry, identity_hash)
    commit()
    pointer = dict(
        version=VERSION,
        identity_sha256=identity_hash,
        entries=([entry] + previous_valid)[:retain],
    )
    _atomic_json(store / "CURRENT.json", pointer)
    commit()
    removed = _garbage_collect(store, pointer)
    if removed:
        commit()
    return dict(
        **entry,
        model_path=str(directory / "model.zip"),
        removed_generations=removed,
        tagged_gate=bool(tagged_gate),
    )


def begin_chunk(out, steps, commit=lambda: None, *, expected_identity=None):
    """Durably bound work at risk before one collection/update chunk begins.

    Call only after a checkpoint and before model.learn. After recovery, first
    save the restored model with updated cumulative lost-work accounting; then
    this method can begin a new chunk without erasing the old interruption.
    """
    if type(steps) is not int or steps <= 0:
        raise CheckpointError("Chunk steps must be a positive integer")
    store = _store(out)
    pointer = _pointer(store)
    if pointer is None:
        raise CheckpointError("Save an initial checkpoint before beginning a chunk")
    if (
        expected_identity is not None
        and _identity(expected_identity) != pointer["identity_sha256"]
    ):
        raise CheckpointError("Chunk identity mismatch")
    entry = pointer["entries"][0]
    _verify(store, entry, pointer["identity_sha256"])
    path = store / "INFLIGHT.json"
    if path.exists() and _read(path).get("base_generation") == entry["generation"]:
        raise CheckpointError(
            "Unresolved chunk: save recovered state/accounting before starting another"
        )
    intent = dict(
        version=VERSION,
        identity_sha256=pointer["identity_sha256"],
        base_generation=entry["generation"],
        model_steps_before=entry["num_timesteps"],
        maximum_new_steps=steps,
    )
    _atomic_json(path, intent)
    commit()
    return intent


def load_resume(
    out,
    *,
    expected_identity,
    env=None,
    device="auto",
    loader=None,
    restore_rng=True,
    resumed_env_seed=None,
):
    """Return ``(model, metadata, recovery)``; reject wrong identities/corruption.

    Falls back to the prior verified generation if the latest bytes are corrupt.
    The loader must accept SB3's env/device/force_reset arguments and restore the
    correct algorithm class. No learning/configuration overrides are passed.
    RNG is restored after loader construction. If supplied, resumed_env_seed is
    queued on the new VecEnv without reseeding global RNG, then actual queued
    worker seeds are recorded. Simulator reset happens at the next learn call.
    """
    identity_hash = _identity(expected_identity)
    store = _store(out)
    pointer = _pointer(store)
    if pointer is None:
        return None
    if pointer["identity_sha256"] != identity_hash:
        raise CheckpointError("Resume recipe/source/config identity mismatch")
    rejected = []
    selected = None
    for entry in pointer["entries"]:
        try:
            selected = (entry, *_verify(store, entry, identity_hash))
            break
        except (CheckpointError, KeyError, OSError) as error:
            rejected.append(dict(generation=entry.get("generation"), error=str(error)))
    if selected is None:
        raise CheckpointError(f"No valid checkpoint generation: {rejected}")
    entry, directory, manifest, metadata = selected
    if loader is None:
        from .policy_loading import load_grasp_policy

        loader = load_grasp_policy
    model = loader(directory / "model.zip", env=env, device=device, force_reset=True)
    if (
        type(model).__module__ + "." + type(model).__qualname__
        != manifest["algorithm_class"]
        or int(model.num_timesteps) != manifest["num_timesteps"]
        or int(getattr(model, "_n_updates", 0)) != manifest["n_updates"]
        or (_settings(model) != manifest["settings"])
        or (_parameter_digest(model.get_parameters()) != manifest["parameter_sha256"])
    ):
        raise CheckpointError(
            "Loader changed algorithm, tensors, optimizer state, counters, or settings"
        )
    model._last_obs = None
    model._last_original_obs = None
    model._last_episode_starts = np.ones(model.n_envs, dtype=bool)
    model.rollout_buffer.reset()
    if restore_rng:
        runtime = torch.load(
            directory / "runtime.pt", map_location="cpu", weights_only=True
        )
        _restore_runtime(model, runtime)
    environment = model.get_env()
    actual_seeds = None
    if resumed_env_seed is not None:
        if (
            type(resumed_env_seed) is not int
            or not 0 <= resumed_env_seed < 2**32 - model.n_envs
        ):
            raise CheckpointError("Explicit valid resumed environment seed required")
        if environment is None:
            raise CheckpointError("Cannot queue resumed seeds without an environment")
        actual_seeds = list(environment.seed(resumed_env_seed))
    elif environment is not None and hasattr(environment, "_seeds"):
        actual_seeds = list(environment._seeds)
    latest_steps = pointer["entries"][0]["num_timesteps"]
    lost_lower = max(0, latest_steps - manifest["num_timesteps"])
    lost_upper = lost_lower
    intent_path = store / "INFLIGHT.json"
    intent = _read(intent_path) if intent_path.exists() else None
    if intent:
        if (
            intent.get("version") != VERSION
            or intent.get("identity_sha256") != identity_hash
        ):
            raise CheckpointError("In-flight provenance mismatch")
        if intent["base_generation"] == pointer["entries"][0]["generation"]:
            if (
                intent["model_steps_before"] != latest_steps
                or type(intent["maximum_new_steps"]) is not int
                or intent["maximum_new_steps"] <= 0
            ):
                raise CheckpointError("In-flight accounting mismatch")
            lost_upper += intent["maximum_new_steps"]
    recovery = dict(
        generation=entry["generation"],
        model_path=str(directory / "model.zip"),
        fallback=bool(rejected),
        rejected_generations=rejected,
        restored_model_steps=manifest["num_timesteps"],
        latest_committed_model_steps=latest_steps,
        lost_interactions_lower_bound=lost_lower,
        lost_interactions_upper_bound=lost_upper if intent else None,
        uncommitted_work_bound_known=intent is not None,
        loss_bound_requires_begin_chunk=True,
        saved_rng_restored=bool(restore_rng),
        fresh_episodes=True,
        rollout_buffer_empty=True,
        simulator_state_restored=False,
        resumed_env_seed=resumed_env_seed,
        actual_queued_worker_seeds=actual_seeds,
        next_action="Save restored state with cumulative lost-work accounting before begin_chunk",
    )
    return (model, metadata, recovery)
