"""Hash-verified final-height benchmark for trusted skill-training checkpoints."""

import hashlib
import json
from pathlib import Path

VERSION = "skill-final-height-evaluation-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basename(value):
    if (
        not isinstance(value, str)
        or value in ("", ".", "..")
        or Path(value).name != value
    ):
        raise ValueError("Run and checkpoint names must be explicit basenames")
    return value


def protocol(
    seeds,
    *,
    conditions=("normal", "frozen", "black"),
    stochastic=False,
    seed_provenance=None,
):
    seeds = list(seeds)
    conditions = list(conditions)
    if (
        len(seeds) != 200
        or len(set(seeds)) != 200
        or any((type(s) is not int or not 0 <= s < 2**32 for s in seeds))
    ):
        raise ValueError("Require 200 distinct integer seeds in [0,2**32)")
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or any((c not in ("normal", "frozen", "black") for c in conditions))
    ):
        raise ValueError("Invalid image conditions")
    if type(stochastic) is not bool:
        raise ValueError("stochastic must be bool")
    return dict(
        seeds=seeds,
        conditions=conditions,
        stochastic=stochastic,
        seed_provenance=seed_provenance,
        seed_reservation_verified=False,
        environment="original JointGraspEnv(stage=3)",
        stochastic_semantics="deployment Gaussian, no training worker noise multipliers",
    )


def resolve_skill_checkpoint(volume_root, source, checkpoint):
    directory = Path(volume_root) / basename(source)
    checkpoint = basename(checkpoint)
    if not checkpoint.endswith(".zip"):
        raise ValueError("Provide explicit .zip checkpoint")
    metadata_path = directory / "summary.json"
    if not metadata_path.exists():
        metadata_path = directory / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    if checkpoint == "policy.zip":
        if metadata_path.name != "summary.json":
            raise ValueError("Final policy requires completed summary provenance")
        record = metadata
    else:
        gates = metadata.get("gates")
        if gates is None:
            gates = json.loads((directory / "gates.json").read_text())
        matches = [r for r in gates if r.get("checkpoint") == checkpoint]
        if len(matches) != 1:
            raise ValueError("Checkpoint must match one source gate record")
        record = matches[0]
    path = directory / checkpoint
    if sha256(path) != record.get("policy_sha256"):
        raise ValueError("Checkpoint hash does not match source record")
    return (path, record, metadata, metadata_path)


def evaluate_skill_hard(
    volume_root,
    source,
    checkpoint,
    seeds,
    *,
    conditions=("normal", "frozen", "black"),
    stochastic=False,
    seed_provenance=None,
    device="cuda",
):
    from .policy_loading import load_grasp_policy, checkpoint_algorithm
    from .grasp_benchmark import evaluate_hard_grasper, ENV_CONFIG

    spec = protocol(
        seeds,
        conditions=conditions,
        stochastic=stochastic,
        seed_provenance=seed_provenance,
    )
    path, record, metadata, metadata_path = resolve_skill_checkpoint(
        volume_root, source, checkpoint
    )
    before = sha256(path)
    model = load_grasp_policy(path, device=device)

    def state_hash():
        digest = hashlib.sha256()
        for name, value in sorted(model.policy.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    model_before = state_hash()
    root = Path(__file__).resolve().parents[1]
    files = [
        "assets/cup_arm.xml",
        "robovision/env.py",
        "robovision/joint_env.py",
        "robovision/grasp_reward.py",
        "robovision/grasp_benchmark.py",
        "robovision/skill_hard_evaluation.py",
        "robovision/policy_loading.py",
        "robovision/cnn.py",
        "robovision/context_exploration.py",
        "robovision/decoupled_ppo.py",
        "robovision/torch_precision.py",
    ]
    hashes = {f: sha256(root / f) for f in files}
    metadata_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    report = evaluate_hard_grasper(
        model,
        spec["seeds"],
        conditions=spec["conditions"],
        stochastic=spec["stochastic"],
        policy_sha256=before,
        training_config_sha256=metadata_hash,
        seed_provenance=seed_provenance,
    )
    model_after = state_hash()
    after = sha256(path)
    if model_before != model_after or before != after:
        raise RuntimeError("Frozen evaluation changed policy state or checkpoint")
    report.update(
        runner_version=VERSION,
        source_run=source,
        checkpoint=checkpoint,
        protocol=spec,
        original_environment=dict(ENV_CONFIG),
        checkpoint_algorithm=checkpoint_algorithm(path),
        source_record=record,
        source_metadata_file=metadata_path.name,
        source_metadata_sha256=sha256(metadata_path),
        evaluation_source_sha256=hashes,
        model_state_sha256_before=model_before,
        model_state_sha256_after=model_after,
        policy_sha256_after=after,
    )
    return report
