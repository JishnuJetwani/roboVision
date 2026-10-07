"""Consistent full-float32 CUDA forwards for narrow Gaussian torque policies."""

import torch


def configure_policy_precision():
    """Call before collection/evaluation, never halfway through a rollout.

    Tiny learned action standard deviations amplify batch-dependent TF32 CNN
    rounding into spurious PPO likelihood ratios. This changes arithmetic only,
    not checkpoint tensors or control semantics.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(
        cudnn_allow_tf32=False,
        matmul_allow_tf32=False,
        float32_matmul_precision="highest",
    )
