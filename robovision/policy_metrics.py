"""Physical exploration scales for force-control PPO policies."""


def physical_std(model):
    return (
        model.policy.log_std.detach().exp().cpu().numpy() * [40, 55, 45, 20, 10]
    ).tolist()


def exploration_report(model):
    if getattr(model, "checkpoint_algorithm", None) == "structured-decoupled-ppo-v1":
        from .structured_exploration import exploration_report as report

        return report(model)
    return dict(
        kind="independent Gaussian every20ms",
        physical_std=physical_std(model),
        ent_coef=model.ent_coef,
    )
