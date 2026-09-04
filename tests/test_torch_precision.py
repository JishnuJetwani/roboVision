import torch
from robovision.torch_precision import configure_policy_precision


def test_precision_configuration_is_explicit_and_idempotent():
    original = (torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        first = configure_policy_precision()
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
        assert configure_policy_precision() == first
    finally:
        torch.backends.cudnn.allow_tf32 = original[0]
        torch.set_float32_matmul_precision(original[1])
