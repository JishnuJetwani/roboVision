import pytest
import torch

from robovision.models import load_target
from robovision.vision import image_moments


def test_color_moments_and_missing_detection():
    image = torch.zeros(2, 6, 96, 96)
    image[0, -2:, 30:40, 60:70] = .7
    moments = image_moments(image)
    assert moments[0, 0] > 0
    assert moments[0, 1] < 0
    assert moments[0, -1] == 1
    assert torch.count_nonzero(moments[1]) == 0


def test_incompatible_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "target.pt"
    torch.save({"version": "unrecognized"}, path)
    with pytest.raises(ValueError, match="Incompatible target checkpoint"):
        load_target(path)
