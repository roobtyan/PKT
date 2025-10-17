from typing import Any, Dict

import torch
from torch import nn


class BaseHead(nn.Module):
    """基础Head，仅约定forward接口。"""

    def forward(self, features: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, Any]:
        raise NotImplementedError
