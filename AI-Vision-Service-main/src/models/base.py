from abc import ABC, abstractmethod
from typing import TypeVar

import torch  # 导入torch库，用于设备管理、模型编译优化、其他控制等

InputType = TypeVar("InputType")  # 括号中的内容是该泛型的名称，用来代替通用名称 T
OutputType = TypeVar("OutputType")


class BaseModel(ABC):
    def __init__(self):
        self.device = self._get_device()  # 获取设备，用于模型推理

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @abstractmethod
    async def infer(self, input: InputType) -> OutputType:
        pass  # 抽象方法，输入一个泛型 InputType，输出一个泛型 OutputType
