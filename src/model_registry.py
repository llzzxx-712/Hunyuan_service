from src.models.base import BaseModel
from src.models.hunyuan import HunYuanModel
from src.models.qwen2_5 import Qwen2_5Model

MODEL_REGISTRY = {  # 模型注册表，字典类型，传入模型类型，返回模型类
    "text_to_image": HunYuanModel,
    "image_to_text": Qwen2_5Model,
}


def get_model(model_type: str) -> BaseModel:
    cls = MODEL_REGISTRY.get(model_type)  # 根据模型名称获取模型类
    if not cls:
        raise ValueError(f"Unsupported model type: {model_type}")
    return cls()
