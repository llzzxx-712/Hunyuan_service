from dataclasses import dataclass

from src.models.base import BaseModel


@dataclass
class TextToImageInput:
    prompt: str
    image_size: int


class HunYuanModel(BaseModel):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    model = HunYuanModel()
    input = TextToImageInput(prompt="A beautiful girl", image_size=512)
    output = model.infer(input)
    print(output.images)
