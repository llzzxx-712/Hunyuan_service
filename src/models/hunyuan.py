import os
from dataclasses import dataclass

import torch
from diffusers import HunyuanDiTPipeline
from PIL import Image

from src.models.base import BaseModel


@dataclass
class TextToImageInput:
    prompt: str


@dataclass
class TextToImageOutput:
    image: Image.Image


class HunYuanModel(BaseModel):
    def __init__(
        self,
        model_name: str = "Tencent-Hunyuan/HunyuanDiT-Diffusers",
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
    ):
        super().__init__()

        # 支持通过环境变量指定本地模型路径
        model_path = os.getenv("HUNYUAN_MODEL_PATH", model_name)

        print(f"[HunyuanModel] 加载模型: {model_path}")

        self.pipeline = HunyuanDiTPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(self.device)

        if compile_model:
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)

            self.pipeline.transformer = torch.compile(
                self.pipeline.transformer, mode="max-autotune", fullgraph=True
            )
            self.pipeline.vae.decode = torch.compile(
                self.pipeline.vae.decode, mode="max-autotune", fullgraph=True
            )

    @torch.inference_mode()
    def infer(
        self,
        input: TextToImageInput,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
    ) -> TextToImageOutput:
        supported_sizes = [  # HunyuanDiT 支持的分辨率
            (1024, 1024),
            (1280, 1280),
            (1024, 768),
            (1152, 864),
            (1280, 960),
            (768, 1024),
            (864, 1152),
            (960, 1280),
            (1280, 768),
            (768, 1280),
        ]

        if (height, width) not in supported_sizes:
            print(f"尺寸 {height}x{width} 不在支持列表中")
            print(f"支持的尺寸: {supported_sizes[:3]}... 等")
            print("模型会自动调整到最接近的支持尺寸")

        image = self.pipeline(
            prompt=input.prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        ).images[0]

        return TextToImageOutput(image=image)


if __name__ == "__main__":
    prompts = [
        "一个充满科技感的教室",
        "A beautiful sunset over the ocean",
        "一只可爱的猫咪在窗台上",
    ]
    save_path = "outputs/hunyuan_output.png"

    model = HunYuanModel(compile_model=False)
    input = TextToImageInput(prompt=prompts[0])

    output = model.infer(input, height=768, width=1024, num_inference_steps=20)
    output.image.save(save_path)
    print(f"图片已保存到: {save_path}")
