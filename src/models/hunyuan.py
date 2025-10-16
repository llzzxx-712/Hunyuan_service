import contextlib
import os
import time
from dataclasses import dataclass

import torch
from diffusers import HunyuanDiTPipeline
from PIL import Image
from torch.profiler import ProfilerActivity, profile

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
        do_warmup: bool = False,
    ):
        super().__init__()

        # 通过环境变量指定本地模型路径
        model_path = os.getenv("HUNYUAN_MODEL_PATH", model_name)
        print(f"[HunyuanModel] 加载模型: {model_path}")

        self.pipeline = HunyuanDiTPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(self.device)

        if compile_model:
            self.pipeline.transformer.to(memory_format=torch.channels_last)
            self.pipeline.vae.to(memory_format=torch.channels_last)

            self.pipeline.transformer = torch.compile(
                self.pipeline.transformer, mode=compile_mode, fullgraph=True
            )
            self.pipeline.vae.decode = torch.compile(
                self.pipeline.vae.decode, mode=compile_mode, fullgraph=True
            )

        if do_warmup:
            self.warmup()

    @torch.inference_mode()
    def warmup(self):
        print("[HunyuanModel] 开始预热...")
        _ = self.batch_infer(
            [TextToImageInput(prompt="一个红色的矩形在黑色的背景上")],
            height=1024,
            width=768,
            num_inference_steps=15,
        )
        print("[HunyuanModel] 预热完成")

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

    @torch.inference_mode()
    def batch_infer(
        self,
        inputs: list[TextToImageInput],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        enable_profiler: bool = False,
    ) -> list[TextToImageOutput]:
        if enable_profiler:
            prof_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
            )
        else:
            prof_context = contextlib.nullcontext()

        start_time = time.time()
        with prof_context as profiler:
            batch_prompts = [input.prompt for input in inputs]
            batch_images = self.pipeline(
                prompt=batch_prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            ).images
            end_time = time.time()
            print(f"Batch inference time: {end_time - start_time:.4f} seconds")

        if enable_profiler:
            profiler.export_chrome_trace("hunyuan_trace.json")

        return [TextToImageOutput(image=image) for image in batch_images]


if __name__ == "__main__":
    # 运行前设置路径: export HUNYUAN_MODEL_PATH="/home/lzx/projects
    # /hunyuan-service/models/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled
    # /snapshots/527cf2ecce7c04021975938f8b0e44e35d2b1ed9"

    prompts = [
        "一个充满科技感的教室",
        "A beautiful sunset over the ocean",
        "一只可爱的猫咪在窗台上",
    ]
    save_path = "outputs/hunyuan_output.png"

    model = HunYuanModel(compile_model=False, do_warmup=True)

    # 单个推理
    input = TextToImageInput(prompt=prompts[0])
    output = model.infer(input, height=768, width=1024, num_inference_steps=20)
    output.image.save(save_path)
    print(f"图片已保存到: {save_path}")

    # 批量推理
    print("开始批量推理")
    batch_inputs = [TextToImageInput(prompt=prompt) for prompt in prompts]
    batch_outputs = model.batch_infer(batch_inputs, height=768, width=1024, num_inference_steps=20)
    for i, output in enumerate(batch_outputs):
        output.image.save(f"outputs/hunyuan_output_{i}.png")
    print("批量推理完成, 图片已保存到 outputs/hunyuan_output_{i}.png")
