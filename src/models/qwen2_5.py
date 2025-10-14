import os
from dataclasses import dataclass

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.models.base import BaseModel


@dataclass
class ImageToTextInput:
    imgs: list[str]
    prompt: str


@dataclass
class ImageToTextOutput:
    result: str


class Qwen2_5Model(BaseModel):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        super().__init__()

        # 支持通过环境变量指定本地模型路径，优先级: 环境变量 > 传入参数
        model_path = os.getenv("QWEN_MODEL_PATH", model_name)

        print(f"[Qwen2_5Model] 加载模型: {model_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        print(f"[Qwen2_5Model] 模型加载完成，使用设备: {self.device}")

    def template_input(self, prompt: str, images: list[str]) -> list:
        message = [{"role": "user", "content": []}]

        message[0]["content"].append({"type": "text", "text": prompt})
        for img in images:
            message[0]["content"].append({"type": "image", "image": img})

        return message

    @torch.inference_mode()
    def infer(self, input: ImageToTextInput) -> ImageToTextOutput:
        messages = self.template_input(input.prompt, input.imgs)

        text = self.processor.apply_chat_template(  # 格式化对话文本
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return ImageToTextOutput(result=output_text[0])


if __name__ == "__main__":
    # 运行前设置路径: export QWEN_MODEL_PATH="/home/lzx/projects/hunyuan-service
    # /models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots
    # /66285546d2b821cf421d4f5eb2576359d3770cd3"
    model = Qwen2_5Model()

    images = [
        "outputs/image_0.png",
    ]

    output = model.infer(
        ImageToTextInput(imgs=images, prompt="这是一张手写字的图片，请识别图片中的文字")
    )
    print(f"\n推理结果: {output.result}")
