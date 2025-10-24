import contextlib
import os
import time
from dataclasses import dataclass

import torch
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.models.base import BaseModel


@dataclass
class ImageToTextInput:
    imgs: list[str]
    prompt: str


@dataclass
class ImageToTextOutput:
    text: str


def clean_cuda_state():
    """彻底清理 CUDA 状态"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        torch.cuda._lazy_init()


class Qwen2_5Model(BaseModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        compile_model: bool = False,
        compile_mode: str = "default",
        do_warmup: bool = True,
    ) -> None:
        super().__init__()

        # 支持通过环境变量指定本地模型路径，优先级: 环境变量 > 传入参数
        model_path = os.getenv("QWEN_MODEL_PATH", model_name)

        print(f"[Qwen2_5Model] 加载模型: {model_path}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            # local_files_only=True, cache_dir=os.getenv("HF_HOME")
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            # local_files_only=True, cache_dir=os.getenv("HF_HOME")
        )

        # 修复 padding 方向：decoder-only 模型需要左侧填充
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        print(f"[Qwen2_5Model] 模型加载完成，使用设备: {self.device}")
        if compile_model and torch.cuda.is_available():
            self.model = torch.compile(self.model, mode=compile_mode, fullgraph=True)
            if do_warmup:
                self.warm_up()

    def template_input(self, prompt: str, images: list[str]) -> list:
        message = [{"role": "user", "content": []}]

        message[0]["content"].append({"type": "text", "text": prompt})
        for img in images:
            message[0]["content"].append({"type": "image", "image": img})

        return message

    def warm_up(self):
        print("[Qwen2_5_VLModel] 开始预热...")
        _ = self.batch_infer(
            [
                ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
                ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
            ]
        )
        print("预热完成")

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
        return ImageToTextOutput(text=output_text[0])

    @torch.inference_mode()
    def batch_infer(
        self,
        inputs: list[ImageToTextInput],
        max_new_tokens: int = 128,
        enable_profiler: bool = False,
        profiler_output: str = "qwen2_5_trace.json",
    ) -> list[ImageToTextOutput]:
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
            batch_messages = [self.template_input(input.prompt, input.imgs) for input in inputs]

            batch_text = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)

            batch_inputs = self.processor(
                text=batch_text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 调试确认padding情况
            print(f"Batch中的样本数: {len(batch_inputs['input_ids'])}")
            for i, (ids, mask) in enumerate(
                zip(batch_inputs["input_ids"], batch_inputs["attention_mask"])
            ):
                actual_tokens = mask.sum().item()  # 实际非padding的token数
                print(f"  样本{i + 1}: 总长度={len(ids)}, 有效tokens={actual_tokens}")

            generated_ids = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                # do_sample=False,
            )

            # 调试信息：检查生成的长度
            print(f"图片数量: {len(image_inputs) if image_inputs else 0}")
            if image_inputs:
                for i, img in enumerate(image_inputs):
                    if hasattr(img, "shape"):
                        print(f"  图片{i + 1} shape: {img.shape}")
                    elif isinstance(img, str):
                        print(f"  图片{i + 1} 是路径: {img}")

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            end_time = time.time()
            print(f"batch_infer 用时：{end_time - start_time}")

        if enable_profiler:
            profiler.export_chrome_trace(profiler_output)

        return [ImageToTextOutput(text=text) for text in output_text]


def main():
    # 运行前设置路径: export QWEN_MODEL_PATH="/home/lzx/projects/hunyuan-service
    # /models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots
    # /66285546d2b821cf421d4f5eb2576359d3770cd3"

    batch_inputs = [
        # ImageToTextInput(imgs=["outputs/sample.png"], prompt="描述这张图片"),
        # ImageToTextInput(imgs=["outputs/image_0.png"], prompt="图片中写了什么？"),
        ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
        ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
        # ImageToTextInput(imgs=["outputs/sample_large1.png"], prompt="描述这张图片"),
    ]

    tem_model = Qwen2_5Model(do_warmup=False, compile_model=False)
    _ = tem_model.batch_infer(batch_inputs, max_new_tokens=100, enable_profiler=False)
    del tem_model
    clean_cuda_state()

    model = Qwen2_5Model(do_warmup=False, compile_model=True)
    batch_outputs = model.batch_infer(batch_inputs, max_new_tokens=100, enable_profiler=False)

    print("\n批量推理结果:")
    for i, output in enumerate(batch_outputs):
        print(f"  [{i + 1}] {output.text}")


if __name__ == "__main__":
    main()
