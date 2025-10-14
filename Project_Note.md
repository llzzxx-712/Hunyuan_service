# 项目笔记

## 项目准备

    在WSL2安装Python 3.10和UV（命令：`curl -LsSf https://astral.sh/uv/install.sh | sh`）。

```
# 然后创建项目目录
mkdir my-new-project
cd my-new-project

# 初始化UV项目
uv init
```

    运行`uv sync`安装依赖。

    然后手动编辑pyproject.toml：打开pyproject.toml，修改（这些都是固定模板）：

```
[project]
name = "my-new-project"                # 改成你的项目名
version = "0.1.0"
description = "My awesome API"         # 写描述
authors = [
    { name = "Your Name", email = "your@email.com" },  # 改成你的信息
]
requires-python = ">=3.10"             # 确认Python版本
dependencies = []                      # 初始为空
```

    随后添加依赖

```
# 添加核心依赖
uv add fastapi uvicorn pydantic
# 添加开发依赖
uv add --group dev pytest ruff pre-commit
```

**关于pyproject.toml**

    dependencies中的内容会自动生成(uv sync 时) 和更新（使用uv add/remove xx )

    uv.lock 文件会自动更新，不要手动编辑。

    pyproject.toml 要用到什么工具就添加什么配置，基本不用自己写，从现有模板中复制并简单调整即可（大多数时候似乎调整都不用）

### Ruff 配置

    安装Ruff（`uv add ruff`），配置pre-commit hook（在.pre-commit-config.yaml中添加）。

**安装Ruff并配置pre-commit hook具体步骤**

- 运行`uv add ruff --dev`（添加为开发依赖）。
  
- 运行`uv add pre-commit --dev`（如果未安装）。
  
- 编辑`.pre-commit-config.yaml`（如果不存在，创建它，复制标准配置：添加repos如ruff的hook）。示例内容（参考Ruff文档）：
  
  ```
  repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.5
    hooks:
    - id: ruff
      args: [--fix, --exit-non-zero-on-fix]
    - id: ruff-format
  ```
  
- 运行`pre-commit install`（安装hook）。
  
- 测试：修改一个文件（如添加空行到README.md），运行`git add .`然后`git commit -m "test"`，观察Ruff是否自动格式化。
  
- 如果文件不存在或出错，检查git status（如果git_status显示很多untracked files，先`git init`和`git add .`初始化仓库）。
  

对于 RUFF 和 pre-commit hook 的详细内容可见 RUFF_PRECOMMIT_GUIDE.md

**ruff 使用相关笔记**

    提交前常用 `ruff check --fix .` 来检查并修复所有文件格式问题，最后的 `.` 可以换成具体文件。实际过程中可能要多次执行3、4两步。

```
ruff check --fix .
ruff format
git add . #将修改的文件从工作区放入暂存区
git commit -m "test" #如果失败则多次使用git add .
```

    Git 的三个区域:

```
工作区          暂存区          仓库
(Working)      (Staging)      (Repository)
   │              │              │
   │  git add     │  git commit  │
   │─────────────>│─────────────>│
   │              │              │
   │  git diff    │              │
   │<─────────────│              │
   │              │              │
   │         git diff --cached   │
   │              │<─────────────│
   │              │              │
   │         git diff HEAD       │
   │<────────────────────────────│
```

### Fastapi

（在使用方法的时候按需 import）

在src文件夹下创建 main.py, 创建 fastapi 变量 ``app = fastapi.FastAPI(title="Hunyuan-service", lifespan=lifespan)``

在 ``if __name__ == "__main__":`` 中使用 ``uvicorn.run(app, host="0.0.0.0", port=8000)`` 来使用, ``host="0.0.0.0"`` 表示外部可以访问，port是端口号。

    接下来写 lifespan，规定项目启动、结束等几个阶段的工作。yield用来分隔运行前后：

```
@asynccontextmanager
async def lifespan(app: FastAPI):
yield
```

    然后写对应http方法，比如写post。括号内的内容为路径。

```
@app.post("/infer")
async def infer(req: TextToImageRequest)
prompt = req.prompt
```

    TextToImageRequest 是自定义的一个类，用来使用 pydantic，BaseModel是 Pydantic 提供的基类，用于数据验证和序列化。以下面格式定义了这个类之后，FastAPI 会自动将请求体的 JSON 转为这个类的实例。

```
class TextToImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
```

当客户端发送：

```
{
  "prompt": "a cat",
  "width": 512
}
```

    FastAPI 会：

    解析 JSON，验证 prompt 是字符串 ✓，验证 width 是整数 ✓，height 使用默认值 1024

然后创建 TextToImageRequest 对象传给你的函数

    我在函数里直接用 req.prompt, req.width, req.height 就行了。

**完善lifespan：**

    在全局空间中：

```
MODEL_TYPE = os.getenv("MODEL_TYPE", "text2image").lower()
model = None
```

    通过第一个函数设置了 MODEL_TYPE 的默认值为 text2image，同时如果有环境变量则将其设置为对应环境变量值并转化为小写。

**关于 await / async ：**

```
# 规则1: 普通函数 → 直接调用
def normal_func():
    return "hello"

result = normal_func()  # ✓ 直接调用

# 规则2: async 函数 → 需要 await
async def async_func():
    return "hello"

result = await async_func()  # ✓ 必须 await

# 规则3: 在 async 函数中可以调用普通函数
async def caller():
    result = normal_func()  # ✓ 没问题
    result2 = await async_func()  # ✓ 也没问题
```

**什么时候需要 async**

```
# ✓ 需要 async：网络请求
async def fetch_data():
    response = await httpx.get("https://api.example.com")
    return response.json()

# ✓ 需要 async：文件 I/O (异步版本)
async def read_file():
    async with aiofiles.open("file.txt") as f:
        return await f.read()

# ✗ 不需要 async：CPU 密集计算
def calculate():
    return sum(range(1000000))

# ✗ 不需要 async：简单逻辑
def get_config():
    return {"model": "flux"}
```

### 队列控制

    在全局变量中创建队列 `` queue = asyncio.Queue(maxsize=1)`` 因为本项目中一次只执行一项任务

    然后在我理解的顺序中，是 http 方法（比如这里的POST）中接收到请求，然后写一个enqueue函数将请求加入队列，同时在lifespan中会设置项目启动前执行任务worker_loop，在这里面持续等待请求，有请求就发动处理。

`` asyncio.create_task(worker_loop())``

**起点：post**

    post操作中大模型相关的使用我还没学到，所以先借用参考项目的代码

```
input_data = TextToImageInput(prompt=input_data.get("prompt"))
output_data = await enqueue(model.infer, input_data)
```

**enqueue函数：**

    然后现在我先记着``enqueue(model.infer, input_data)`` 对应三个参数，enqueue函数可以这样定义： ``async def enqueue_task(func, *args, **kwargs):``

    里面定义 future 变量，用来准备接收处理结果，然后将这些内容打包加入队列，worker_loop检测到队列内有元素会进行处理。

```
async def enqueue_task(func, *args, **kwargs):
    fut = asyncio.Future()
    await queue.put((func, args, kwargs, fut))
    return await fut
```

**worker_loop函数：**

    项目启动时开始执行并一直执行，只要发现队列内有元素就拿出进行处理。

处理逻辑并不复杂，从队列中拿出元素，分一个线程用来处理，将结果放回 future 变量中。

```
async def worker_loop():
    while True:
        func, args, keyargs, fut = await queue.get()
        try:
            result = await asyncio.to_thread(func, *args, **keyargs)
            fut.set_result(result)
        except Exception as e:
            fut.set_exception(e)
        finally:
            queue.task_done()
```

    finally应该是try完成后执行的内容，``queue.task_done()`` 会更新任务计数器，使得项目可以用 `` await queue.join() `` 来等待所有任务结束。

***args 和 **kwargs 解释：**

```
# *args: 接收任意数量的位置参数（打包成 tuple）
# **kwargs: 接收任意数量的关键字参数（打包成 dict）

def example(*args, **kwargs):
    print(f"args = {args}")      # tuple
    print(f"kwargs = {kwargs}")  # dict

example(1, 2, 3, name="Alice", age=25)
# 输出:
# args = (1, 2, 3)
# kwargs = {'name': 'Alice', 'age': 25}
```

## 模型处理

**抽象基类 BaseModel**

    示例中首先实现了抽象基类 ``BaseModel`` ，使用 `@abstractmethod` 定义抽象方法，其子类必须实现。

    项目中类里使用的都是实例方法，应该就对应java中类里面的一般方法（不用任何@修饰）python中这类方法都以 self 参数为开头。

```
from abc import ABC, abstractmethod
from typing import TypeVar

import torch # 导入torch库，用于设备管理、模型编译优化、其他控制等

InputType = TypeVar("InputType") # 括号中的内容是该泛型的名称，用来代替通用名称 T
OutputType = TypeVar("OutputType")


class BaseModel(ABC):
    def __init__(self):
        self.device = self._get_device() # 获取设备，用于模型推理

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @abstractmethod
    async def infer(self, input: InputType) -> OutputType: 
        pass # 抽象方法，输入一个泛型 InputType，输出一个泛型 OutputType

```

**@dataclass**

    接下来进入具体模型，首先使用了 @dataclass 快速定义了两个简单的输入输出类。用法：

```
# 传统写法
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

# dataclass 简化写法
@dataclass
class Point:
    x: int
    y: int
    # 自动生成 __init__, __repr__, __eq__ 等方法
```

**创建模型类：** 

    创建本次要执行的模型类，通过括号放入父类来完成继承

``class FluxModel(BaseModel): ``

    接着实现其构造函数，设置参数及其默认值，使用时想要传入对应参数就用 ``(变量名=xx)`` 来传入，其他参数使用默认值。

    其内部首先调用父类的构造函数，设置 device 变量。随后使用 from_pretrained，实现了：

- 从 HuggingFace Hub 下载模型
  
- 自动处理配置文件、权重文件
  
- 缓存到本地（~/.cache/huggingface/）
  

    不同模型可以查看官网来得知其对应用法。比如其中 torch_dtype=torch.float16 就是设置模型每个参数的字节数（精度）

    不过这里按照我的理解 self.model 应该是类中创建的 model 变量在实例中的使用，但是变量为什么会有 ``.to()`` 方法？（``self.model.to(self.device)``)

    下面 if 中的部分应该是模型性能测试，这个后面再学。

### 模型推理函数 infer

    flux_1.py中的 ``@torch.inference_mode() def infer `` 似乎只是设置 from_pretrained 使用的参数，我先重点看更复杂的 qwen2.5 中的内容。不过我不是很理解 @torch.inference_mode() 的作用。

    首先前面的 processor 可以将普通的输入转化为图片可以理解的编码，例如：

```
processor = AutoProcessor  # 自动选择合适的 processor
├── Tokenizer      # 处理文本
│   └── "你好" → [101, 872, 123, 102]
└── ImageProcessor # 处理图像
    └── PIL.Image → torch.Tensor (3, 224, 224)
```

    infer 的定义中传入两个参数 input 和 max_token，输出格式为 output 泛型。

**模板构建：**

    processor 需要传入特定格式的 messages —— 必须是一个字典的列表，字典中两个键名称严格是"role"和"content"。但是content的值可以是文本也可以是一个列表，例如：

```
"role": "user",
"content": [
      {"type": "image", "image": "cat.jpg"},
      {"type": "text", "text": "这是什么动物？"}
]
```

在当前的需求中，对象为user，传入多张图片和一个提示词，我可以写出如下代码:

```
def template_input(self, prompt: str, images: list[str]) -> list:
    messages = [{"role": "user", "content": []}]
    
    messages[0]["content"].append({"type": text, "text": prompt})
    for img in range(images):
        messages[0]["content"].append({"type": image, "image": img})
    
    return messages
```

**格式化对话文本：**

```
text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
```

通过这个函数将上述 input 形式变为以下格式：

```
messages = [{"role": "user", "content": "描述这张图片"}]
——>
"<|im_start|>user\n描述这张图片<|im_end|>\n<|im_start|>assistant\n"
```

`tokenize=False` 表示返回原始文本字符串，而不是分词后的token ID列表（因为后面还要和图片一起处理）

`add_generation_prompt=True` 在对话末尾添加生成提示符，告诉模型这是需要生成回复的位置

**提取实际图像和视频信息：**

```
image_inputs, video_inputs = process_vision_info(messages)
# process_vision_info提取后:
image_inputs = [PIL.Image对象, PIL.Image对象, ...]  # 实际图像数据
```

**编码阶段（张量）**

```
inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

# 处理后
inputs = {
    'input_ids': tensor([[101, 872, 123, ...]]),      # 文本 tokens
    'attention_mask': tensor([[1, 1, 1, ...]]),       # 注意力掩码
    'pixel_values': tensor([[[[0.5, 0.3, ...]]]])    # 图像像素值
}
```

`.to(self.device) `则是指将上述张量移动到指定设备（GPU/CPU）

**generate 生成**

```
generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
```

`**inputs` 将字典解包为关键字参数 - 等价于：model.generate(input_ids=tensor1, attention_mask=tensor2, images=tensor3)

可能可以用到的其他参数：

```
model.generate(
    input_ids,           # 输入token ID
    attention_mask,      # 注意力掩码
    max_new_tokens=128,  # 最大生成token数
    temperature=1.0,     # 温度参数
    do_sample=True,      # 是否采样
    top_p=0.9,          # nucleus采样参数
    top_k=50,           # top-k采样参数
    repetition_penalty=1.1,  # 重复惩罚
    pad_token_id=...,   # 填充token ID
    eos_token_id=...,   # 结束token ID
)
```

**裁剪输入：**

```
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
```

裁剪作用：

```
# 模型输入（input_ids）
[101, 872, 123, 456]  # "这是什么动物？"

# 模型输出（generated_ids）
[101, 872, 123, 456, 789, 234, 567]  # "这是什么动物？这是一只猫"
#                    ^^^^^^^^^^^^^ 这部分是新生成的

# 裁剪后（只保留新生成的）
[789, 234, 567]  # "这是一只猫"
```

**解码**

```
output_text = self.processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
```

作用：把 token ID 转回文本

```
# 输入
[789, 234, 567]
# 输出
"这是一只猫"
```

- skip_special_tokens=True: 跳过 <|im_end|> 等特殊标记
  
- clean_up_tokenization_spaces=False: 保留原始空格