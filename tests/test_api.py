#!/usr/bin/env python3
"""
Docker API 测试脚本

功能：
- 测试 Docker 容器中的图像识别 API
- 使用 outputs/sample.png 作为测试图片
- 与 src/api/single_task.py 对应

使用方法：
    python3 test_api.py
"""

import base64
import sys
from pathlib import Path

import requests


def image_to_base64_url(image_path: str) -> str:
    """
    将图片文件转换为 base64 data URL 格式

    Args:
        image_path: 图片文件路径

    Returns:
        base64 data URL 字符串
    """
    with open(image_path, "rb") as f:
        image_data = f.read()

    base64_str = base64.b64encode(image_data).decode()

    # 根据文件扩展名确定 MIME 类型
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/png")

    return f"data:{mime};base64,{base64_str}"


def test_image_recognition(
    image_path: str = "outputs/sample.png",
    prompt: str = "请详细描述这张图片的内容",
    api_url: str = "http://localhost:8000/infer",
):
    """
    测试图像识别 API

    Args:
        image_path: 测试图片路径
        prompt: 提示词
        api_url: API 地址
    """
    print("=" * 60)
    print("Docker API 测试 - 图像识别")
    print("=" * 60)
    print(f"测试图片: {image_path}")
    print(f"提示词: {prompt}")
    print(f"API 地址: {api_url}")
    print()

    # 1. 检查图片文件
    if not Path(image_path).exists():
        print("✗ 错误：图片文件不存在")
        print(f"  路径: {image_path}")
        return False

    file_size = Path(image_path).stat().st_size
    print("[1/3] 读取图片...")
    print(f"      大小: {file_size / 1024:.2f} KB")

    # 2. 转换为 base64
    try:
        image_base64 = image_to_base64_url(image_path)
        print(f"      Base64 长度: {len(image_base64)} 字符")
    except Exception as e:
        print("✗ 错误：图片编码失败")
        print(f"  {e}")
        return False

    # 3. 发送 API 请求
    print()
    print("[2/3] 发送推理请求...")

    payload = {"imgs": [image_base64], "prompt": prompt}

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=60,  # 60秒超时
        )

        if response.status_code == 200:
            result = response.json()
            text_result = result.get("text", "")

            print("      ✓ 请求成功")
            print()
            print("=" * 60)
            print("[3/3] 识别结果:")
            print("=" * 60)
            print(text_result)
            print()
            print("=" * 60)
            print("✓ 测试完成")
            print("=" * 60)
            return True

        else:
            print(f"      ✗ 请求失败: HTTP {response.status_code}")
            print(f"      响应: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("      ✗ 请求超时（60秒）")
        print("      提示：首次推理需要较长时间（10-20秒）")
        return False

    except requests.exceptions.ConnectionError:
        print("      ✗ 连接失败")
        print("      提示：请确认容器正在运行")
        print()
        print("检查命令：")
        print("  docker ps | grep hunyuan")
        print("  docker logs hunyuan-qwen")
        return False

    except Exception as e:
        print(f"      ✗ 发生错误: {e}")
        return False


def main():
    """主函数"""
    # 默认参数
    image_path = "outputs/sample.png"
    prompt = "请详细描述这张图片的内容，包括场景、物体、颜色等"

    # 支持命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        prompt = sys.argv[2]

    # 运行测试
    success = test_image_recognition(image_path, prompt)

    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
