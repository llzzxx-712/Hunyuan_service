import base64
import io

from PIL import Image


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
