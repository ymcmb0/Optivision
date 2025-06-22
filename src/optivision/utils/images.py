import io
from typing import Union
from PIL import Image

def compress_image(img: Image.Image,
                   destination: Union[str, io.BytesIO],
                   quality: int = 40) -> None:
    """JPEG-compress PIL image.  Destination may be filepath or BytesIO."""
    if isinstance(destination, str):
        img = img.resize((img.width // 2, img.height // 2))
        img.save(destination, format="JPEG", optimize=True, quality=quality)
    else:  # BytesIO
        tmp = img.resize((img.width // 2, img.height // 2))
        tmp.save(destination, format="JPEG", optimize=True, quality=quality)
        destination.seek(0)
