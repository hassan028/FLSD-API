import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw

def draw_box(img, boxes):
    box = ImageDraw.Draw(img)
    for i in range(boxes.shape[0]):
        data = list(boxes[i])
        shape = [data[0], data[1], data[2], data[3]]
        box.rectangle(shape, outline ="#02d5fa", width=3)
    return img

def url_to_img(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def img_to_bytes(img):
    buffered = BytesIO()
    img.save(buffered, format=img.format)
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")

def bytes_to_img(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file)
    img_format = img.format.lower()
    return img, img_format