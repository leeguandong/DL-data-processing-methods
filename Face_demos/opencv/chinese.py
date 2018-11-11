'''
Opencv 添加中文乱码，添加英文没有问题，需要处理一下
'''

'''
实现思路：使用 PIL 的图片绘制中文，可以指定字体文件，那么也就是可以使用 PIL 可以实现中文的输出。
1. Opencv 图片格式转化成 PIL 的图片的格式
2. 使用 PIL 绘制文字
3. PIL 图片格式转换成 Opencv 的图片格式
'''

import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

# def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
#     if (isinstance(img, numpy.ndarray)):  # 判断是否 Opencv 图片类型
#         # Opencv 图片转换成为 PIL 图片格式
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#
#     # 使用 PIL 绘制文字
#     draw = ImageDraw.Draw(img)
#     fontText = ImageFont.truetype('font/simsun.ttc', textSize, encoding='utf-8')
#     draw.text((left, top), text, textColor, font=fontText)
#
#     # PIL 图片格式转换成 Opencv 的图片格式
#     return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
