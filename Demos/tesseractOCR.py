'''
Tesseract的 OCR 业内三款识别引擎之一，开源
'''

from PIL import Image
import pytesseract

path = 'F:/Github/DL-data-processing-methods/Demos/img/fsb.png'

text = pytesseract.image_to_string(Image.open(path), lang='chi_sim')
print(text)
# f = open("text.txt", 'w+')
# f.write(text)
# f.close()
