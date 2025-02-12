import os
from PIL import Image

FromImgName = ""
ToImgName = ""

if not os.path.exists(ToImgName):
    os.makedirs(ToImgName)
files = os.listdir(FromImgName)

for file in files:
    img = Image.open(os.path.join(FromImgName, file))
    img_resize = img.resize((96, 96))
    img_resize.save(os.path.join(ToImgName, file))
                

print("Successed!")