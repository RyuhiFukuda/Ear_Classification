import cv2
import numpy as np
import os


def main():
    input_dir = ""
    output_dir = ""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    photos = os.listdir(input_dir)
    for pic_name in photos:

        img = cv2.imread(input_dir + "/" + pic_name, cv2.IMREAD_COLOR)

        img_gray = cv2.imread(input_dir + "/" + pic_name, cv2.IMREAD_GRAYSCALE)
        M = cv2.moments(img_gray, False)
        contours, hierarchy = cv2.findContours(
            img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        cv2.drawContours(img, contours, -1, color=(0, 0, 0), thickness=5)
        

        if x-400 < 0:
            dst = img[y-400:y+400, 0:x+400-(x-400)]
            cv2.imwrite(output_dir + '/' + pic_name , dst)
            print(pic_name)
        
        elif y-400 < 0:
            dst = img[y-400-(y-400):y+400, x-400:x+400]
            cv2.imwrite(output_dir + '/' + pic_name , dst)
            print(pic_name)

        else:
            dst = img[y-400:y+400, x-400:x+400]
            cv2.imwrite(output_dir + '/' + pic_name , dst)
            print(pic_name)
    print("Successed!")


if __name__ == "__main__":
    main()