import cv2
import random
import os

WIDTH = 96
HEIGHT = 96
PX = WIDTH * HEIGHT

def main():
    input_dir = ""
    output_dir = ""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    photos = os.listdir(input_dir)
    for photo in photos:
        img = cv2.imread(input_dir + "/" + photo)

        for i in range(WIDTH):
            for j in range(HEIGHT):
                b, g, r = img[i, j]

                if b < 20:
                    if g < 20:
                        if r < 20:
                            rand = random.randint(0, 255)
                            img[i, j] = (rand, rand, rand)
        
        savename = output_dir + "/" + photo
        cv2.imwrite(savename, img)


if __name__ == "__main__":
    main()