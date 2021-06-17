import cv2
import os

for current in range (0, 63):
    mask = cv2.imread("./database IC/Acorder_Maiores/SOL/Segmentado/"+str(current)+"_mao.jpg", 0)
    for i in range (mask.shape[0]):
        for j in range (mask.shape[1]):
            if (current != 0):
                if (mask[i][j] < 127):
                    mask[i][j] = 0
                else:
                    mask[i][j] = 255
            else:
                if (mask[i][j] > 100):
                    mask[i][j] = 255
                else:
                    mask[i][j] = 00
    cv2.imwrite("./scripts/masks_chords/SOL/"+str(current)+".jpg", mask)
    print(current)