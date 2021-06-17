import cv2
import os, os.path

def getImage () :
    allMasks = os.listdir(".\scripts\classes_chords")
    for folder in allMasks:
        _, _, files = next(os.walk(".\scripts\classes_chords\\"+str(folder)))
        for val in range(len(files)):
            if (folder == "DO"):
                img = cv2.imread(".\scripts\classes_chords\\"+str(folder)+"\\"+str(val)+".jpg.jpg", cv2.IMREAD_COLOR)    
            else:
                img = cv2.imread(".\scripts\classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", cv2.IMREAD_COLOR)

            mask = cv2.imread(".\scripts\masks_chords\\"+str(folder)+"\\"+str(val)+".jpg", 0)

            if (folder == "DO") :
                if (val in [6]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, 0)
                elif (val in [18]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -150, 0)
                elif (val in [26]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -450, 0)
                elif (val in [28]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                elif (val in [43]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "FA") :
                if (val in [0, 4, 6]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 600, 0)
                elif (val in [1, 3, 5]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, 0)
                elif (val in [7]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 180, 0)
                elif (val in [9]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -180, 0)
                elif (val in [16]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -180, 0)
                elif (val in [25]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                elif (val in [31]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 210, 0)
                elif (val in [36]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 150, 0)
                elif (val in [40, 41, 48]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 210, 0)
                elif (val in [54, 59]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "FA#") :
                if (val in [0, 1, 2, 3, 4, 5]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, -600)
                elif (val in [6]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, -360)
                elif (val in [16]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 150, -120)
                elif (val in [24, 26, 28, 30]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 150, 0)
                elif (val in [25, 27, 28, 29]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, 0)
                elif (val in [30]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 360, 0)
                elif (val in [35]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 150, 0)
                elif (val in [43]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                elif (val in [60]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "LA") :
                if (val in [0, 1, 2, 3]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 750, 0)
                elif (val in [4, 5, 6, 7, 11, 12]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, 0)
                elif (val in [17]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -99, 0)
                elif (val in [22, 23, 24]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                elif (val in [30, 31, 32]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -450, 0)
                elif (val in [54, 56, 61]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 240, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "MI") :
                if (val in [0, 2, 3, 4, 5]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 300, 0)
                elif (val in [15, 16, 23, 24, 29]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -180, 0)
                elif (val in [31, 35, 38]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 600, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "RE") :
                if (val in [0, 1, 2, 5, 6, 12]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 630, 0)
                elif (val in [3, 4]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 210, 0)
                elif (val in [28, 29]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -210, 0)
                elif (val in [38]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 120, 0)
                elif (val in [42]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -150, 0)
                elif (val in [65, 66, 67, 68, 69]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -450, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "SI") :
                if (val in [2, 3]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -150, 0)
                elif (val in [9, 10]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -99, 0)
                elif (val in [13, 17, 23, 24, 25]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -330, 0)
                elif (val in [33, 34, 35]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 180, 0)
                elif (val in [36]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 150, -180)
                elif (val in [37]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, -450)
                elif (val in [43, 44, 46, 47]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 600, 0)
                elif (val in [48, 49, 52, 54]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 150, 0)
                elif (val in [55, 57, 63, 64]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 600, 0)
                elif (val in [58, 59, 60]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 330, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            elif (folder == "SOL") :
                if (val in [0, 1, 41]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 450, 0)
                elif (val in [1, 6, 12]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 210, 0)
                elif (val in [24]) :
                    reducedImg, reducedMask = adjustImg(img, mask, -99, 0)
                elif (val in [31, 32, 38, 39, 40, 43, 44, 49, 50, 61]) :
                    reducedImg, reducedMask = adjustImg(img, mask, 210, 0)
                elif (val in []) :
                    reducedImg, reducedMask = adjustImg(img, mask, -99, 0)
                else :
                    reducedImg, reducedMask = adjustImg(img, mask, 0, 0)

            cv2.imwrite(".\scripts\\reduced_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", reducedImg)
            cv2.imwrite(".\scripts\\reduced_masks_chords\\"+str(folder)+"\\"+str(val)+".jpg", reducedMask)

            print(str(val)+"/"+str(len(files) - 1)+" - "+str(folder)+" - mask")


def adjustImg (img, mask, HeightAdjust, WidthAdjust) :
    minorHeight = int(img.shape[0] / 2) - int(img.shape[0] / 4)
    majorHeight = int(img.shape[0] / 2) + int(img.shape[0] / 4)
    minorWidth = int(img.shape[1] / 2)
    majorWidth = int(img.shape[1] - 1)

    reducedImg = img[minorHeight + HeightAdjust : majorHeight + HeightAdjust, minorWidth + WidthAdjust : majorWidth + WidthAdjust]

    minorHeight = int(mask.shape[0] / 2) - int(mask.shape[0] / 4)
    majorHeight = int(mask.shape[0] / 2) + int(mask.shape[0] / 4)
    minorWidth = int(mask.shape[1] / 2)
    majorWidth = int(mask.shape[1] - 1)

    reducedMask = mask[minorHeight + int(HeightAdjust / 3) : majorHeight + int(HeightAdjust / 3), minorWidth + int(WidthAdjust / 3) : majorWidth + int(WidthAdjust / 3)]

    return reducedImg, reducedMask

# getImage()

def seeImg () :
    allMasks = os.listdir(".\scripts\classes_chords")
    for folder in allMasks:
        _, _, files = next(os.walk(".\scripts\classes_chords\\"+str(folder)))
        for val in range(len(files)):
            img = cv2.imread(".\scripts\\reduced_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", cv2.IMREAD_COLOR)
            img = cv2.resize(img, (512, 512))
            cv2.imshow("img", img)
            cv2.waitKey(0)
            mask = cv2.imread(".\scripts\\reduced_masks_chords\\"+str(folder)+"\\"+str(val)+".jpg", 0)
            mask = cv2.resize(mask, (512, 512))
            cv2.imshow("img", mask)
            cv2.waitKey(0)
            mask = cv2.resize(mask, (256, 256))
            cv2.imshow("img", mask)
            cv2.waitKey(0)
            mask = cv2.resize(mask, (128, 128))
            cv2.imshow("img", mask)
            cv2.waitKey(0)

            print(str(val)+"/"+str(len(files) - 1)+" - "+str(folder)+" - mask")

# seeImg()

def gray () :
    img = cv2.imread(".\scripts\\reduced_classes_chords\DO\\0.jpg", 0)
    cv2.imwrite(".\scripts\do_gray.jpg", img)

gray()