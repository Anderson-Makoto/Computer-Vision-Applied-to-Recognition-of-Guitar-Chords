import cv2
import numpy as np    
import os, os.path
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SequentialFeatureSelector

def getAllMaks () :
    height = 32
    width = 32
    features = []
    x = []
    y = []
    print(features)
    allMasks = os.listdir(".\scripts\deep_learning_segmentation")
    for folder in allMasks:
        count = 0
        _, _, files = next(os.walk(".\scripts\deep_learning_segmentation\\"+str(folder)))
        for val in range(len(files)):
            img = cv2.imread(".\scripts\deep_learning_segmentation\\"+str(folder)+"\\"+str(val)+".jpg", 0)
            img = cv2.resize(img, (height, width))
            feat, xVal, yVal = createFeatureArray(img, folder, height, width)
            features.append(feat)
            x.append(xVal)
            y.append(yVal)
            print(str(count)+"/"+str(len(files) - 1)+" - "+str(folder)+" - mask")
            count += 1
    features = np.asarray(features)
    x = np.asarray(x)
    y = np.asarray(y)
    print(features.shape)
    print(x.shape)
    print(y.shape)
    trainAndPredict(x, y)

def createFeatureArray (img, folder, h, w) :
    img = np.array(img)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            if (img[i][j] > 200):
                img[i][j] = 1
            else:
                img[i][j] = 0
    img = img.reshape(h ** 2)
    feat = np.append(img, [folder])
    print(img)

    return feat, img, folder

def posProcess (img) :
    kernel = np.ones((15,15),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img



# def getChainCode (img) :
#     top = 0
#     left = 0
#     right = 0
#     down = 0

# def ratio (img) :
#     img= cv2.threshold(img, 150, 255,  cv2.THRESH_BINARY)[1]
#     selectedCnt = []
#     selectedRazao = 100
#     cnts, _= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in cnts:
#         esquerda= tuple(cnt[cnt[:, :, 0].argmin()][0])
#         direita= tuple(cnt[cnt[:, :, 0].argmax()][0])
#         cima= tuple(cnt[cnt[:, :, 1].argmin()][0])
#         baixo= tuple(cnt[cnt[:, :, 1].argmax()][0])

#         largura= direita[0]- esquerda[0]
#         altura= baixo[1]- cima[1]
        
#         razao= largura/altura
#         print(abs(razao - 1.4))
#         if (abs(razao - 1.4) < selectedRazao and len(cnt) > 80 and direita[0] != 255 and esquerda[0] != 0 and cima[1] != 0 and baixo[1] != 255) :
#             selectedRazao = abs(razao - 1.4)
#             selectedCnt = cnt

#             imgAux = img
#             imgAux = cv2.cvtColor(imgAux, cv2.COLOR_GRAY2RGB)
#             cv2.drawContours(imgAux, cnt, -1, (0,255,0), 3)
#             cv2.imshow("cnt", imgAux)
#             cv2.waitKey(0)
        

def trainAndPredict (x, y) :
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.33, random_state = 0, stratify = y)

    rf = RandomForestClassifier()
    et = ExtraTreesClassifier()
    mp = MLPClassifier()
    svc = SVC()
    # baggingClf = BaggingClassifier(svc, n_estimators = 8, random_state=42, verbose = 1)
    # sfs = SequentialFeatureSelector(baggingClf, n_features_to_select=100)
    # vote = VotingClassifier([("Random Forest", rf), ("Extra Trees", et), ("Multilayer Perceptron", mp), ("SVM", svc)], voting = "hard")
    stacking = StackingClassifier([("Extra Trees", et), ("Multilayer Perceptron", mp), ("SVM", svc)], final_estimator = rf)
    stacking.fit(xTrain, yTrain)
    yPred = stacking.predict(xTest)
    cr = classification_report(yTest, yPred, target_names = ["DO", "FA", "FA#", "LA", "MI", "RE", "SI", "SOL"])
    print(cr)





getAllMaks()