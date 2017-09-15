import glob
import os
import cv2

import numpy as np
import chainer
import chainer.functions as F
import net

NAMES={
'asai':0,
'inagaki':1,
'umemoto':2,
'kurosu':3,
'sato':4,
'shoji':5,
'suzuki':6,
'taguchi':7,
'taya':8,
'nagatomo':9,
'noguchi':10,
'harima':11,
'homma':12,
'maeda':13,
'michieda':14,
'muto':15,
'yasuda':16,
'yamauchi':17,
'yamane':18,
'none':19
}

imx = 224
imy = 224
n_cat = len(NAMES)

model = net.MTNNet(n_out=n_cat)
chainer.serializers.load_npz('./net/net.model', model)

def main():
    mtc = MTClass()
    mtc.run()

class MTClass():
    def __init__(self):
        self.cascade = cv2.CascadeClassifier('meta/haarcascade_frontalface_default.xml')
        print('init MTClass')
        self.namedict = {v:k for k, v in NAMES.items()}
    
    def inference(self, face):
        img = np.asarray(face/255.0, dtype=np.float32).transpose(2,0,1).reshape((1,3,224,224))
        f = model(img)
        result = int(F.argmax(f).data)
        name = self.namedict[result]
        return name
    
    def run(self):
        for name in NAMES:
            self.folderName = os.getcwd() + '/tmp/'
            if os.path.exists(self.folderName)==False:
                os.mkdir(self.folderName)
            
            self.count = 0
            
            files = glob.glob('data/' + name + '/*.jpg')
            for file in files:
                print(file)
                img = cv2.imread(file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
                faces = self.cascade.detectMultiScale(gray
                                                      ,scaleFactor=1.8
                                                      ,minNeighbors=1
                                                      ,minSize=(24,24)
                                                      )
                
                if len(faces) > 0:
                    for rect in faces:
                        dst_img = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                        dst_img = cv2.resize(dst_img, (224,224))
                        name_key = self.inference(dst_img)
                        fname = self.folderName + name_key + '/face' + str(self.count) + '.jpg' 
                        cv2.imwrite(fname, dst_img)
                        self.count += 1
    
if __name__ == '__main__':
    main()