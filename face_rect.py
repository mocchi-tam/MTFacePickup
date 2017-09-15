import glob
import os
import shutil
import cv2

import numpy as np
import chainer
import chainer.functions as F
import net

NAMES={
'asai':0,
'harima':1,
'homma':2,
'inagaki':3,
'kurosu':4,
'maeda':5,
'michieda':6,
'muto':7,
'nagatomo':8,
'noguchi':9,
'sato':10,
'shoji':11,
'suzuki':12,
'taguchi':13,
'taya':14,
'umemoto':15,
'yamane':16,
'yamauchi':17,
'yasuda':18,
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
        #img = cv2.imread(face)
        img = np.asarray(face/255.0, dtype=np.float32).transpose(2,0,1).reshape((1,3,224,224))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            f = model(img)
        result = int(F.argmax(f).data)
        name = self.namedict[result]
        return name
		
    def run_cor(self):
        files = glob.glob('./img/tmp/*')
        for file in files:
            name_key = self.inference(file)
            shutil.move(file, './img/' + name_key)
	
    def run(self):
        #for name in NAMES:
            #if os.path.exists(self.folderName)==False:
            #    os.mkdir(self.folderName)
            
		self.count = 0
		
		files = glob.glob('data/*.jpg')
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
					fname = './tmp/facetmp' + str(self.count) + '.jpg' 
					cv2.imwrite(fname, dst_img)
					self.count += 1
    
if __name__ == '__main__':
    main()