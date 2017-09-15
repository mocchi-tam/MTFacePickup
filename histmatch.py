# -*- coding: UTF-8 -*-
import cv2
import os
import glob
import itertools

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

BASE_DIR = './img/'

def main():
    fm = FaceMatch()
    fm.run()

class FaceMatch():
    def __init__(self):
        print('init facematch')
    
    def run(self):
        print('run FaceMatch...')
        for name in NAMES:
            dirname = BASE_DIR + name
            files = glob.glob(dirname + '/*.jpg')
            for file1, file2 in list(itertools.combinations(files,2)):
                ret = self.CompFace(file1, file2)
                if ret > 0.99:
                    print(file1, file2)
                    if os.path.exists(file2):
                        os.remove(file2)
                    
    def CompFace(self, base, target):
        base_img = cv2.imread(base)
        base_hist = cv2.calcHist([base_img], [0], None, [256], [0, 256])
        
        target_img = cv2.imread(target)
        target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])
        
        return cv2.compareHist(base_hist, target_hist, 0)
        
if __name__ == '__main__':
    main()