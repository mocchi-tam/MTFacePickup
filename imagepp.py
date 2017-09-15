# -*- coding: utf-8 -*-
import chainer
import numpy as np
import glob
import cv2
import augmentation
from chainer.datasets import tuple_dataset

class ImagePP():
    def __init__(self, gpu):
        self.xp = np if gpu < 0 else chainer.cuda.cupy
        
    def makedataset(self, filenames, cat, train=True):
        xp = self.xp
        N_dataset = 0
        
        for f in filenames:
            files = glob.glob(f + '/*.jpg')
            for file in files:
                N_dataset += 1
                
        dx = 224
        dy = 224
        n_da = 18 if train else 1
        
        df = xp.empty((N_dataset*n_da,3,dy,dx), dtype=xp.float32)
        label = xp.zeros((N_dataset*n_da), dtype=xp.int32)
        fnames = []
        
        count = 0
        ag = augmentation.ImgAug()
        
        def add(ad_n, ad_label, ad_img):
            ad_img = xp.asarray(ad_img/255.0, dtype=xp.float32).transpose(2,0,1)
            
            df[ad_n] = ad_img
            label[ad_n] = ad_label
        
        for (i, f) in enumerate(filenames):
            files = glob.glob(f + '/*.jpg')
            for file in files:
                print(file)
                fnames.append(file)
                if train:
                    imgs = ag.aug(file)
                    for img in imgs:
                        add(count, i, img)
                        count += 1
                else:
                    img = cv2.imread(file)
                    add(count, i, img)
                    count += 1
        
        #df = self.preprocess(df, dx, dy, train=train)
        df = tuple_dataset.TupleDataset(df, label)
        
        return df, fnames
    
    def preprocess(self, df, ims_x, ims_y, train=True):
        ar = df.reshape((-1, 3, ims_x*ims_y))
        if train:
            #self.mean = ar.mean(axis=(0, 2))
            #self.std = ar.std(axis=(0, 2))
            self.mean = 0
            self.std = 255
            
        x = self.normalize(ar, df, self.mean, self.std)
        return x
    
    def normalize(self, x, df, mean, std):
        shape = df.shape
        x = x.reshape((-1, 3)) - mean
        if std is not None:
            x /= std
        return x.reshape(shape)