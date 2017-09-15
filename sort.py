# coding: UTF-8
import sys
import os
import re
import glob

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

for name in NAMES:
	files = glob.glob('./img/' + name + '/*')
	a = 0
	
	for file in files:
		#jpg = re.compile("jpg")
		#if jpg.search(file):
		os.rename(file, './img/' + name + '/' + name + "%06d.jpg"%(a))
		a += 1
		#else:
		#pass