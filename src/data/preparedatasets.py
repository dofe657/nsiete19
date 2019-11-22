from os import listdir
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

def load_dataset(path):
    data = list()
    entries = listdir(path)
    for e in entries:
        img = load_img(path+e)
        img = img_to_array(img)
        data.append(img)
    return np.asarray(data)

#path = '../../data/processed/'
path = '../../data/sranda/'
dataMale = load_dataset(path+'male/')
dataMaleTest = load_dataset(path+'testMale/')
dataMaleComplete = np.vstack((dataMale,dataMaleTest))
dataFemale = load_dataset(path+'female/')
dataFemaleTest = load_dataset(path+'testFemale/')
dataFemaleComplete = np.vstack((dataFemale,dataFemaleTest))
#np.savez_compressed('genderswap.npz',dataMaleComplete,dataFemaleComplete)
np.savez_compressed('genderswaptest.npz',dataMaleComplete,dataFemaleComplete)