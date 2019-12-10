import sys
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from numpy import vstack, expand_dims
from loadingdata import load_dataset
from matplotlib import pyplot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def image_to_translate(path):
    img = load_img(path)
    img = img_to_array(img)
    img = expand_dims(img, 0)
    img = (img - 127.5) / 127.5
    return img

def translate(domain, img, AtoB, BtoA):
    if domain == 'm':
        generated = AtoB.predict(img)
        reconstructed = BtoA.predict(generated)
    else:
        generated = BtoA.predict(img)
        reconstructed = AtoB.predict(generated)
    return generated, reconstructed

def save_img(img, generated, reconstructed,path):
    images = vstack((img,generated,reconstructed))
    titles = ['Input','Generated','Reconstructed']
    images = (images + 1) / 2.0
    for i in range(len(images)):    
        pyplot.subplot(1,len(images),i+1)
        pyplot.axis('off')
        pyplot.imshow(images[i])
        pyplot.title(titles[i])
    pyplot.savefig(path+'translation.png')

# Generate translation with: python3 generate.py [name of AtoB model] [name of BtoA model] [picture to translate] [m/f]
# Models should be placed in models folder in the repo and image to translate should be placed in translate folder in repo.

path_models = '../../models/'
path_image = '../../translate/'

cust = {'InstanceNormalization': InstanceNormalization}
AtoB = load_model(path_models+sys.argv[1],cust)
BtoA = load_model(path_models+sys.argv[2],cust)

img = image_to_translate(path_image+sys.argv[3])

generated, reconstructed = translate(sys.argv[4],img,AtoB,BtoA)
save_img(img,generated,reconstructed,path_image)