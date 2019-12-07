from numpy import load
from numpy import ones
from numpy import zeros
from numpy import asarray
from numpy.random import randint
from matplotlib import pyplot
from os import listdir
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


def load_dataset(path):
    entriesA = listdir(path+'male/')
    entriesB = listdir(path+'female/')
    domainA = list()
    domainB = list()
    print('Loading male images')
    lenA = len(entriesA)
    lenB = len(entriesB)
    for e in entriesA:
        img = load_img(path+'male/'+e)
        img = img_to_array(img)
        domainA.append(img)
    print('Loading female images')
    for e in entriesB:
        img = load_img(path+'female/'+e)
        img = img_to_array(img)
        domainB.append(img)
    return asarray(domainA), asarray(domainB)

def load_dataset_test(path):
    entriesA = listdir(path+'testMale/')
    entriesB = listdir(path+'testFemale/')
    domainA = list()
    domainB = list()
    print('Loading male test images')
    lenA = len(entriesA)
    lenB = len(entriesB)
    for e in entriesA:
        img = load_img(path+'testMale/'+e)
        img = img_to_array(img)
        domainA.append(img)
    print('Loading female test images')
    for e in entriesB:
        img = load_img(path+'testFemale/'+e)
        img = img_to_array(img)
        domainB.append(img)
    return asarray(domainA), asarray(domainB)

def generate_real_samples(dataset,n_samples, patch_shape):
    ix = randint(0, dataset.shape[0],n_samples)
    X = dataset[ix]
    Y = ones((n_samples,patch_shape,patch_shape,1))
    return X, Y

def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    Y = zeros((len(X),patch_shape,patch_shape,1))
    return X, Y

def save_models(epoch, g_model_AtoB, g_model_BtoA):
    path = '../../models/'
    AtoB = 'g_model_AtoB_%03d.h5' % (epoch+1)
    BtoA = 'g_model_BtoA_%03d.h5' % (epoch+1)
    g_model_AtoB.save(path+AtoB)
    g_model_BtoA.save(path+BtoA)

def summarize_performance(epoch, g_model, trainX, name, n_samples=5):
    path = '../../performace/'
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i])
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i])
    filename1 = '%s_generated_plot_%03d.png' % (name, (epoch+1))
    pyplot.savefig(path+filename1)
    pyplot.close()