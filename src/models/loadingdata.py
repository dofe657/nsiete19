from numpy import load
from numpy import ones
from numpy import zeros
from numpy.random import randint
from matplotlib import pyplot

def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1,X2]

def generate_real_samples(dataset,n_samples, patch_shape):
    ix = randint(0, dataset.shape[0],n_samples)
    X = dataset[ix]
    Y = ones((n_samples,patch_shape,patch_shape,1))
    return X, Y

def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    Y = zeros((len(X),patch_shape,patch_shape,1))
    return X, Y

def save_models(step, g_model_AtoB, g_model_BtoA):
    path = '../../models/'
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(path+filename1)
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(path+filename2)

def summarize_performance(step, g_model, trainX, name, n_samples=5):
    path = '../../models/'
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
    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
    pyplot.savefig(path+filename1)
    pyplot.close()