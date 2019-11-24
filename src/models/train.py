from random import random
from numpy.random import randint
from numpy import asarray
from model import *
from loadingdata import *

def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			selected.append(image)
		else:
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	n_epochs, n_batch, = 5, 1
	n_patch = d_model_A.output_shape[1]
	trainA, trainB = dataset
	poolA, poolB = list(), list()
	bat_per_epo = int(len(trainA) / n_batch)
	n_steps = bat_per_epo * n_epochs

	for i in range(n_steps):

		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)

		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))

		if (i+1) % (bat_per_epo * 1) == 0:

			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')

			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo / 10) == 0:
            
			save_models(i, g_model_AtoB, g_model_BtoA)


dataset = load_real_samples('../data/genderswap.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

image_shape = dataset[0].shape[1:]

g_model_AtoB = generator(image_shape)

g_model_BtoA = generator(image_shape)

d_model_A = discriminator(image_shape)

d_model_B = discriminator(image_shape)

c_model_AtoB = composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)

c_model_BtoA = composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
