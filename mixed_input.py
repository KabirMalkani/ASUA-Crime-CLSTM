from keras.layers import Input, Dense, concatenate, LSTM, ConvLSTM2D, BatchNormalization, Conv2D
from keras.models import Model
import time
import numpy as np
tm = time.clock()


og_mat = np.load('first_layer.npy')

tdim, xdim, ydim = og_mat.shape

n_sample = 320
batch_size = 36
mat = np.zeros((n_sample, batch_size, xdim, ydim, 1))

for i in range(n_sample):
	a = np.random.randint(tdim - batch_size)
	mat[i, 0:batch_size, :, :, 0] = og_mat[a:a+batch_size, :, :]

x = mat[:, :-1, :, :, :]
y = np.reshape(mat[:, -1, :, :, :], (n_sample, xdim, ydim, 1))

print(x.shape, y.shape)

print("Preprocess: ", time.clock() - tm)
tm = time.clock()


input_crime = Input(shape = (batch_size-1, xdim, ydim, 1))

crime = ConvLSTM2D(
	filters=1, 
	kernel_size=(3, 3), 
	padding='same', 
	return_sequences=True)(input_crime)

crime = BatchNormalization()(crime)

crime = ConvLSTM2D(
	filters=1, 
	kernel_size=(3, 3), 
	padding='same', 
	return_sequences=True)(crime)

crime = BatchNormalization()(crime)

crime = ConvLSTM2D(
	filters=1, 
	kernel_size=(3, 3), 
	padding='same', 
	return_sequences=False)(crime)

# crime = BatchNormalization()(crime)

# crime = Conv2D(filters=1, 
# 	kernel_size=(3, 3),
# 	activation='sigmoid',
# 	padding='same',
# 	data_format='channels_last')(crime)

model = Model(inputs=input_crime, outputs=crime)
print(model.summary())
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(x, y, epochs=20, batch_size=32)

print("Train: ", time.clock() - tm)
tm = time.clock()


ypred = model.predict(x[0:1])
print(ypred.shape)