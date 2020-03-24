from keras.layers import Input, Dense, add, concatenate, multiply, LSTM, ConvLSTM2D, BatchNormalization, Conv2D, AveragePooling2D, Reshape, Lambda
from keras.models import Model
from keras import backend as K
import time
import numpy as np
from keras.utils import plot_model

tm = time.clock()

# Load environment data
og_env = np.load('environment.npy')
og_env = np.reshape(og_env, (1, 128, 128, 19))

# Load weather data
og_wet = np.load('weather.npy')

# # Load and fix crime data
og_crime = np.load('first_layer.npy')

tdim, xdim, ydim = og_crime.shape

n_sample = 320
batch_size = 36
mat = np.zeros((n_sample, batch_size, xdim, ydim, 1))
wet = np.zeros((n_sample, batch_size, og_wet.shape[1]))

for i in range(n_sample):
	a = np.random.randint(tdim - batch_size)
	mat[i, 0:batch_size, :, :, 0] = og_crime[a:a+batch_size, :, :]
	wet[i, 0:batch_size, :] = og_wet[a:a+batch_size, :]

x = mat[:, :-1, :, :, :]
wet = wet[:, :-1, :]
y = np.reshape(mat[:, -1, :, :, :], (n_sample, xdim, ydim, 1))



# # Mixed Input Model

# Environment
input_env = K.variable(og_env)
input_env = Input(tensor=input_env)

env = AveragePooling2D(pool_size=(2, 2), data_format='channels_last')(input_env)

env = Conv2D(filters=40, 
	kernel_size=(2, 2),
	strides=(2, 2),
	activation='sigmoid',
	padding='valid',
	data_format='channels_last')(env)

env = BatchNormalization()(env)


# Weather
input_wet = Input(shape = (batch_size-1, 12))

weather = LSTM(
	units=20,
	return_sequences=False)(input_wet)

weather = BatchNormalization()(weather)


# Crime Layers
input_crime = Input(shape = (batch_size-1, xdim, ydim, 1))

crime = ConvLSTM2D(
	filters=20, 
	kernel_size=(5, 5), 
	padding='same', 
	return_sequences=True)(input_crime)

crime = BatchNormalization()(crime)

crime = multiply([crime, weather])

crime = ConvLSTM2D(
	filters=40, 
	kernel_size=(4, 4), 
	padding='same', 
	return_sequences=True)(crime)

crime = BatchNormalization()(crime)

crime = add([env, crime])

crime = ConvLSTM2D(
	filters=50, 
	kernel_size=(2, 2), 
	padding='same', 
	return_sequences=False)(crime)

crime = BatchNormalization()(crime)

crime = Conv2D(
	filters=1, 
	kernel_size=(1, 1), 
	padding='same')(crime)


# print(env.shape)
# print(crime.shape)


model = Model(inputs=[input_wet, input_env, input_crime], outputs=crime)
plot_model(model, to_file='model.png', show_shapes=True)


# model.compile(optimizer='adadelta', loss='binary_crossentropy')
# model.fit(x, y, epochs=20, batch_size=32)

# print("Train: ", time.clock() - tm)
# tm = time.clock()


# ypred = model.predict(x[0:1])
# print(ypred.shape)