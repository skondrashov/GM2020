from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

import csv

def plot_weights(model):
    for layer in model.layers:
        W = layer.get_weights()[0]

        for i in range(W.shape[2]):
            for j in range(W.shape[3]):
                print(W[:,:,i,j])
                plt.subplot(W.shape[2], W.shape[3], 1+j+i*W.shape[3])
                plt.imshow((W[:,:,i,j]+1)/2, cmap="gray")
        plt.show()

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def show_board(board):
    for row in board:
        for piece in row:
            if piece[2]:
                letter = 'K'
            elif piece[3]:
                letter = 'Q'
            elif piece[4]:
                letter = 'R'
            elif piece[5]:
                letter = 'N'
            elif piece[6]:
                letter = 'B'
            elif piece[7]:
                letter = 'P'
            else:
                letter = '.'

            if piece[1]:
                letter = letter.lower()

            print(letter, end='')
        print()

def show_moves(moves):
    for row in moves:
        for move in row:
            move = 'X' if move[0] else '.'
            print(move, end='')
        print()

print("Done importing.")














































from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

import csv

def plot_weights(model):
	for layer in model.layers:
		W = layer.get_weights()[0]

		for i in range(W.shape[2]):
			for j in range(W.shape[3]):
				print(W[:,:,i,j])
				plt.subplot(W.shape[2], W.shape[3], 1+j+i*W.shape[3])
				plt.imshow((W[:,:,i,j]+1)/2, cmap="gray")
		plt.show()

def chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]

def show_board(board):
	for row in board:
		for piece in row:
			if piece[2]:
				letter = 'K'
			elif piece[3]:
				letter = 'Q'
			elif piece[4]:
				letter = 'R'
			elif piece[5]:
				letter = 'N'
			elif piece[6]:
				letter = 'B'
			elif piece[7]:
				letter = 'P'
			else:
				letter = '.'

			if piece[1]:
				letter = letter.lower()

			print(letter, end='')
		print()

def show_moves(moves):
	for row in moves:
		for move in row:
			move = 'X' if move[0] else '.'
			print(move, end='')
		print()

FILTERS = {
'neutral':
	[[[0],[0],[0]],
	 [[0],[0],[0]],
	 [[0],[0],[0]]],
'diagonal':
	[[[1],[0],[1]],
	 [[0],[0],[0]],
	 [[1],[0],[1]]],
 'adjacent':
	[[[1],[1],[1]],
	 [[1],[0],[1]],
	 [[1],[1],[1]]],
 }

WEIGHTS = [
	np.array(
		#   Wh   Bl    K    Q    R    N    B    P   mv   ep
		[[[[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 1],[ 0],[ 0],[ 0]],
		  [[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0]],
		  [[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 1],[ 0],[ 0],[ 0]]],
		 [[[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0]],
		  [[-1],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0]],
		  [[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0]]],
		 [[[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 1],[ 0],[ 0],[ 0]],
		  [[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 0]],
		  [[ 0],[ 0],[ 0],[ 0],[ 0],[ 0],[ 1],[ 0],[ 0],[ 0]]]]
	),
	np.array([0])
]

TRAINING_SIZE = 19000
TOTAL_SIZE = 31000

piece_to_id = {
	'K':[1,0,0,0,0,0],
	'Q':[0,1,0,0,0,0],
	'R':[0,0,1,0,0,0],
	'N':[0,0,0,1,0,0],
	'B':[0,0,0,0,1,0],
	'P':[0,0,0,0,0,1],
}

LAYERS = [
	'white',
	'black',
	'king',
	'queen',
	'rook',
	'knight',
	'bishop',
	'pawn',
	'has_moved',
	'enpassant',
]

if True:
	X = []
	Y = []
	i = 0
	for line in csv.reader(open('moves.csv'), delimiter='|'):
		i+= 1
		if i > TOTAL_SIZE:
			break

		player = 1 if int(line[0]) == 1 else 0
		if not player:
			continue

		found_bishop=False

		colors = [[1,0]]*16 + [[0,1]]*16
		pieces = [piece_to_id[char] for char in line[2].split(',')]
		has_moved = [([1] if n!='0' else [0]) for n in line[3].split(',')]
		enpassant = [[0]]*32
		if line[4]:
			enpassant[int(line[4])-1] = [1]

		squares = []
		for row in chunks(line[1].split(','), 8):
			rank = []
			for id in row:
				if not id:
					square = [0,0] + [0]*6 + [0] + [0]
				else:
					id = int(id)-1
					if pieces[id] == [0,0,0,0,1,0]:
						found_bishop = True
					square = colors[id] + pieces[id] + has_moved[id] + enpassant[id]
				rank.append(square)
			squares.append(rank)

		if not found_bishop:
			continue

		X.append(squares)

		destinations = []
		for line in chunks(line[5].split(','), 8):
			destinations.append([(1 if n!='0' else 0) for n in line])
		Y.append(destinations)

	X = np.array(X).astype('float32')
	Y = np.array(Y)
	Y = np.expand_dims(Y, 3)

	X_train = X[:TRAINING_SIZE]
	Y_train = Y[:TRAINING_SIZE]
	X_test  = X[TRAINING_SIZE:]
	Y_test  = Y[TRAINING_SIZE:]

	model = Sequential()
	model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=X[0].shape))
	model.add(Conv2D(4, (5, 5), padding='same', activation='relu'))
	model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(4, (5, 5), padding='same', activation='relu'))
	model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))
	# model.add(Conv2D(1, (3, 3), padding='same', activation='relu', input_shape=(8,8,1)))

	model.compile(
		loss=keras.losses.mean_squared_error,
		metrics=['accuracy']
	)

	best_accuracy = 0
	best_filter = ''
	best_property = None
	for key, filter in FILTERS.items():
		model.layers[0].set_weights([
				np.expand_dims(np.array(filter), 3),
				np.array([0])
		])
		# plot_weights(model)

		for i, X_test_slice in enumerate(np.split(X_test, 10, 3)):
			score = model.evaluate(X_test_slice, Y_test, verbose=0)
			if best_accuracy < score[1]:
				best_accuracy = score[1]
				best_property = i
				best_filter = key
			# print(key, LAYERS[i], 'accuracy:', score[1])

	print(best_filter, 'got an accuracy of', best_accuracy, 'on', LAYERS[best_property])

	model.layers[0].set_weights([
		np.expand_dims(np.array(FILTERS[best_filter]), 3),
		np.array([0])
	])

	Y_prob = (model.predict(np.split(X_test, 10, 3)[best_property]) > .5).astype('int')

	print(np.concatenate(X_train, Y_prob, 3).shape)

	for i, y_prob in enumerate(Y_prob):
		if not (y_prob == Y_test[i]).all():
			show_board(X_test[i])
			print()
			show_moves(y_prob)
			print()
			show_moves(Y_test[i])
			print('-----')


	Y_prob = model.predict(X)
	losses = []
	for i, y_prob in enumerate(Y_prob):
		losses.append(((Y[i] - y_prob)**2).mean())
	losses = np.array(losses)
	np.argpartition(losses

	# model.add(Conv2D(1, (3, 3), padding='same', activation='relu', input_shape=X[0].shape))-
	# model.add(Flatten())
	# model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(num_classes, activation='softmax'))

	# print(model(X_train[0]))

	model.compile(
		loss=keras.losses.mean_squared_error,
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy']
	)

	model.fit(
		X_train,
		Y_train,
		batch_size=128,
		epochs=20,
		verbose=1,
		validation_data=(X_test, Y_test),
		shuffle=True,
		callbacks = [
			ModelCheckpoint(
				filepath='best_model.h5',
				monitor='val_loss',
				save_best_only=True,
				verbose=1,
			)]
	)
else:
	model = load_model('best_model.h5')

# W = model.get_layer(name=layer).get_weights()[0]
# if len(W.shape) == 4:
#     W = np.squeeze(W)
#     W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3]))
#     fig, axs = plt.subplots(5,5, figsize=(8,8))
#     fig.subplots_adjust(hspace = .5, wspace=.001)
#     axs = axs.ravel()
#     for i in range(25):
#         axs[i].imshow(W[:,:,i])
#         axs[i].set_title(str(i))

# print(WEIGHTS.shape)
# print(model.layers[0].get_weights())
# print(model.layers[0].get_weights()[0].shape)

plot_weights(model)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

Y_prob = (model.predict(X_test) > .5).astype('int')

for i, y_prob in enumerate(Y_prob):

	if not (y_prob == Y_test[i]).all():
		show_board(X_test[i])
		print()
		show_moves(y_prob)
		print()
		show_moves(Y_test[i])
		print('-----')

# Extracts the outputs of the top 5 layers
layer_outputs = [layer.output for layer in model.layers[:7]]

# # Creates a model that will return these outputs, given the model input
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# # Returns a list of five Numpy arrays: one array per layer activation
# activations = activation_model.predict(img_tensor)

# layer_names = []
# for layer in model.layers[:5]:
# 	layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

# images_per_row = 16

# for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
# 	n_features = layer_activation.shape[-1] # Number of features in the feature map
# 	size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
# 	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
# 	display_grid = np.zeros((size * n_cols, images_per_row * size))
# 	for col in range(n_cols): # Tiles each filter into a big horizontal grid
# 		for row in range(images_per_row):
# 			channel_image = layer_activation[
# 				0,
# 				:, :,
# 				col * images_per_row + row]
# 			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
# 			channel_image /= channel_image.std()
# 			channel_image *= 64
# 			channel_image += 128
# 			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
# 			display_grid[col * size : (col + 1) * size, # Displays the grid
# 						 row * size : (row + 1) * size] = channel_image
# 	scale = 1. / size
# 	plt.figure(figsize=(scale * display_grid.shape[1],
# 						scale * display_grid.shape[0]))
# 	plt.title(layer_name)
# 	plt.grid(False)
# 	plt.imshow(display_grid, aspect='auto', cmap='viridis')


# csv_reader = csv.reader(open('moves.csv'), delimiter='|')
# for line in csv_reader:
# 	player = int(line[0]) == 1 ? 1 : 0
# 	piece_ids = [piece_to_id[char] for char in line[2].split(',')]
# 	times_moved = [(1 if n!='0' else 0) for n in line[3].split(,)]
# 	enpassant = line[4]
# 	for destinations in line[5:]:
# 		if not destinations:
# 			destinations = np.array(8,8)

# input_shape = (8, 8, 1)

# from keras.datasets import mnist
# # download mnist data and split into train and test sets
# (X_train, Y_train), (X_test, y_test) = mnist.load_data()

# plt.imshow(X_train[0])


def visualize_layer(
	model,
	layer_name,
	step=1.,
	epochs=15,
	upscaling_steps=9,
	upscaling_factor=1.2,
	output_dim=(412, 412),
	filter_range=(0, None)
):
	"""Visualizes the most relevant filters of one conv-layer in a certain model.

	# Arguments
		model: The model containing layer_name.
		layer_name: The name of the layer to be visualized.
			Has to be a part of model.
		step: step size for gradient ascent.
		epochs: Number of iterations for gradient ascent.
		upscaling_steps: Number of upscaling steps.
			Starting image is in this case (80, 80).
		upscaling_factor: Factor to which to slowly upgrade
			the image towards output_dim.
		output_dim: [img_width, img_height] The output image dimensions.
		filter_range: Tupel[lower, upper]
			Determines the to be computed filter numbers.
			If the second value is `None`,
			the last filter will be inferred as the upper boundary.
	"""

	def _generate_filter_image(
		input_img,
		layer_output,
		filter_index
	):
		"""Generates image for one particular filter.

		# Arguments
			input_img: The input-image Tensor.
			layer_output: The output-image Tensor.
			filter_index: The to be processed filter number.
			Assumed to be valid.

		# Returns
			Either None if no image could be generated.
			or a tuple of the image (array) itself and the last loss.
		"""
		start_time = time.time()

		# we build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		if K.image_data_format() == 'channels_first':
			loss = K.mean(layer_output[:, filter_index, :, :])
		else:
			loss = K.mean(layer_output[:, :, :, filter_index])

		# we compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]

		# normalization trick: we normalize the gradient
		grads = normalize(grads)

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		# we start from a gray image with some random noise
		intermediate_dim = tuple(
			int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
		if K.image_data_format() == 'channels_first':
			input_img_data = np.random.random((1, 3, intermediate_dim[0], intermediate_dim[1]))
		else:
			input_img_data = np.random.random((1, intermediate_dim[0], intermediate_dim[1], 3))
		input_img_data = (input_img_data - 0.5) * 20 + 128

		# Slowly upscaling towards the original size prevents
		# a dominating high-frequency of the to visualized structure
		# as it would occur if we directly compute the 412d-image.
		# Behaves as a better starting point for each following dimension
		# and therefore avoids poor local minima
		for up in reversed(range(upscaling_steps)):
			# we run gradient ascent for e.g. 20 steps
			for _ in range(epochs):
				loss_value, grads_value = iterate([input_img_data])
				input_img_data += grads_value * step

				# some filters get stuck to 0, we can skip them
				if loss_value <= K.epsilon():
					return None

				# Calulate upscaled dimension
				intermediate_dim = tuple(
					int(x / (upscaling_factor ** up)) for x in output_dim)
				# Upscale
				img = deprocess_image(input_img_data[0])
				img = np.array(pil_image.fromarray(img).resize(
					intermediate_dim,
					pil_image.BICUBIC)
				)
				input_img_data = [process_image(img, input_img_data[0])]

		# decode the resulting input image
		img = deprocess_image(input_img_data[0])
		end_time = time.time()
		print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(
			filter_index,
			loss_value,
			end_time - start_time
		))
		return img, loss_value

	def _draw_filters(filters, n=None):
		"""Draw the best filters in a nxn grid.

		# Arguments
			filters: A List of generated images and their corresponding losses for each processed filter.
			n: dimension of the grid.
				If none, the largest possible square will be used
		"""
		if n is None:
			n = int(np.floor(np.sqrt(len(filters))))

		# the filters that have the highest loss are assumed to be better-looking.
		# we will only keep the top n*n filters.
		filters.sort(key=lambda x: x[1], reverse=True)
		filters = filters[:n * n]

		# build a black picture with enough space for
		# e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
		MARGIN = 5
		width = n * output_dim[0] + (n - 1) * MARGIN
		height = n * output_dim[1] + (n - 1) * MARGIN
		stitched_filters = np.zeros((width, height, 3), dtype='uint8')

		# fill the picture with our saved filters
		for i in range(n):
			for j in range(n):
				img, _ = filters[i * n + j]
				width_margin = (output_dim[0] + MARGIN) * i
				height_margin = (output_dim[1] + MARGIN) * j
				stitched_filters[
					width_margin: width_margin + output_dim[0],
					height_margin: height_margin + output_dim[1],
					:
				] = img

		# save the result to disk
		save_img('vgg_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)

	# this is the placeholder for the input images
	assert len(model.inputs) == 1
	input_img = model.inputs[0]

	# get the symbolic outputs of each "key" layer (we gave them unique names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	output_layer = layer_dict[layer_name]
	assert isinstance(output_layer, layers.Conv2D)

	# Compute to be processed filter range
	filter_lower = filter_range[0]
	filter_upper = (
		filter_range[1]
			if filter_range[1] is not None
			else len(output_layer.get_weights()[1])
	)
	assert(
		filter_lower >= 0
			and filter_upper <= len(output_layer.get_weights()[1])
			and filter_upper > filter_lower
	)
	print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))

	# iterate through each filter and generate its corresponding image
	processed_filters = []
	for f in range(filter_lower, filter_upper):
		img_loss = _generate_filter_image(input_img, output_layer.output, f)

		if img_loss is not None:
			processed_filters.append(img_loss)

	print('{} filter processed.'.format(len(processed_filters)))
	# Finally draw and store the best filters to disk
	_draw_filters(processed_filters)
