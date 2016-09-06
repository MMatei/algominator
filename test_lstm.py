from __future__ import division

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM

# Returns a pair of (inputData, expectedOutput)
# input size, sequence size, batch size
def getData(inSz, seqSz, btSz):
	inData = np.random.binomial(1,0.5,size=(btSz, seqSz, inSz))
	return (inData, inData.copy())

# Return accurracy of output compared to target as a pair of
# (wholeOutCorrect, bitsCorrect)
def analyzeRez(output, target, outSz):
	wholeOutCorrect = 0
	bitsCorrect = 0
	# we get numbers between 0 and 1, we need bits 0/1
	output = np.around(output)
	for i, sequence in enumerate(output):
		for j, val in enumerate(sequence):
			errors = int(np.absolute(val - target[i][j]).sum())
			bitsCorrect += (outSz - errors)
			if errors == 0:
				wholeOutCorrect += 1
	return (wholeOutCorrect/(len(output) * len(output[0])), bitsCorrect/(len(output) * len(output[0]) * len(output[0][0])))

# Baseline model: a LSTM with 2 layers
seqSz = 10
inSz = 8
model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(seqSz, inSz)))  # returns a sequence of vectors of dimension 16
model.add(LSTM(inSz, return_sequences=True))  # returns a sequence of vectors of dimension inSz
model.compile(loss='mean_squared_error', optimizer='rmsprop')
#model.compile(loss='mean_squared_error', optimizer='sgd')

data = getData(inSz, seqSz, 3000)
model.fit(data[0], data[1], nb_epoch=10)
data = getData(inSz, seqSz, 200)
output = model.predict(data[0])
print analyzeRez(output, data[1], inSz)