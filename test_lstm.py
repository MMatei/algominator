from __future__ import division

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Masking

from tasks.copy import CopyTask, CopyFirstTask

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
maxSeq = 20
inSz = 8
model = Sequential()
# The masking layer will ensure that padded values are removed from consideration
model.add(Masking(mask_value=0, input_shape=(maxSeq, inSz)))
model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
model.add(LSTM(inSz, return_sequences=True))  # returns a sequence of vectors of dimension inSz
model.compile(loss='mean_squared_error', optimizer='rmsprop') # sgd is crap

task = CopyFirstTask(inSz, maxSeq)
inData, target = task.getData(10, 3000)
model.fit(inData, target, nb_epoch=15)
inData, target = task.getData(20, 200)
output = model.predict(inData)
allCorrect, bitsCorrect = analyzeRez(output, target, inSz)
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"