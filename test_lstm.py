from __future__ import division

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Masking

from tasks.copy import CopyTask, CopyFirstTask, IndexTask

maxSeq = 20
inSz = 8
task = IndexTask(inSz, maxSeq)
# Baseline model: a LSTM with 2 layers
model = Sequential()
# The masking layer will ensure that padded values are removed from consideration
model.add(Masking(mask_value=0, input_shape=(maxSeq, inSz)))
model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
model.add(LSTM(inSz, return_sequences=task.sequenceOut))
model.compile(loss=task.loss, optimizer='rmsprop') # sgd is crap

inData, target = task.getData(10, 3000)
model.fit(inData, target, nb_epoch=15)
inData, target = task.getData(20, 200)
output = model.predict(inData)
allCorrect, bitsCorrect = task.analyzeRez(output, target, inSz)
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"