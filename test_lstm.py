from __future__ import division

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Masking

import matplotlib
import matplotlib.pyplot as plt

from tasks.copy import CopyTask, CopyFirstTask, IndexTask
from tasks.algo import GCDTask

# task = CopyTask(8, 20)
task = GCDTask(8)
# Baseline model: a LSTM with 2 layers
model = Sequential()
# The masking layer will ensure that padded values are removed from consideration
model.add(Masking(mask_value=0, input_shape=(task.maxSeq, task.inputSz)))
model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
model.add(LSTM(task.outputSz, return_sequences=task.sequenceOut))
model.compile(loss=task.loss, optimizer='rmsprop', metrics=['accuracy']) # sgd is crap

inData, target = task.getData(6, 3300)
history = model.fit(inData, target, nb_epoch=20, validation_split=0.1, verbose=0)
inData, target = task.getData(8, 200) # test data
output = model.predict(inData)
allCorrect, bitsCorrect = task.analyzeRez(output, target)
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()