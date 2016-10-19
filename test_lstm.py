from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Masking

from tasks.copy import CopyTask, CopyFirstTask, IndexTask
from tasks.algo import GCDTask
from tasks.simpleOp import SumTask, ChoiceTask

ts1 = time.time()
# task = CopyTask(8, 20)
task = ChoiceTask(16)
# Baseline model: a LSTM with 2 layers
model = Sequential()
# The masking layer will ensure that padded values are removed from consideration
model.add(Masking(mask_value=0, input_shape=(task.maxSeq, task.inputSz)))
model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
model.add(LSTM(task.outputSz, return_sequences=task.sequenceOut))
model.compile(loss=task.loss, optimizer='rmsprop', metrics=['accuracy']) # sgd is crap

ts2 = time.time()
print "Initialization completed in " + str(ts2 - ts1) +" seconds."

trainX, trainY = task.getData(12, 3300)
testX, testY = task.getData(16, 200)
ts1 = time.time()
print "Data generated in " + str(ts1 - ts2) +" seconds."

history = model.fit(trainX, trainY, nb_epoch=20, validation_split=0.1, verbose=2)
ts2 = time.time()
print "Model trained in " + str(ts2 - ts1) +" seconds."

output = model.predict(testX)
allCorrect, bitsCorrect = task.analyzeRez(output, testY)
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