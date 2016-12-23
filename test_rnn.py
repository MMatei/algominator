from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time
from models.rnn import RNN
from models.sequential import Sequential

from tasks.copy import CopyTask, CopyFirstTask, IndexTask
from tasks.algo import GCDTask
from tasks.simpleOp import SumTask, ChoiceTask

def testModelOn(testX, testY, task):
	output = []
	for X in testX:
		model.reset(testX.shape[2]) # set to compute whole batch at once
		Y = model.forward(X)
		output.append(Y)
	output = np.array(output)
	loss = np.square(output - testY).sum() / testY.size
	allCorrect, bitsCorrect = task.analyzeRez(output, testY)
	return loss, allCorrect, bitsCorrect

ts1 = time.time()
task = CopyTask(8, 10)
batchSz = 30
model = Sequential()
model.add(RNN(8, 8, batchSz))
model.add(RNN(8, 8, batchSz))

ts2 = time.time()
print "Initialization completed in " + str(ts2 - ts1) +" seconds."

trainX, trainY = task.getData(10, 3000)
validX, validY = task.getData(10, 300)
testX, testY = task.getData(10, 300)
history = {'loss':[],'val_loss':[],'val_acc':[],'val_acce':[]}
ts1 = time.time()
print "Data generated in " + str(ts1 - ts2) +" seconds."

for epoch in range(0,50):
	avgLoss = 0
	for i in range(0,100):
		model.reset()
		seqX = trainX[:,:,batchSz*i:batchSz*(i+1)]
		Y = []
		for X in seqX:
			Y.append(model.forward(X))
		dY = np.array(Y) - trainY[:,:,batchSz*i:batchSz*(i+1)]
		model.backward(dY, 0.1)
		avgLoss += np.square(dY).sum()
	history['loss'].append(avgLoss/30000) # divide by total nr of outputs over all examples
	loss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
	history['val_loss'].append(loss)
	history['val_acc'].append(bitsCorrect)
	history['val_acce'].append(allCorrect)
ts2 = time.time()
print "Model trained in " + str(ts2 - ts1) +" seconds."

loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# summarize history for accuracy
plt.figure(1)
plt.plot(history['val_acc'])
plt.plot(history['val_acce'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['% numbers correct', '% bits correct'], loc='upper left')

# summarize history for loss
plt.figure(2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()