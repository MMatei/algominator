import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from tasks.copy import MemoryTask
import numpy as np

class MemoryModel(nn.Module):
	def __init__(self):
		super(MemoryModel, self).__init__()
		self.layer1 = nn.LSTM(22, 110)
		self.add_module("Layer1",self.layer1)
		self.layer2 = nn.LSTM(110, 11)
		self.add_module("Layer2",self.layer2)
		self.sm = nn.LogSoftmax()
		self.ll = nn.NLLLoss()
	
	def forward(self, X, target, optimizer=None):
		# X.cuda()
		# target.cuda()
		y, _ = self.layer1(X)
		y, _ = self.layer2(y)
		output = []
		lss = 0
		for i in range(0,y.size(0)):
			aux = self.sm(y[i])
			output.append(aux.data.numpy())
			l = self.ll(aux, target[i])
			lss += l.data[0]
			# RuntimeError: Trying to backward through the graph second time, but the buffers have already been freed. Please specify retain_variables=True when calling backward for the first time.
			l.backward(retain_variables=True)
			# if optimizer is not None:
			# 	optimizer.step()
		return output, lss

def testModelOn(testX, testY, task):
	batchSz = testX.size()[1]
	y, loss = model.forward(testX, testY)
	allCorrect, bitsCorrect = task.analyzeRez(y, testY.data.numpy())
	return loss, allCorrect, bitsCorrect

def threshPruning(params, thresh):
	for param in params:
		p = param.data.numpy()
		for i,row in enumerate(p):
			if not isinstance(row, np.float32):
				for j,col in enumerate(row):
					if abs(p[i,j]) < thresh:
						p[i,j] = 0
		param.data = torch.from_numpy(p)

def hardPruning(model, validX, validY, task):
	loss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
	for param in model.parameters():
		p = param.data.numpy()
		for i,row in enumerate(p):
			if isinstance(row, np.ndarray):
				for j,col in enumerate(row):
					if abs(p[i][j]) <= 0.3: #combine with thresh pruning
						p[i][j] = 0
						continue
					p[i][j] = 0
					loss, ac, bc = testModelOn(validX, validY, task)
					if ac >= allCorrect:
						allCorrect = ac
					else:
						p[i][j] = col
		param.data = torch.from_numpy(p)

ts1 = time.time()
task = MemoryTask()
model = MemoryModel()
# optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
BATCH_SIZE = 100
TRAIN_EXPLES = 3000
PASSES = TRAIN_EXPLES/BATCH_SIZE

ts2 = time.time()
print("Initialization completed in " + str(ts2 - ts1) +" seconds.")

trainX, trainY = task.getDataStress(25, TRAIN_EXPLES, True)
trainX = Variable(torch.from_numpy(trainX)).float()
trainY = Variable(torch.from_numpy(trainY)).long()
validX, validY = task.getDataStress(25, 300, True)
validX = Variable(torch.from_numpy(validX)).float()
validY = Variable(torch.from_numpy(validY)).long()
testX, testY = task.getDataStress(25, 300, True)
testX = Variable(torch.from_numpy(testX)).float()
testY = Variable(torch.from_numpy(testY)).long()
history = {'loss':[],'val_loss':[],'val_acc':[],'val_acce':[]}
ts1 = time.time()
print("Data generated in " + str(ts1 - ts2) +" seconds.")

for epoch in range(1,45):
	trainLoss = 0
	for i in range(0, TRAIN_EXPLES, BATCH_SIZE):
		trainLoss += model.forward(trainX[:,i:i+BATCH_SIZE], trainY[:,i:i+BATCH_SIZE], optimizer)[1]
		optimizer.step()
	if epoch % 20 == 0:
		# threshPruning(model.parameters(), 0.1)
		hardPruning(model, validX, validY, task)
	validLoss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
	history['loss'].append(trainLoss/PASSES)
	history['val_loss'].append(validLoss)
	history['val_acc'].append(bitsCorrect*100)
	history['val_acce'].append(allCorrect*100)
	print(epoch)
	print(allCorrect)
ts2 = time.time()
print "Model trained in " + str(ts2 - ts1) +" seconds."

loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
print loss
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# for param in model.parameters():
# 	# print(param.data)
# 	p = param.data.numpy()
# 	for i,row in enumerate(p):
# 		for j,col in enumerate(row):
# 			p[i][j] = 0
# 			loss, allCorrect, bc = testModelOn(validX, validY, task)
# 			if bc >= bitsCorrect:
# 				bitsCorrect = bc
# 			else:
# 				p[i][j] = col
# 	# param.data = torch.from_numpy(param.data.numpy().round())
# 	param.data = torch.from_numpy(p)
# 	# print(param.data)
# loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
# print loss
# print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
# print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# summarize history for accuracy
plt.figure(1)
plt.plot(history['val_acc'])
plt.plot(history['val_acce'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['% numbers correct', '% bits correct'], loc='upper left')
axes = plt.gca()
axes.set_ylim([0,100])

# summarize history for loss
plt.figure(2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()