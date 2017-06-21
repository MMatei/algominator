import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from tasks.copy import CopyTask, CopyFirstTask, IndexTask, MemoryTask
import numpy as np
from heatmap import heatmap

class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ninp, nhid, nlayers):
		super(RNNModel, self).__init__()
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, bias=False)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers
		self.l1loss = nn.L1Loss(size_average=False)
		self.regtarget = Variable(torch.from_numpy(np.zeros((32,8)))).float()

	def forward(self, input):
		return self.rnn(input)

	def loss(self, t, o, l1=0):
		l = (t - o).pow(2).sum() / (t.size(0) * t.size(1) * t.size(2))
		if l1 > 0:
			regLoss = 0
			for param in self.parameters():
				regLoss += self.l1loss(param, self.regtarget)
			l += regLoss * l1
		return l
		# t = (t + 1) / 2
		# o = (o + 1) / 2
		# perNumber = -(t * (o+0.00001).log() + (1 - t) * (1.00001 - o).log()).sum(dim=2) / t.size(2)
		# return perNumber.sum() / (t.size(0) * t.size(1))

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
					Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
		else:
			return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class MemoryModel(nn.Module):
	def __init__(self):
		super(MemoryModel, self).__init__()
		self.layer1 = nn.LSTM(21, 100)
		self.add_module("Layer1",self.layer1)
		self.layer2 = nn.LSTM(100, 10)
		self.add_module("Layer2",self.layer2)
	
	def forward(self, X):
		# X.cuda() # seems to not have an impact
		y, h = self.layer1(X)
		y, h = self.layer2(y)
		return y, h
	
	def loss(self, t, o):
		return (t - o).pow(2).sum() / (t.size(0) * t.size(1) * t.size(2))

def testModelOn(testX, testY, task):
	batchSz = testX.size()[1]
	y, h = model(testX)
	loss = model.loss(testY, y).data[0]
	allCorrect, bitsCorrect = task.analyzeRez(y.data.numpy(), testY.data.numpy())
	return loss, allCorrect, bitsCorrect

def threshPruning(params, thresh):
	for param in params:
		p = param.data.numpy()
		for i,row in enumerate(p):
			for j,col in enumerate(row):
				if abs(p[i,j]) < thresh:
					p[i,j] = 0
		param.data = torch.from_numpy(p)

def hardPruning(model, validX, validY, task):
	loss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
	for param in model.parameters():
		p = param.data.numpy()
		for i,row in enumerate(p):
			for j,col in enumerate(row):
				if abs(p[i][j]) <= 0.3:
					p[i][j] = 0
					continue # no reason to go through expensive testing
				p[i][j] = 0
				loss, allCorrect, bc = testModelOn(validX, validY, task)
				if bc >= bitsCorrect:
					bitsCorrect = bc
				else:
					p[i][j] = col
		param.data = torch.from_numpy(p)

ts1 = time.time()
task = CopyFirstTask(8, 30)
# task = MemoryTask()
# model = RNNModel('RNN_TANH', 8, 8, 1)
model = RNNModel('GRU', 8, 8, 1)
# model = MemoryModel()
# optimizer = optim.RMSprop(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
EPOCHS = 80
PRUNE_STEP = 20
BATCH_SIZE = 100
TRAIN_EXPLES = 3000
PASSES = TRAIN_EXPLES/BATCH_SIZE

ts2 = time.time()
print("Initialization completed in " + str(ts2 - ts1) +" seconds.")

trainX, trainY = task.getData(15, TRAIN_EXPLES)
trainX = Variable(torch.from_numpy(trainX)).float()
trainY = Variable(torch.from_numpy(trainY)).float()
validX, validY = task.getData(15, 300)
validX = Variable(torch.from_numpy(validX)).float()
validY = Variable(torch.from_numpy(validY)).float()
testMX, testMY = task.getData(20, 300)
testMX = Variable(torch.from_numpy(testMX)).float()
testMY = Variable(torch.from_numpy(testMY)).float()
testHX, testHY = task.getData(30, 300)
testHX = Variable(torch.from_numpy(testHX)).float()
testHY = Variable(torch.from_numpy(testHY)).float()
history = {'loss':[],'val_loss':[],'med_loss':[],'hrd_loss':[],'val_acc':[],'val_acce':[]}
ts1 = time.time()
print("Data generated in " + str(ts1 - ts2) +" seconds.")

for epoch in range(1,EPOCHS+1):
	trainLoss = 0
	for i in range(0, TRAIN_EXPLES, BATCH_SIZE):
		y, h = model(trainX[:,i:i+BATCH_SIZE])
		loss = model.loss(trainY[:,i:i+BATCH_SIZE], y)
		trainLoss += loss.data[0]
		loss.backward()
		optimizer.step()
	if epoch % PRUNE_STEP == 0:
		# threshPruning(model.parameters(), 0.1)
		hardPruning(model, validX, validY, task)
	validLoss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
	testMLoss, allCorrect, bitsCorrect = testModelOn(testMX, testMY, task)
	testHLoss, allCorrect, bitsCorrect = testModelOn(testHX, testHY, task)
	history['loss'].append(trainLoss/PASSES)
	history['val_loss'].append(validLoss)
	history['med_loss'].append(testMLoss)
	history['hrd_loss'].append(testHLoss)
	history['val_acc'].append(bitsCorrect*100)
	history['val_acce'].append(allCorrect*100)
ts2 = time.time()
print "Model trained in " + str(ts2 - ts1) +" seconds."

loss, allCorrect, bitsCorrect = testModelOn(testHX, testHY, task)
print loss
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# loss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
for param in model.parameters():
	# print(param.data)
	p = param.data.numpy()
	# heatmap(p)
	for i,row in enumerate(p):
		for j,col in enumerate(row):
			p[i][j] = round(p[i][j])
			loss, allCorrect, bc = testModelOn(validX, validY, task)
			if bc >= bitsCorrect:
				bitsCorrect = bc
			else:
				p[i][j] = col
	param.data = torch.from_numpy(p)
	# print(param.data)
	# heatmap(p)
loss, allCorrect, bitsCorrect = testModelOn(testHX, testHY, task)
print loss
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

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
plt.plot(history['med_loss'])
plt.plot(history['hrd_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation', '20 seq', '30 seq'], loc='upper left')
plt.show()