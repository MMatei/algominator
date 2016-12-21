import numpy as np

class RNN:
	def __init__(self, numIn, cells, batchSize):
		self.W_xh = np.random.uniform(-0.01,0.01,(cells,numIn))
		self.W_hh = np.random.uniform(-0.01,0.01,(cells,cells))
		self.W_hy = np.random.uniform(-0.01,0.01,(cells,cells))
		self.bh = np.zeros((cells,1)) # hidden bias
		self.by = np.zeros((cells,1)) # output bias
		self.cells = cells
		self.batchSize = batchSize
		# Initialize momentum to 0
		self.mWxh = np.zeros_like(self.W_xh)
		self.mWhh = np.zeros_like(self.W_hh)
		self.mWhy = np.zeros_like(self.W_hy)
		self.mbh = np.zeros_like(self.bh)
		self.mby = np.zeros_like(self.by)

	# call before every new batch, in order to clear hidden state; batch size can be dinamically adjusted for ease of use in testing
	def reset(self, batchSize=None):
		if batchSize is None:
			batchSize = self.batchSize
		self.h = np.zeros((self.cells, batchSize)) # hidden state
		self.H = [] # history of hidden states

	def forward(self, X):
		# update the hidden state
		self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, X) + self.bh)
		self.H.append(self.h)
		# compute the output vector
		return np.dot(self.W_hy, self.h) + self.by

	def backward(self, X, loss):
		self.dWxh = np.zeros_like(self.W_xh)
		self.dWhh = np.zeros_like(self.W_hh)
		self.dWhy = np.zeros_like(self.W_hy)
		self.dbh = np.zeros_like(self.bh)
		self.dby = np.zeros_like(self.by)
		dhnext = np.zeros_like(self.h)
		lossDown = [0] * len(loss)
		for t in range(len(loss)-1,-1,-1):
			dy = loss[t]
			self.dWhy += np.dot(dy, self.H[t].T)
			self.dby += np.sum(dy,axis=1,keepdims=True)
			dh = np.dot(self.W_hy.T, dy) + dhnext # backprop into h
			dhraw = (1 - self.H[t] * self.H[t]) * dh # backprop through tanh nonlinearity
			self.dbh += np.sum(dhraw,axis=1,keepdims=True)
			self.dWxh += np.dot(dhraw, X[t].T)
			self.dWhh += np.dot(dhraw, self.H[t-1].T)
			dhnext = np.dot(self.W_hh.T, dhraw)
			lossDown[t] = np.dot(self.W_xh.T, dhraw) # the input gradient, sent downwards
		for dparam in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return lossDown

	def updateParams(self, lr):
		# perform parameter update with Adagrad
		for param, dparam, mom in zip([self.W_xh, self.W_hh, self.W_hy, self.bh, self.by], [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby], [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
			mom += dparam * dparam
			param += -lr * dparam / np.sqrt(mom + 1e-8) # adagrad update

# No batch computation - both inefficient and with far poorer results; leaving this here for historical purposes
class BasicRNN:
	def __init__(self, numIn, cells):
		self.W_xh = np.random.uniform(-0.01,0.01,(cells,numIn))
		self.W_hh = np.random.uniform(-0.01,0.01,(cells,cells))
		self.W_hy = np.random.uniform(-0.01,0.01,(cells,cells))
		self.bh = np.zeros(cells) # hidden bias
		self.by = np.zeros(cells) # output bias
		# Initialize momentum to 0
		self.mWxh = np.zeros_like(self.W_xh)
		self.mWhh = np.zeros_like(self.W_hh)
		self.mWhy = np.zeros_like(self.W_hy)
		self.mbh = np.zeros_like(self.bh)
		self.mby = np.zeros_like(self.by)

	# call before every new batch, in order to clear hidden state
	def reset(self):
		self.h = np.zeros_like(self.bh) # hidden state
		self.H = [] # history of hidden states

	def forward(self, X):
		# update the hidden state
		self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, X) + self.bh)
		self.H.append(self.h)
		# compute the output vector
		return np.dot(self.W_hy, self.h) + self.by

	def backward(self, X, loss):
		self.dWxh = np.zeros_like(self.W_xh)
		self.dWhh = np.zeros_like(self.W_hh)
		self.dWhy = np.zeros_like(self.W_hy)
		self.dbh = np.zeros_like(self.bh)
		self.dby = np.zeros_like(self.by)
		dhnext = np.zeros_like(self.h)
		lossDown = [0] * len(loss)
		for t in range(len(loss)-1,-1,-1):
			dy = loss[t]
			self.dWhy += np.dot(dy, self.H[t].T)
			self.dby += dy
			dh = np.dot(self.W_hy.T, dy) + dhnext # backprop into h
			dhraw = (1 - self.H[t] * self.H[t]) * dh # backprop through tanh nonlinearity
			self.dbh += dhraw
			self.dWxh += np.dot(dhraw, X[t].T)
			self.dWhh += np.dot(dhraw, self.H[t-1].T)
			dhnext = np.dot(self.W_hh.T, dhraw)
			lossDown[t] = np.dot(self.W_xh.T, dhraw) # the input gradient, sent downwards
		for dparam in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return lossDown

	def updateParams(self, lr):
		# perform parameter update with Adagrad
		for param, dparam, mom in zip([self.W_xh, self.W_hh, self.W_hy, self.bh, self.by], [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby], [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
			mom += dparam * dparam
			param += -lr * dparam / np.sqrt(mom + 1e-8) # adagrad update