class Sequential:
	def __init__(self):
		self.layers = []

	def add(self, layer):
		self.layers.append(layer)

	def reset(self, batchSize=None):
		self.histoX = [[]] * 32 # if bigger possible sequences, alter this
		for layer in self.layers:
			layer.reset(batchSize)

	def forward(self, X):
		for i, layer in enumerate(self.layers):
			self.histoX[i].append(X)
			X = layer.forward(X)
		return X

	def backward(self, loss, lr):
		for l in range(len(self.layers)-1,-1,-1):
			loss = self.layers[l].backward(self.histoX[l], loss)
			self.layers[l].updateParams(lr)