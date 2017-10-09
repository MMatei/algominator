import numpy as np

class IdealAddModule():
	def __init__(self):
		self.carry = 0

	def forward(self, x1, x2):
		# x1, x2 are digits represented as one-hot encoding of size 11, where class 11 represents 'null' value
		x1 = x1.argmax()
		x2 = x2.argmax()
		y = -np.ones(11)
		if x1 == 10 or x2 == 10: # if one of inputs is null, no valid computation can be offered
			y[self.carry] = 1
			self.carry = 0
			return y
		_y = x1 + x2 + self.carry
		y [_y % 10] = 1
		self.carry = _y / 10
		return y

class IdealSubModule():
	def __init__(self):
		self.carry = 0

	def forward(self, x1, x2):
		x1 = x1.argmax()
		x2 = x2.argmax()
		y = -np.ones(11)
		if x1 == 10 or x2 == 10:
			y[self.carry] = 1
			self.carry = 0
			return y
		_y = x1 - x2 - self.carry
		self.carry = 0 # carry got used
		if _y < 0:
			self.carry = 1
			_y = 10 + _y
		y [_y] = 1
		return y

class IdealMemModule():
	def __init__(self, N):
		self.mem = -np.ones((N,11))
		self.mem[:,10] = 1 # initialize all values to null

	def forward(self, x):
		if x[21] == 1: # read from memory
			addr = x[11:21].argmax()
			return self.mem[addr]
		else:
			addr = x[11:21].argmax()
			self.mem[addr] = x[:11]
			return self.mem[addr]