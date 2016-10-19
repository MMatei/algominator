from __future__ import division
import numpy as np
from random import randint
from task import NoSeqTask

# Given two values, output their greatest common divisor
# each value is represented by valLen bits
class GCDTask(NoSeqTask):
	def __init__(self, valLen):
		self.valLen = valLen
		self.inputSz = 2 * valLen
		self.outputSz = valLen
		self.maxSeq = 1

	def gcd(self, a, b):
		while a != b:
			if a > b:
				a -= b
			else:
				b -= a
		return a

    # For the purposes of this task, seqSz represents the nr of bits of the values in data, 0 < seqSz <= valLen
	def getData(self, seqSz, batchSz):
		if seqSz > self.valLen:
			raise Exception('seqSz exceeds max value length', str(seqSz)+' > '+str(self.valLen))

		inData = np.zeros((batchSz, self.inputSz))
		target = np.zeros((batchSz, self.outputSz))
		maxVal = pow(2, seqSz) - 1
		for i in range(0, batchSz):
			a = randint(1, maxVal)
			b = randint(1, maxVal)
			c = self.gcd(a, b)
			inData[i,:self.valLen] = self.toBinary(a)[self.valLen:]
			inData[i,self.valLen:] = self.toBinary(b)[self.valLen:]
			target[i,:] = self.toBinary(c)[self.valLen:]
		return (inData.reshape((batchSz,1,self.inputSz)), target)