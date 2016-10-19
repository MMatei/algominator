from __future__ import division
import numpy as np
from random import randint
from task import NoSeqTask
import math

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
    # We select random inputs, but this produces an unbalanced data set
    # This should probably be used for testing, but almost certainly not for testing
	def getDataUnbalanced(self, seqSz, batchSz):
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

	def getData(self, seqSz, batchSz):
		if seqSz > self.valLen:
			raise Exception('seqSz exceeds max value length', str(seqSz)+' > '+str(self.valLen))

		inData = np.zeros((batchSz, self.inputSz))
		target = np.zeros((batchSz, self.outputSz))
		maxVal = int(math.sqrt(pow(2, seqSz) - 1))
		for i in range(0, batchSz):
			a = randint(1, maxVal)
			c = randint(1, maxVal)
			b = a * c
			inData[i,:self.valLen] = self.toBinary(a)[self.valLen:]
			inData[i,self.valLen:] = self.toBinary(b)[self.valLen:]
			target[i,:] = self.toBinary(c)[self.valLen:]
		return (inData.reshape((batchSz,1,self.inputSz)), target)