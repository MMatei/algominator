from __future__ import division
import numpy as np

# The archetypal task, which all tasks must inherit
class Task:
	# True if we expect (and correct) output at every timestep
	# False if we only care about the final output value
	sequenceOut = True
	# The objective function for this task
	loss = 'mean_squared_error'

	# Fix the input size, as well as the maximum sequence length that the model can support
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.maxSeq = maxSeq

	# Given a sequence size and batch size, return a dataset (input, target)
	# Each is a numpy array of shape (batchSz, seqSz, inputSz)
	def getData(self, seqSz, batchSz):
		raise NotImplementedError

	def toBinary(self, number):
		return np.fromstring(np.binary_repr(number, width=self.inputSz), dtype=np.uint8) - 48

	# Return accurracy of output compared to target as a pair of
	# (wholeOutCorrect, bitsCorrect)
	def analyzeRez(self, output, target, outSz):
		wholeOutCorrect = 0
		bitsCorrect = 0
		# we get numbers between 0 and 1, we need bits 0/1
		output = np.around(output)
		for i, sequence in enumerate(output):
			for j, val in enumerate(sequence):
				errors = int(np.absolute(val - target[i][j]).sum())
				bitsCorrect += (outSz - errors)
				if errors == 0:
					wholeOutCorrect += 1
		return (wholeOutCorrect/(output.size / outSz), bitsCorrect/output.size)