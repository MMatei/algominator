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
	# Define the output size as needed; this is the purview of the model
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = 0
		self.maxSeq = maxSeq

	# Given a sequence size and batch size, return a dataset (input, target)
	# Each is a numpy array of shape (batchSz, seqSz, inputSz)
	def getData(self, seqSz, batchSz):
		raise NotImplementedError

	# The dataset returned by getData is BALANCED
	# But, depending on the problem, this may not be realistic
	# In such cases, use this second function to obtain testing examples
	def getDataUnbalanced(self, seqSz, batchSz):
		return getData(self, seqSz, batchSz)

	def toBinary(self, number):
		return np.fromstring(np.binary_repr(number, width=self.inputSz), dtype=np.uint8) - 48

	# Return accurracy of output compared to target as a pair of
	# (wholeOutCorrect, bitsCorrect)
	def analyzeRez(self, output, target):
		wholeOutCorrect = 0
		bitsCorrect = 0
		# we get numbers between 0 and 1, we need bits 0/1
		# output = np.around(output).astype(int)
		output[output <= 0] = 0
		output[output > 0] = 1
		for i, batch in enumerate(output):
			for j in range(0, output.shape[2]): # output.shape[2] = batchSz
				errors = int(np.absolute(batch[:,j] - target[i,:,j]).sum())
				bitsCorrect += (self.outputSz - errors)
				if errors == 0:
					wholeOutCorrect += 1
		return (wholeOutCorrect / (output.size/self.outputSz), bitsCorrect/output.size)

# A task that doesn't output a sequence
# Therefore, its result has a different shape (no sequence length!), and thus a different analyzeRez
class NoSeqTask(Task):
	sequenceOut = False

	def analyzeRez(self, output, target):
		wholeOutCorrect = 0
		bitsCorrect = 0
		# we get numbers between 0 and 1, we need bits 0/1
		output = np.around(output).astype(int)
		for i in range(0, output.shape[1]):
			errors = int(np.absolute(output[:,i] - target[:,i]).sum())
			bitsCorrect += (self.outputSz - errors)
			if errors == 0:
				wholeOutCorrect += 1
		return (wholeOutCorrect / (output.size/self.outputSz), bitsCorrect/output.size)