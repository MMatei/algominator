# The archetypal task, which all tasks must inherit
class Task:
	# Fix the input size, as well as the maximum sequence length that the model can support
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.maxSeq = maxSeq

	# Given a sequence size and batch size, return a dataset (input, target)
	# Each is a numpy array of shape (batchSz, seqSz, inputSz)
	def getData(self, seqSz, batchSz):
		raise NotImplementedError