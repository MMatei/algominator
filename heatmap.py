import matplotlib.pyplot as plt

def heatmap(matrix):
	plt.imshow(matrix, interpolation='none')
	plt.colorbar()
	plt.xticks(range(0,matrix.shape[1]),range(0,matrix.shape[1]))
	plt.yticks(range(0,matrix.shape[0]),range(0,matrix.shape[0]))
	plt.show()