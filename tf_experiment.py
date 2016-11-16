import math
import time
import tensorflow as tf
from tasks.algo import GCDTask

task = GCDTask(16)
maxCycles = int(math.sqrt(pow(2, 16) - 1))

computeModel = tf.nn.rnn_cell.MultiRNNCell([
	tf.nn.rnn_cell.LSTMCell(32),
	tf.nn.rnn_cell.LSTMCell(32),
	tf.nn.rnn_cell.LSTMCell(task.outputSz)])
doneModel = tf.nn.rnn_cell.MultiRNNCell([
	tf.nn.rnn_cell.LSTMCell(32),
	tf.nn.rnn_cell.LSTMCell(16),
	tf.nn.rnn_cell.LSTMCell(1)])
# Can't operate on batches because different inputs might stop at different times
zeroOut = output = tf.zeros([1, task.outputSz], tf.float64)
done = tf.zeros([1, 1], tf.float64)
computeState = computeModel.zero_state(1, tf.float64)
doneState = doneModel.zero_state(1, tf.float64)
stopCondition = tf.constant([[0.5]], tf.float64)
notDone = tf.constant([[1]], tf.float64)
yesDone = tf.constant([[0]], tf.float64)
iteration = -1

def f():
	global X, origIn, output, done, computeState, doneState
	with tf.variable_scope("gigi1") as scope:
		if iteration > 0:
			scope.reuse_variables()
		output, computeState = computeModel(X, computeState)
	X = tf.concat(1, [origIn, output])
	with tf.variable_scope("gigi2") as scope:
		if iteration > 0:
			scope.reuse_variables()
		done, doneState = doneModel(X, doneState)
	return [output, done]

def g():
	global output, done
	return [output, done]

def forward(_input):
	global iteration
	global origIn, output
	global computeState, doneState, X
	# This is arguably helping the net by providing free memory; to discuss
	origIn = _input
	X = tf.concat(1, [_input, output])
	for iteration in range(0, maxCycles):
		tf.cond(tf.greater(done, stopCondition)[0,0], g, f)
	iteration = -1
	return output, iteration

# iter is the number of the iteration at which the model stopped
# it is -1 if the model didn't stop of its own accord
def loss(target):
	global zeroOut, yesDone, notDone
	global output, done, iteration
	# model fails to stop, no error for computation, serios error for doneModel
	if iteration < 0:
		return tf.mul(output, zeroOut), tf.square(tf.sub(notDone, done))
	# model has stopped; mse for output, no error for done
	else:
		return tf.square(tf.sub(target, output)), tf.mul(done, yesDone)

LR = 0.001
def train(e1, e2):
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		train_step = tf.train.GradientDescentOptimizer(LR).minimize(e1)
		print 'running session 1...'
		session.run(train_step)
		print 'session 1 done'
		train_step = tf.train.GradientDescentOptimizer(LR).minimize(e2)
		print 'running session 2...'
		session.run(train_step)
		print 'session 2 done'

trainX, trainY = task.getData(12, 3300)
testX, testY = task.getDataUnbalanced(16, 200)

# for X,T in zip(trainX, trainY):
X = trainX[0]
T = trainY[0]
print 'got data'
Y = forward(tf.Variable(X))
e1, e2 = loss(T)
print 'computational graph built'
train(e1, e2)
print 'THE END'