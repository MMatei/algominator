# Algominator

The (eventual) result of research into Algorithm-Learning Neural Systems.

The base folder contains a set of models, each of which is tested with the same barrage of tasks, that seek to prove the model's worthiness as algorithm-solving machine.

All tasks are defined in the tasks folder and they must inherit one of the abstract classes in task.py:
 * **Task** represents the most basic kind of task, with a sequence as input and expecting a sequence as output.
 * **NoSeqTask** extends Task for those problems where we only provide a single output (instead of a sequence).

Regardless of your choice, you must implement the `getData(seqSz, batchSz)` function, which returns a pair (input, target). This function will be used to generate data sets for both training and testing. Please note that `getData` should always return a balanced data set; otherwise, the network will not learn an algorithm, but rather shortcuts that allow it to artificially inflate accuracy. For this reason, **Task** also defines `getDataUnbalanced(seqSz, batchSz)`, so that you can test the network on a realistic data set.

**Task** contains the useful `toBinary(number)` function, which will transform a given number into an array of 0/1 of length `self.inputSz`. **Task** also implements `analyzeRez(output, target)`, which returns the accurracy of the output compared to the given target.

If you wish to change the loss function, write the appropriate keras loss function string in `yourTask.loss`. If in doubt, check task.py, as well as the default test_lstm.py for clues.

Needless to say, this is very much a work in progress.