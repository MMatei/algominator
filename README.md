# Algominator

The (eventual) result of research into Algorithm-Learning Neural Systems.

The base folder contains a set of models, each of which is tested with the same barrage of tasks, that seek to prove the model's worthiness as algorithm-solving machine.

All tasks are defined in the task folder and they must inherit the Task abstract class. Task already provides a constructor that takes the problem's input size (which should remain fixed, since the model depends on this). You must then implement the getData(seqSz, batchSz) function, which returns a pair (input, target). This function will be used to generate data sets for both training and testing.

Needless to say, this is very much a work in progress.