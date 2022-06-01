Required library: 
 - tensorflow-gpu(recommended) or tensorflow
 
Software Requirement:
 - CUDA 9(fully support tensorflow)
 - GPU drivers
 - CUPTI
 - cuDNN SDK

Experiment start point: run_batch.py

Setting:
 - having tensorflow-gpu ready: https://www.tensorflow.org/install/gpu
 - (optional) TensorBoard: included in tensorflow. Run with code: tensorboard --logdir=path/to/log-directory. Note: the path should paired to the one in DQN code.
