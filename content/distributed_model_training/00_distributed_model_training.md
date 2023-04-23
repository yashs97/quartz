
# Distributed Model Training

Neural network models are getting bigger and bigger. To keep up with this trend, it's important to be able to scale the hardware that we use to train these models. One way to do this is to use multiple accelerators, such as GPUs or TPUs. There are several different ways to train a model on multiple accelerators.

In this series, I'll be mainly covering the following techniques:

1. [Data Parallelism](01_dp.md)
2. [Model Parallelism](distributed_model_training/02_mp.md)
3. [Full Sharded Data Parallel](distributed_model_training/02_fsdp.md)
4. [GSPMD](04_gspmd.md)
