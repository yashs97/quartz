
# Distributed Model Training

Neural network models are getting bigger and bigger. To keep up with this trend, it's important to be able to scale the hardware that we use to train these models. One way to do this is to use multiple accelerators, such as GPUs or TPUs. There are several different ways to train a model on multiple accelerators.

In this series, I'll be mainly covering advanced techniques like FSDP and GSPMD are but the first post will be a brief overview of Data and Model Parallelism is.

1. [Data and Model Parallelism](01_dp_and_mp.md)
2. [Full Sharded Data Parallel](distributed_model_training/03_fsdp.md)
3. [GSPMD](03_gspmd.md)
