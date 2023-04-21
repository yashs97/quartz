
# Distributed Model Training

The size of neural network models has been increasing rapidly. In order to meet these every increasing scaling needs, horizontal scaling of GPUs/TPUs is of utmost importance. There several paradigms to enable model training across multiple hardware accelerators.

In this series, I'll be mainly covering the following techniques:

1. [Distributed Data Parallel](distributed_model_training/01_ddp.md)
2. [Full Sharded Data Parallel](02_fsdp.md)
3. [GSPMD](distributed_model_training/03_gspmd.md)
