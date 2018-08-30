# Visualizing Loss Functions

In DeepMind's recent paper [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508),
they showed that NALUs and NACs (neural accumulators) perform better that RNNs,
LSTMs, and vanilla MLPs in tasks that involve some notion of counting, keeping
track of numbers, and approximating mathematical functions. While such
effectiveness at numerical tasks is obvious from their internal architecture,
I was (and still am) wondering how much of it could be explained by the shape
of the loss function graphed against a few randomly chosen weights.

In the case of a simple MLP (2 hidden units!) vs. a simple NAC (2 hidden
units as well) being trained to learn addition, the MLP loss surface has
subtle local minima, and the MLP requires multiple training samples to
fully understand the task. But the NAC loss surface monotonically
changes in the correct direction even with a single training sample! This is
because the architecture of the NAC forces it to learn a smaller range of tasks
than an MLP, allowing it to converge with fewer training samples.

If I have more time, I want to compare an MLP with a NALU network on the MNIST
counting task as described in the original paper and see how the loss surfaces
differ.