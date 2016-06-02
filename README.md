# parallel-neural-net

##Dependencies:
* Windows
* Visual Studio Command Prompt


##To install:

Open Visual Studio Command Line and clone the repository
```
git clone https://github.com/connorlee77/parallel-neural-net.git

cd parallel-neural-net
``` 

##To run

Change the file path in line 5 at the top of `ANN.h` to your own file path. Note that this must be the absolute path. Then, to make, type 
```
make
```

To run the application, type
```
ANN
```

The number and size of hidden layers can be changed in `ANN.cc`. The default hyperparameters should lead to 0.98 accuracy after 10 epochs.


##About  
This is a fully-connected parallelized neural network written with CUDA C++. The network is currently setup to classify MNIST handwritten digits to easily around 98.5% accuracy using stochastic gradient descent. The network utilizes cross entropy loss along with a softmax activation function and L1 regularization.

For neural networks, there exist six possibly ways to achieve parallelism. This ranges from:

* Epoch level
* Data sample level
* Layer level
* Neuron level
* Weight level
* Bit level

The lower 3 and topmost levels of parllelism are clearly impractical. While most neural network implementations parallelize over iterations of training examples, this implementation parallelizes over the matrix operations in each layer of the network. We seek to establish the tradeoff between the two more practical parallelization strategies by analyzing the GPU/CPU speedups from layer parallelization. 

It is reasonable to believe the hypothesis that the GPU code will see relatively little speedup when the neural network is small. 

###Parallelization techniques
All the memory was allocated on the GPU except for the dataset itself. This includes weights, bias weights, weight gradients, deltas, and bias gradients. Keeping as much memory on the device as possible prevented unnecessary IO latencies to slow down the parallization. 

One key parallelization technique that greatly improved the speed of this network was using CUDA streams. This essentially hid the latency associated with host/device data transfers. Streaming was used to copy batches of training data from host memory to device memory.

The multiplication kernels were all optimized to perform a specific type of matrix/vector multiplication. This was done because different types of vector/matrix multiplications allowed for better coalesced memory accesses or usage of shared memory. 

In addition to optimizing the multiplication kernels, we reduce the number of unnecessary cudaMemcpy's by combining the copies with other kernels. This meant that kernels in which threads operate on the same size as what needed to be copied were used to perform the copies. This reduced the number of kernel launches needed and thus, increased the speed slightly. 

###Results

The error rates acheived by the network after 10 epochs on the default setting was consistently around 2%. The error rates decreases to around 1.3% after more epochs of stochastic gradient descent is run. 

The computation time used for one epoch (a full pass over the training set) is shown for GPU and CPU code of the same layer size and structure.

|Layer Structure				| GPU	| CPU  |
|----------------------------------------------|
|[784, 300, 10]					| 28.5s | 32.0s|
|[784, 300, 300, 10]			| 42.5s | 44.3s|
|[784, 800, 300, 300, 10]		| 120s	| 136s |
|[784, 800, 800, 300, 10]		| 181s 	| 223s |
|[784, 800, 800, 800, 10]		| 225s	| 284s |
|[784, 800, 800, 800, 800, 10]	| 286s	| 386s |

From the table above, it is clear that our initial hypothesis was correct. The GPU and CPU code had relatively similar times for networks of less and smaller hidden layers. As the layers and size increase, the difference between the GPU and CPU times also increase. 

From the experiments ran, it is clear that parallelizing across the layer level is slower compared to parallelizing across data samples. This is most likely due to numerous data transfers between the various layers of the neural network. Thus, parallelizing data samples would have an edge by computing batches of data all at the same time instead of sequentially. However, parallelizing across the layer level allows for fast stochastic gradient descent which has been shown to lead to faster convergence times as compared to batch gradient descent. 

