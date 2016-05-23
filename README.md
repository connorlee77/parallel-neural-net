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
'''
ANN
'''

The number and size of hidden layers can be changed in `ANN.cc`. The default hyperparameters should lead to ~0.98 accuracy after 10 epochs. 