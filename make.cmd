@echo off
nvcc -O3 ANN.cc parse.cc helpers.cc ANN.cu -lcurand -o ANN -arch=sm_20

IF EXIST ANN.lib. (
    del ANN.lib
)
IF EXIST ANN.exp. (
	del ANN.exp
)