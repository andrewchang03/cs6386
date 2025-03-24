# module load cudatoolkit/12.2
# module load gcc/12.2

.PHONY: main movielens

main:
	rm main
	nvcc main.cu -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcusparse -o main

alloc:
	salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account m4341

movielens:
	./main movielens 69879 10678 10000054
