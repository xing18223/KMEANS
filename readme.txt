// Author: Xingtang Sun - ni18223@bristol.ac.uk
// Date: 25th April 2022

K-means clustering benchmark for running on the ZCU102 Ultrascale+ heterogeneous platform

1) ZCU102 system setup: load interrupt driver and FPGA bitstream via system.sh

	./system.sh -n 1 -f zcu102_kmeans_int.bit.bin

2) Compile host code and link accelerator kernel library
	
	g++ -std=c++11 -pthread kmeans_top.cpp -ltbb -o run_kmeans lib_kmeans_accel.a

3) Benchmark instructions
	
	./run_kmeans
