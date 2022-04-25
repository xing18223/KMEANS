// Author: Xingtang Sun - ni18223@bristol.ac.uk
// Date: 25th April 2022

#include "sds_lib.h"
#include "knl_kmeans.h"
#include <sys/ioctl.h>

// IRQ line address
#define IOCTL_WAIT_INTERRUPT_1 _IOR(100, 0, char *)

#define POINT_COORD(n_ind, coord, dim) ((n_ind)*((dim)+2) + 2+(coord))
#define CLUSTER_COORD(k_ind, coord, dim) (((2*(dim))+2)*(k_ind) +2+(coord))
#define SQUARE(diff) ((diff)*(diff))

// Accelerator dimensions
const int MAX_DIM = 150;
const int MAX_PTS = 2000;
const int MAX_CLS = 100;

//---------------------------------------------//
#pragma SDS data sys_port(points_ptr: ps_e_S_AXI_HPC0_FPD)
#pragma SDS data sys_port(clusters: ps_e_S_AXI_HPC0_FPD)
#pragma SDS data sys_port(dist_out: ps_e_S_AXI_HPC0_FPD)

#pragma SDS data zero_copy(points_ptr[0:(end-begin)*(dim+2)])
#pragma SDS data zero_copy(clusters[0:k*(2+(2*dim))])
#pragma SDS data zero_copy(dist_out[0:k*(end-begin)])

#pragma SDS data mem_attribute(points_ptr: PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(clusters: PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(dist_out: PHYSICAL_CONTIGUOUS)

void kmeans_accel(float *points_ptr, float *clusters, float *dist_out, int k, int dim, int begin, int end){

	// Local buffers
    float local_points[MAX_PTS*(MAX_DIM+2)];
	#pragma HLS ARRAY_PARTITION variable=local_points cyclic factor=2+MAX_DIM

    float local_clusters[MAX_CLS*(2+(2*MAX_DIM))];
	#pragma HLS ARRAY_PARTITION variable=local_clusters cyclic factor=MAX_CLS

    float local_dist_out[MAX_CLS*MAX_PTS];
	#pragma HLS ARRAY_PARTITION variable=local_dist_out cyclic factor=MAX_CLS

	// Function parameters stored locally
    int local_dim = dim; 
	int local_k = k;
	int local_begin = begin;
	int local_end = end;

    for(int i=0; i<(local_end-local_begin)*(local_dim+2); i++){ // Burst read from global to local
		#pragma HLS loop_tripcount min=MAX_PTS*(MAX_DIM+2) max=MAX_PTS*(MAX_DIM+2)
		#pragma HLS PIPELINE
    	local_points[i] = points_ptr[i];
    }
    for(int j=0; j<local_k*(2+(local_dim*2)); j++){
		#pragma HLS loop_tripcount min=MAX_CLS*(2+(2*MAX_DIM)) max=MAX_CLS*(2+(2*MAX_DIM))
		#pragma HLS PIPELINE
        local_clusters[j] = clusters[j];
    }

	// chunk x k distances computed
    for(int j=0; j<local_k; j++){
		#pragma HLS unroll factor=MAX_CLS/2
		#pragma HLS loop_tripcount min=MAX_CLS max=MAX_CLS
        for(int i=0; i<(local_end-local_begin); i++){
			#pragma HLS loop_tripcount min=MAX_PTS max=MAX_PTS
        	float dist = 0;
            for(int d=0; d<local_dim; d++){
				#pragma HLS loop_tripcount min=MAX_DIM max=MAX_DIM
				#pragma HLS PIPELINE
                float diff = local_clusters[CLUSTER_COORD(j, d, local_dim)] - local_points[POINT_COORD(i, d, local_dim)];
                dist += SQUARE(diff);
            }
            local_dist_out[i*local_k+j] = dist;
        }
    }

    for(int i=0; i<local_k*(local_end-local_begin); i++){ // Burst writing from local to global
		#pragma HLS loop_tripcount min=MAX_CLS*MAX_PTS max=MAX_CLS*MAX_PTS
		#pragma HLS PIPELINE
    	dist_out[i] = local_dist_out[i];
	}
}

void knl_kmeans(float *points, float *clusters, float *dist_out, int k, int dim, int begin, int end, int file_desc, int ioctl_flag){

	float *points_ptr = points + begin*(dim+2); // Apply chunk size offset to read from global points array

	#pragma SDS resource(1) // Resource allocation
	#pragma SDS async(1) // Accelerator thread issue

    kmeans_accel(points_ptr, clusters, dist_out, k, dim, begin, end); // Accelerator function call

    if (ioctl_flag) { // Interrupt mechanism enabled
		int ret_value;
		ret_value = ioctl(file_desc, IOCTL_WAIT_INTERRUPT_1); //Sleep until interrupt, pass block size to driver for debugging
	}

	#pragma SDS wait(1) // Terminate accelerator thread

}
