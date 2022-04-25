#include "knl_kmeans.h"
#include "KMEANS.h"

class Body{
public:
    bool firsttime;
public:
    void OperatorGPU(int begin, int end, int id){
        // FPGA compute function declaration
        bodies_F+=end-begin;
        switch (id)
        {
        case 1:

            dist_out = (float *)sds_alloc((end-begin)*k*sizeof(float));
            if(!dist_out){
                cout << "Unable to allocate accelerator output memory" << endl;
                exit(1);
            }

            knl_kmeans(points, clusters, dist_out, k, dim, begin, end, file_desc_1, ioctl_flag); // Map stage - distance compute
            for(int i=begin; i<end; i++){ // Reduce stage - cluster assignment
                for(int j=0; j<k; j++){
                    if(dist_out[(i-begin)*k+j]<points[POINT_DIST(i, dim)]){
                        points[POINT_DIST(i, dim)] = dist_out[(i-begin)*k+j];
                        points[POINT_ID(i, dim)] = j;
                    }
                }
            }

            sds_free(dist_out);
            break;
        }
    }

    void OperatorCPU(int begin, int end){
        // CPU compute function declaration
        bodies_C+=end-begin;
        compute_cpu(begin, end);
    }
};
