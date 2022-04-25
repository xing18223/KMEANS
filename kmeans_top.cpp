// Author: Xingtang Sun - ni18223@bristol.ac.uk
// Date: 25th April 2022

#include "../KMEANS/schedulers/MultiDynamic.h"
#include "sds_lib.h"
#include "Body.h"

using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

int main(int argc, char* argv[]){
    if(argc!=12){
        cout << "Please provide the following parameters in the following order" << endl;
        cout << "+--------------------------------------------------------------------------+" << endl;
        cout << "| <n> - number of data points                                              |"<< endl;
        cout << "| <k> - number of clusters                                                 |" << endl;
        cout << "| <max_coord> - maximum value for the coordinates of the feature space     |" << endl;
        cout << "| <dim> - dimensionality of the data points                                |" << endl;
        cout << "| <epochs> - number of iterations                                          |" << endl;
        cout << "| <CC> - number of CPU cores: 0 to 4                                       |" << endl;
        cout << "| <ACC> - number of FPGA accelerators: 0 or 1                              |" << endl;
        cout << "| <chunk> - chunk size for the FPGA accelerators                           |" << endl;
        cout << "| <ioctl> - 1/0 to activate/deactivate the scheduler interrupt mechanism   |" << endl;
        cout << "| <in_file> - name of input data csv file (set to 1 for random generation) |" << endl;
        cout << "| <out_file> - name of output csv file to write the results to             |" << endl;
        cout << "+--------------------------------------------------------------------------+" << endl;
        exit(1);
    }
    // Benchmark parameters
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    max_coord = atoi(argv[3]);
    dim = atoi(argv[4]);
    epochs = atoi(argv[5]);

    // ENEAC scheduler initilisation
    Body body; 
    Params p;
    p.numcpus = atoi(argv[6]);
    p.numgpus = atoi(argv[7]);
    p.gpuChunk = atoi(argv[8]);
    Dynamic *hs = Dynamic::getInstance(&p);

    // Global arrays for points and clusters
    points = (float *)sds_alloc((dim+2)*n*sizeof(float));
    clusters = (float *)sds_alloc(((2*dim)+2)*k*sizeof(float));
    if(!points||!clusters){
        cout << "Unable to allocate memory" << endl;
        exit(1);
    }

    // Open kernel driver for the interrupt mechanism
    ioctl_flag = atoi(argv[9]);
    if (ioctl_flag > 0){
		file_desc_1 = open(DRIVER_FILE_NAME_1, O_RDWR);
		if (file_desc_1 < 0) {
			fprintf(stderr,"Can't open driver file: %s\n", DRIVER_FILE_NAME_1);
			exit(-1);
		} else {
			fprintf(stderr,"Driver successfully opened: %s\n", DRIVER_FILE_NAME_1);
		}
	}

    auto t1 = high_resolution_clock::now();
    /* INPUT */
    if(atoi(argv[10])){
        random_input();
    }
    else{
        input(argv[10]);
    }

    auto t2 = high_resolution_clock::now();
    /* CLUSTER INITIALISATION */
    init_clusters();

    auto t3 = high_resolution_clock::now();
    /* COMPUTE */
    int iter = 0;
    while(iter < epochs){
        iter++;
        hs->heterogeneous_parallel_for(0, n, &body); // Distance compute
        update_clusters(); 
        if(iter < epochs-1){
            reset_points();
        }
    }

    auto t4 = high_resolution_clock::now();
    /* OUTPUT */
    output(argv[11]);

    auto t5 = high_resolution_clock::now();
    //Close interrupt drivers
    if (ioctl_flag > 0) {
        fprintf(stderr,"Closing driver: %s\n", DRIVER_FILE_NAME_1); 
        close(file_desc_1);
    }  

    duration<double, milli> time_a = t2 - t1;
    duration<double, milli> time_b = t3 - t2;
    duration<double, milli> time_c = t4 - t3;
    duration<double, milli> time_d = t5 - t4;
    cout << "+--------------------------------- Benchmark conditions ---------------------------------+" << endl;
    cout << " Number of points (n): " << n << endl;
    cout << " Number of clusters (k): " << k << endl;
    cout << " Feature space size (max_coord): " << max_coord << endl;
    cout << " Iterations (epochs): " << epochs << endl;
    cout << " Dimensionality (dim): " << dim << endl;
    cout << "+------------------------------------- Compute time -------------------------------------+" << endl;
    cout << " Input: " << time_a.count() << "ms" << endl;
    cout << " Cluster initialisation: " << time_b.count() << "ms" << endl;
    cout << " Compute (distance compute + cluster update + reset points): " << time_c.count() << "ms" << endl;
    cout << " Output: " << time_d.count() << "ms" << endl;
    cout << "+----------------------------------Workload distribution---------------------------------+" << endl;
    cout << " Total workload: " << bodies_F+bodies_C << endl;
    cout << " Total workload on CPU: " << bodies_C << endl;
    cout << " Total workload on FPGA: " << bodies_F << endl;
    cout << " Percentage of work done by the FPGA" << bodies_F*100/(bodies_F+bodies_C) << endl;
    cout << "+----------------------------------------------------------------------------------------+" << endl;
    
    sds_free(points);
    sds_free(clusters);

    return 0;
}
