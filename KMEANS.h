#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
using namespace std;
using namespace tbb;

float *clusters; // Global array of k clusters and its indexing 
#define CLUSTER_ID(k_ind, dim) (((2*(dim))+2)*(k_ind))
#define CLUSTER_SIZE(k_ind, dim) (((2*(dim))+2)*(k_ind) +1)
#define CLUSTER_COORD(k_ind, coord, dim) (((2*(dim))+2)*(k_ind) +2+(coord))
#define CLUSTER_SUM_COORD(k_ind, coord, dim) (((2*(dim))+2)*(k_ind) +2+(dim)+(coord))

float *points; // Global array of n clusters and its indexing
#define POINT_ID(n_ind, dim) ((n_ind)*((dim)+2)) 
#define POINT_DIST(n_ind, dim) ((n_ind)*((dim)+2)+1)
#define POINT_COORD(n_ind, coord, dim) ((n_ind)*((dim)+2) + 2+(coord))

// Benchmark parameters
int n;
int k;
int epochs;
int max_coord;
int dim;
int ioctl_flag;
float *dist_out; // Distance reduce buffer
long unsigned int bodies_C=0, bodies_F=0; 

#define SQUARE(diff) ((diff)*(diff))

#define DRIVER_FILE_NAME_1 "/dev/intgendriver1" // Kernel driver location
int file_desc_1;

void input(char *in_file){
    cout << "Reading input from csv" << endl;
    string line;
    ifstream file(in_file);
    int ind = 0;
    while(getline(file, line)){
        stringstream lineStream(line);
        string bit;
        points[POINT_ID(ind, dim)] = -1;
        points[POINT_DIST(ind, dim)] = __FLT_MAX__;
        for(int d=0; d<dim; d++){
            getline(lineStream, bit, ',');
            points[POINT_COORD(ind, d, dim)] = stof(bit);
        }
        ind++;
    }
}

void random_input(){
    cout << "Random input generation" << endl;
    srand(time(NULL));
    for(int i=0; i<n; i++){
        points[POINT_ID(i, dim)] = -1;
        points[POINT_DIST(i, dim)] = __FLT_MAX__;
        for(int d=0; d<dim; d++){
            points[POINT_COORD(i, d, dim)] = rand()%max_coord;
        }
    }
}

void output(char *out_file){
    cout << "Writing to output csv" << endl;
    ofstream myfile;
    myfile.open(out_file);
    for(int i=0; i<n; i++){
        myfile << points[POINT_ID(i, dim)] << endl;
    }
    myfile.close();
}

void init_clusters(){
    cout << "Random cluster initialisation" << endl;
    srand(time(NULL));
    for(int j=0; j<k; j++){
        clusters[CLUSTER_ID(j, dim)] = j;
        clusters[CLUSTER_SIZE(j, dim)] = 0;
        for(int d=0; d<dim; d++){
            clusters[CLUSTER_COORD(j, d, dim)] = rand()%max_coord;
            clusters[CLUSTER_SUM_COORD(j, d, dim)] = 0;
        }
    }
}

void update_clusters(){
    for(int i=0; i<n; i++){ //Update cluster accummulator and size
        int id = points[POINT_ID(i, dim)];
        for(int d=0; d<dim; d++){
            clusters[CLUSTER_SUM_COORD(id, d, dim)] += points[POINT_COORD(i, d, dim)];
        } 
        clusters[CLUSTER_SIZE(id, dim)] ++;
    }

    for(int j=0; j<k; j++){
        if(clusters[CLUSTER_SIZE(j, dim)] !=0){ // Update cluster coord
            for(int d=0; d<dim; d++){
                clusters[CLUSTER_COORD(j, d, dim)] = clusters[CLUSTER_SUM_COORD(j, d, dim)]/clusters[CLUSTER_SIZE(j, dim)];
            }

            clusters[CLUSTER_SIZE(j, dim)] = 0; // Reset cluster size
            for(int d=0; d<dim; d++){ // Reset accumulators
                clusters[CLUSTER_SUM_COORD(j, d, dim)] = 0;
            }
        }
    }
} 

void reset_points(){ 
    for(int i=0; i<n; i++){
        points[POINT_ID(i, dim)] = -1;
        points[POINT_DIST(i, dim)] = __FLT_MAX__;
    }
}

float sq_eucl_dist(int cur_cluster_id, int cur_point){
    float distance = 0; // Squared euclidean distance between point and cluster
    for(int d=0; d<dim; d++){
        float diff = clusters[CLUSTER_COORD(cur_cluster_id, d, dim)] - points[POINT_COORD(cur_point, d, dim)];
        distance += SQUARE(diff);
    }
    return distance;
}

void compute_cpu(int begin, int end){ 
    for(int j=0; j<k; j++){
        for(int i=begin; i<end; i++){ 
            float distance = sq_eucl_dist(j, i);
            if(distance < points[POINT_DIST(i, dim)]){
                points[POINT_DIST(i, dim)] = distance;
                points[POINT_ID(i, dim)] = j;
            } 
        }
    }
}
