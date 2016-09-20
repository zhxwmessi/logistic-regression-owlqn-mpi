#include <string>
#include "mpi.h"
#include "load_data.h"
#include "owlqn.h"
#include "predict.h"
#include <glog/logging.h>
//#include "gtest/gtest.h"

int main(int argc,char* argv[]){  
    int rank, nproc;
    int namelen = 1024;
    char processor_name[namelen];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Get_processor_name(processor_name,&namelen);
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "./log";
    LOG(INFO) << "my process rank: "<< rank <<", totoal process num: "<< nproc <<std::endl;
    std::cout<<"my host = "<<processor_name<<" my rank = "<<rank<<std::endl;

    int stepnum = atoi(argv[2]);
    int batchsize = atoi(argv[3]);
    char train_data_path[1024];
    snprintf(train_data_path, 1024, "%s-%05d", argv[4], rank);
    char test_data_path[1024];
    snprintf(test_data_path, 1024, "%s-%05d", argv[5], rank);

    Load_Data test_data(test_data_path);
    test_data.load_data_batch(nproc, rank);
    Predict predict(&test_data, nproc, rank);

    Load_Data train_data(train_data_path); 
    train_data.load_data_batch(nproc, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (strcmp(argv[1], "owlqn") == 0){
        OWLQN owlqn(&train_data, &predict, nproc, rank);
        owlqn.steps = stepnum;
        owlqn.batch_size = batchsize;
        owlqn.owlqn();
        std::cout<<"rank "<<rank<<" finish train! "<<processor_name<<std::endl;
    }
    MPI::Finalize();
    return 0;
}
