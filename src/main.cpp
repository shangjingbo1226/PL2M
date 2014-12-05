#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#define PCD_OMP_CHUNK_SIZE 100
#define PCD_INST_CHUNK_SIZE 1000
#define PCD_UPDATE_USER_CHUNK_SIZE 500
#define PCD_UPDATE_FEAT_CHUNK_SIZE 50

#include "apex_pcd_model.h"
#include "apex_pcd_data.h"
#include "apex_pcd_train.h"
#include "apex_pcd_solver.h"
#include <ctime>

using namespace apex_svd;

int main( int argc, char* argv[]){
	if ( argc!=7 ){
		puts("[usage] <config-file> <userFeatureMatrix-file> <itemFeatureMatrix-file> <train-file> <test-file> <model-out>");
		return -1;
	}
    apex_svd::PCDSolver* solver=new apex_svd::PCDSolver();
	solver->initialize(argv[1],argv[2],argv[3],argv[4],argv[6]);
	
	double start=(double)time(NULL);
	for (int round=solver->startRound();round<solver->getMaxRound();++round){        
        printf("round %.2d\t",round);fflush(stdout);
	    solver->update_one_round();
        char file[1000];
        sprintf(file,"%s.%d",argv[6],round+1);
        solver->finish(file);
        /*char command[1000];
        sprintf(command,"svd_pcdpp_infer %s %s %s %s",argv[1],argv[4],argv[5],file);
        int rt = system(command);
        apex_utils::assert_true( rt == rt );*/
	}
	double end=(double)time(NULL);
	printf("train time all = %.0f sec\n",(end-start));
	
	delete solver;
	
	return 0;
}
