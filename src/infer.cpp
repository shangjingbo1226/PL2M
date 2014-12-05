#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#include <cstdio>
#include <cstring>
#include <vector>
#include "apex_pcd_model.h"
#include "../apex-utils/apex_config.h"
#include "../apex-utils/apex_utils.h"
#include "../apex-tensor/apex_tensor.h"

apex_svd::PCDModel model;

inline void calcTest(char* testfile, char* output)
{
	FILE* out = fopen(output, "w");
	FILE* ftest = fopen(testfile, "r");
	float r;
	double sum = 0;
	int cnt=0;
	for (int u, i;fscanf(ftest, "%d%d%f", &u, &i, &r)==3;){
	    float pred = model.realScore(u,i);
	    fprintf(out, "%f\n", pred);
	    sum += (pred - r) * (pred - r);
	    ++cnt;
	}
    printf("%d pairs infered, RMSE = %.10f", cnt, sqrt(sum / cnt));fflush(stdout);
    fclose(ftest);
    fclose(out);
}

int main(int argc,char* argv[])
{
    if (argc!=4){
        puts("[usage] <test-file> <model-file> <predict-file>");
        puts("	if you don't have ground-truth, also leave the third column has numbers, e.g. all 0s.");
        return -1;
    }
	FILE* fmodel=fopen(argv[2],"rb");
	model.loadModel(fmodel);
	
	calcTest(argv[1], argv[3]);
	
	printf("\n");fflush(stdout);
	
    return 0;
}

