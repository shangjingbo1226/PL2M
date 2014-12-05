/*
 *  Copyright 2013-2014 APEX Data & Knowledge Management Lab, Shanghai Jiao Tong University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*!
 * \file apex_pcd_solver.h
 * \brief load train data, run train round
 * \author JingboShang, Tianqi Chen: {shangjingbo,tqchen}@apex.sjtu.edu.cn
 */
#ifndef _APEX_PCD_SOLVER_H_
#define _APEX_PCD_SOLVER_H_

#include "apex_pcd_model.h"
#include "apex_pcd_data.h"
#include "apex_pcd_train.h"
#include "../apex-utils/apex_config.h"
#include <omp.h>
#include <ctime>

namespace apex_svd{
	char line[1000000+5];
	
    class PCDSolverCore{    
    public:
		/*!\brief model contains latent vectors, bias, ...*/
        apex_svd::PCDModel model;
    protected:
		/*!\brief training parameters, containing regs, options, ...*/
        apex_svd::PCDTrainParam param;
        /*!\brief Sparse coef Matrix, user*userFeature, item*itemFeature */
        apex_svd::PCDMatrixS<Pair> u2uf,i2if;
        /*!\brief Sparse coef Matrix, userFeature*user, itemFeature*user */
        apex_svd::PCDMatrixS<Pair> uf2u,if2i;
        /*!\brief Sparse 0/1 Matrix, user*rateRecord, item*rateRecord, 1-entry only */
        apex_svd::PCDMatrixS<Single> u2inst,i2inst;
        /*!\brief rate of observed records, and their weight in trainning*/
        std::vector<float> rate, weight;
        /*!\brief userID, itemID of observed records*/
        std::vector<int> userID, itemID;
        /*!\brief user/item in each parallel part*/
        apex_svd::PCDMatrixS<Single> p2u,p2i; 
    protected:
		/*!\brief trainer used for trainning*/
        IPCDTrainer* trainer;
        /*!\brief record the current round*/
        int roundCounter;
        /*!\brief permutation of 1..dimension*/
        std::vector<int> dimIndex;
        /*!\brief number of observed records involves a feature uf/if*/
        std::vector<float> ufInsCnt,ifInsCnt;
        /*!\brief gradient of each observed records and its prediction*/
        std::vector<float> grad, pred;
    private:
		/*!\brief calculate the gradient of each observed records, and get the prediction if needed*/
        inline void runPredict( float *grad, float *pred, 
                                const int *userID, 
                                const int *itemID,
                                const float *rate,
                                const float *weight,
                                const PCDModel &model,
                                int numInst ){
			const int inst_chunck_size = std::max(numInst/param.getBlock(),1);
            if( pred != NULL ){
                #pragma omp parallel for schedule(dynamic,inst_chunck_size)
                for (int i=0;i<numInst;++i){
                    int uid=userID[i];
                    int iid=itemID[i];
                    pred[i]=model.predict(uid,iid);
                    grad[i]=model.param.calcGrad(pred[i],rate[i])*weight[i];
                }
            }else{
                #pragma omp parallel for schedule(dynamic,inst_chunck_size)
                for (int i=0;i<numInst;++i){
                    int uid=userID[i];
                    int iid=itemID[i];
                    float p =model.predict(uid,iid);
                    grad[i]=model.param.calcGrad(p,rate[i])*weight[i];
                }
            }                
        }
    protected:
		/*!\brief get the linear combination U=PX*/
        inline void calcWU( apex_tensor::CTensor2D &WUser,
                            const apex_tensor::CTensor2D &PUFeat,
                            const PCDMatrixS<Pair> &u2uf ){
            const int numUser = (int)u2uf.numRow();
            const int omp_chunck_size = std::max(numUser/param.getBlock(),1);
            #pragma omp parallel for schedule(dynamic,omp_chunck_size)
            for (int uid=0;uid<numUser;++uid){
				PCDMatrixS<Pair>::RLine fline=u2uf[uid];
                for( int k = 0; k < WUser.y_max; k ++ ){
                    float s=0;
                    for (unsigned j=0;j<fline.length;++j){
                        const unsigned fid=fline[j].cindex;
                        const float value=fline[j].fvalue;
                        s+=value*PUFeat[ k ][ fid ];
                    }
                    WUser[ k ][uid] = s;
                }
            }
        }
        /*!\brief update featurewise, update user/item bias first, and then, each dimension user/item*/
        inline void updateFeatureWise( void )
        {
            long timer=(long)time(0);
            {// update bias
                const int k = model.param.numDimension;
                if (!param.skipUserBias){
	                trainer->update( model.PUFeat[k], model.WUser[k], 
	                                 model.WItem[k], grad, pred, itemID, rate, weight,
	                                 u2inst, u2uf, uf2u, p2u, ufInsCnt, 
	                                 param.getRegBias(), true );
				}
                trainer->update( model.PIFeat[k], model.WItem[k], 
                                 model.WUser[k], grad, pred, userID, rate, weight,
                                 i2inst, i2if, if2i, p2i, ifInsCnt, 
                                 param.getRegBias(), true );
            }
            if (param.debugFlag!=0){
                printf("bias side time = %ld sec\t",(long)(time(0)-timer));fflush(stdout);
                timer=(long)time(0);
			}
            if( roundCounter > param.biasOnlyRound ){
                if( param.shuffleDim != 0 ) apex_random::shuffle( dimIndex );
                for( int i = 0; i < model.param.numDimension; i ++ ){
                    const int k = dimIndex[i];
                    trainer->update( model.PUFeat[k], model.WUser[k], 
                                     model.WItem[k], grad, pred, itemID, rate, weight,
                                 u2inst, u2uf, uf2u, p2u, ufInsCnt, 
                                     param.getRegUFeat(), false );
                    trainer->update( model.PIFeat[k], model.WItem[k], 
                                     model.WUser[k], grad, pred, userID, rate, weight,
                                     i2inst, i2if, if2i, p2i, ifInsCnt, 
                                     param.getRegIFeat(), false );
                }
            }
            if (param.debugFlag!=0){
                printf("dim iter time = %ld sec\t",(long)(time(0)-timer));fflush(stdout);
			}
        }
        /*!\brief update userwise, user bais, user dimension first, and then item bias, item dimension*/
        inline void updateUserWise( void )
        {
            long timer=(long)time(0);
            if (!param.skipUserBias){// update bias
                const int k = model.param.numDimension;                
                trainer->update( model.PUFeat[k], model.WUser[k], 
                                 model.WItem[k], grad, pred, itemID, rate, weight,
                                 u2inst, u2uf, uf2u, p2u, ufInsCnt, 
                                 param.getRegBias(), true );
            }
            if( roundCounter > param.biasOnlyRound ){
                if( param.shuffleDim != 0 ) apex_random::shuffle( dimIndex );
                for( int i = 0; i < model.param.numDimension; i ++ ){
                    const int k = dimIndex[i];
                    trainer->update( model.PUFeat[k], model.WUser[k], 
                                     model.WItem[k], grad, pred, itemID, rate, weight,
                                     u2inst, u2uf, uf2u, p2u, ufInsCnt, 
                                     param.getRegUFeat(), false );                
                }
            }
            if (param.debugFlag!=0){
                printf("user side time = %ld sec\t",(long)(time(0)-timer));fflush(stdout);
                timer=(long)time(0);
			}

            this->runPredict( &grad[0], 
                              pred.size() == 0 ? NULL: &pred[0],
                              &userID[0], &itemID[0], &rate[0], &weight[0],
                              model, (int)userID.size() );
            {// update bias           
                const int k = model.param.numDimension;            
                trainer->update( model.PIFeat[k], model.WItem[k], 
                                 model.WUser[k], grad, pred, userID, rate, weight,
                                 i2inst, i2if, if2i, p2i, ifInsCnt, 
                                 param.getRegBias(), true );
            } 
            if( roundCounter > param.biasOnlyRound ){
                if( param.shuffleDim != 0 ) apex_random::shuffle( dimIndex );
                for( int i = 0; i < model.param.numDimension; i ++ ){
                    const int k = dimIndex[i];
                    trainer->update( model.PIFeat[k], model.WItem[k], 
                                     model.WUser[k], grad, pred, userID, rate, weight,
                                     i2inst, i2if, if2i, p2i, ifInsCnt, 
                                     param.getRegIFeat(), false );
                }
            }
            if (param.debugFlag!=0){
                printf("item side time = %ld sec\t",(long)(time(0)-timer));fflush(stdout);
			}
        }
    protected:
		/*!\brief update one round according to the configuration*/
        inline void updateOneRound( void ){
            param.setRound( roundCounter ++ );
            // calculate prediction, get grad, pred ready
            long timer=(long)time(0);
            this->runPredict( &grad[0], 
                              pred.size() == 0 ? NULL: &pred[0],
                              &userID[0], &itemID[0], &rate[0], &weight[0],
                              model, (int)userID.size() );
            if( param.updateWise == 0 ) {
                this->updateFeatureWise();
            }else{
                this->updateUserWise();
            }
            if (param.debugFlag!=0){
                printf("train time = %ld sec\t",(long)(time(0)-timer));fflush(stdout);
			}
        }
    protected:
		/*!\brief calculate the InsCnt, defaultly 1.0*/
        inline void initInsCnt( void ){
            const int numUF = model.param.numFeatUser;
            const int numIF = model.param.numFeatItem;
            ufInsCnt.resize(numUF);
            ifInsCnt.resize(numIF);
        	if (param.useInsCnt==1){
				//puts("===mode 1===");
            	double sum=0;
            	for (int i=0;i<numUF;++i){
            	    PCDMatrixS<Pair>::RLine line=uf2u[i];
            	    ufInsCnt[i]=0;
            	    for (unsigned j=0;j<line.length;++j){
            	        unsigned uid=line[j].cindex;
            	        ufInsCnt[i]+=u2inst[uid].length;
            	    }
            	    sum+=ufInsCnt[i];
            	    if (ufInsCnt[i] == 0) {
                	    fprintf(stderr, "[Warning] no users have feature %d!\n", i);
                	    fflush(stderr);
            	    }
            	}
                fprintf(stderr, "avg number of users for each feature = %.10f\n", sum / numUF);fflush(stderr);
                sum=0;
            	for (int i=0;i<numIF;++i){
            	    PCDMatrixS<Pair>::RLine line=if2i[i];
            	    ifInsCnt[i]=0;
            	    for (unsigned j=0;j<line.length;++j){
            	        unsigned iid=line[j].cindex;
            	        ifInsCnt[i]+=i2inst[iid].length;
            	    }
            	    sum+=ifInsCnt[i];
            	    if (ifInsCnt[i] == 0) {
                	    fprintf(stderr, "[Warning] no items have feature %d!\n", i);
                	    fflush(stderr);
            	    }
            	}
            	fprintf(stderr, "avg number of items for each feature = %.10f\n",sum/numIF);fflush(stderr);
        	}else if (param.useInsCnt==2){
            	double sum=0;
            	for (int i=0;i<numUF;++i){
            	    PCDMatrixS<Pair>::RLine line=uf2u[i];
            	    ufInsCnt[i]=0;
            	    for (unsigned j=0;j<line.length;++j){
            	        unsigned uid=line[j].cindex;
            	        float fvalue=line[j].fvalue;
            	        ufInsCnt[i]+=u2inst[uid].length*sqr(fvalue);
            	    }
            	    sum+=ufInsCnt[i];
            	    if (ufInsCnt[i] == 0) {
                	    fprintf(stderr, "[Warning] no users have feature %d!\n", i);
                	    fflush(stderr);
            	    }
            	}
                fprintf(stderr, "avg number of users for each feature = %.10f\n", sum / numUF);fflush(stderr);
                sum=0;
            	for (int i=0;i<numIF;++i){
            	    PCDMatrixS<Pair>::RLine line=if2i[i];
            	    ifInsCnt[i]=0;
            	    for (unsigned j=0;j<line.length;++j){
            	        unsigned iid=line[j].cindex;
            	        ifInsCnt[i]+=i2inst[iid].length;
            	    }
            	    sum+=ifInsCnt[i];
            	    if (ifInsCnt[i] == 0) {
                	    fprintf(stderr, "[Warning] no items have feature %d!\n", i);
                	    fflush(stderr);
            	    }
            	}
            	fprintf(stderr, "avg number of items for each feature = %.10f\n",sum/numIF);fflush(stderr);
			}else{
        	    std::fill(ufInsCnt.begin(),ufInsCnt.end(),1.0f);
        	    std::fill(ifInsCnt.begin(),ifInsCnt.end(),1.0f);
        	}
        }
    public:
        PCDSolverCore( void ){
            roundCounter = 0;
            trainer = new PCDPPTrainer( model.param, param );
        }
        ~PCDSolverCore( void ){
            delete trainer;
        }
        inline void init( void ){
            if( param.nthread > 0 ){
                omp_set_num_threads( param.nthread  );
            }else {
                if( param.nthread < 0 ){
                    omp_set_num_threads( - param.nthread );                    
                    param.nthread = 0;
                }
            }
            this->initInsCnt();
            trainer->init();
            grad.resize( rate.size() );
            if( param.useHess == 0 ){
                pred.resize( rate.size() );
            }
            for( int k = 0; k < model.param.numDimension; k ++ ){
                dimIndex.push_back( k );
            }
        }
        inline void setParam(const char* name,const char* val)
        {
            model.setParam(name,val);
            param.setParam(name,val);            
            std::stringstream in(val);
            if (!strcmp(name,"continue")){
            	in >> roundCounter;
            	return;
            }
        }
        inline int getMaxRound( void ) const
        {
            return param.maxRound;
        }

    };        
};


/// outside code
namespace apex_svd{
    class PCDSolver: public PCDSolverCore{
    private:
		/*!\brief load a matrix from a ASCII file, row and col should be indicated explicitly*/
        inline void loadMatrix(apex_svd::PCDMatrixS<Pair>& matrix, const char* filename,int rowLimit,int colLimit)
        {
			matrix.init_budget( rowLimit );
			
            FILE* in=apex_utils::fopen_check(filename,"r");
            for (int uid,cnt;fscanf(in,"%d%d",&uid,&cnt)==2;){
                for (int i=0;i<cnt;++i){
                    int iid;
                    float coef;
                    if (fscanf(in,"%d:%f",&iid,&coef)!=2){
                        fprintf(stderr, "[Error in load matrix] while reading %s\n",filename);
                        fprintf(stderr, "uid= %d, i= %d\n",uid,i);
                        exit(-2);
                    }
				    if (iid>=colLimit || iid<0){
				        fprintf(stderr, "%d %d %d\n",uid,iid,colLimit);fflush(stderr);
				    }
                    apex_utils::assert_true(iid>=0 && iid<colLimit,"error while loadMatrix");
                    matrix.add_budget(uid);
                }
            }
            fclose(in);
            matrix.init_storage();
            
            in=apex_utils::fopen_check(filename,"r");
            for (int uid,cnt;fscanf(in,"%d%d",&uid,&cnt)==2;){
                for (int i=0;i<cnt;++i){
                    int iid;
                    float coef;
                    if (fscanf(in,"%d:%f",&iid,&coef)!=2){
                        fprintf(stderr, "[Error in load matrix] while reading %s\n",filename);
                        fprintf(stderr, "uid= %d, i= %d\n",uid,i);
                        exit(-2);
                    }
				    if (iid>=colLimit || iid<0){
				        fprintf(stderr, "%d %d %d\n",uid,iid,colLimit);fflush(stderr);
				    }
                    apex_utils::assert_true(iid>=0 && iid<colLimit,"error while loadMatrix");
                    matrix.push_elem( uid, Pair(iid, coef) );
                }
            }
	    }
    	/*!\brief transpose a matrix, row and col should be indicated explicitly*/
        inline void transpose(apex_svd::PCDMatrixS<Pair> & matrix,apex_svd::PCDMatrixS<Pair> &result,int row,int col)
        {
			result.init_budget( col );
            int numRow=(int)matrix.numRow();
            for (int i=0;i<numRow;++i){
                apex_svd::PCDMatrixS<Pair>::RLine line=matrix[i];
                for (size_t j=0;j<line.length;++j){
                    int cindex=line[j].cindex;
					if (cindex >= col || cindex < 0) {
						fprintf(stderr, "[Error] column index exceeds! %d\n", cindex);
						fflush(stderr);
					}
                    result.add_budget(cindex);
                }
            }
            result.init_storage();
            for (int i=0;i<numRow;++i){
                apex_svd::PCDMatrixS<Pair>::RLine line=matrix[i];
                for (size_t j=0;j<line.length;++j){
                    int cindex=line[j].cindex;
                    float fvalue=line[j].fvalue;
                    result.push_elem( cindex, Pair((size_t)i, fvalue) );
                }
            }
        }
        /*!\brief load a matrix from a binary file*/
        template<typename Entry>
        inline bool tryLoadBuffer( PCDMatrixS<Entry> &mat, const char *fname ){
            char name[ 256 ];
            sprintf( name, "%s.buf", fname );
            FILE *fi = fopen64( name, "rb" );
            if( fi == NULL ) return false;
            mat.loadBinary( fi );
            fclose( fi );
            return true;
        }
        /*!\brief save a matrix to a binary file*/
        template<typename Entry>
        inline void saveBuffer( PCDMatrixS<Entry> &mat, const char *fname ){
            char name[ 256 ];
            sprintf( name, "%s.buf", fname );
            FILE *fo = apex_utils::fopen_check( name, "wb" );
            mat.writeBinary( fo );
            fclose( fo );
        }
        /*!\brief load observed record from a binary buffer, return false if buffer not exist*/
        inline bool tryLoadBuffer( std::vector<int> &userID,
								   std::vector<int> &itemID, 
								   std::vector<float> &rate,
								   std::vector<float> &weight,
								   const char *fname ){
			char name[ 256 ];
            sprintf( name, "%s.buf", fname );
            FILE *fi = fopen64( name, "rb" );
            if( fi == NULL ) return false;   
            size_t n;
            apex_utils::assert_true( fread( &n,sizeof(size_t), 1, fi ) > 0, "load" );
            userID.resize(n); itemID.resize(n); rate.resize(n); weight.resize(n);
            if( n > 0 ){
                apex_utils::assert_true( fread( &userID[0],sizeof(int), n, fi ) > 0, "load" );
                apex_utils::assert_true( fread( &itemID[0],sizeof(int), n, fi ) > 0, "load" );
                apex_utils::assert_true( fread( &rate[0],sizeof(float), n, fi ) > 0, "load" );
                apex_utils::assert_true( fread( &weight[0],sizeof(float), n, fi ) > 0, "load" );
            }
            fclose( fi );
            return true;
        }
        /*!\brief save observed record to a binary buffer*/
        inline void saveBuffer( const std::vector<int> &userID, 
                                const std::vector<int> &itemID, 
                                const std::vector<float> &rate,
                                const std::vector<float> &weight,
								const char *fname ){
			char name[ 256 ];
            sprintf( name, "%s.buf", fname );
            FILE *fo = apex_utils::fopen_check( name, "wb" );
            size_t n = itemID.size();
            fwrite( &n,sizeof(size_t), 1, fo );
            if( n > 0 ){
                fwrite( &userID[0],sizeof(int), n, fo );
                fwrite( &itemID[0],sizeof(int), n, fo );
                fwrite( &rate[0],sizeof(float), n, fo );
                fwrite( &weight[0],sizeof(float), n, fo );
            }
            fclose( fo );
        }
        /*!\brief calculate user/item in each parallel part*/
        inline void buildPart(PCDMatrixS<Pair> &u2uf, PCDMatrixS<Pair> &uf2u, PCDMatrixS<Single> &result)
        {
        	int numUser=(int)u2uf.numRow();
        	int numFeature=(int)uf2u.numRow();
        	
        	int step=param.sizePCDBlock;
        	int numPart=(numFeature+step-1)/step;
        	
        	result.init_budget( numPart );
        	for (int i=0;i<numUser;++i){
        		PCDMatrixS<Pair>::RLine line=u2uf[i];
        		int last=-1;
        		for (unsigned j=0;j<line.length;++j){
        			int fid=line[j].cindex;
        			int partid=fid/step;
        			if (last!=partid){
        				result.add_budget(partid);
        				last=partid;
					}
        		}
        	}
        	result.init_storage();
        	for (int i=0;i<numUser;++i){
        		PCDMatrixS<Pair>::RLine line=u2uf[i];
        		int last=-1;
        		for (unsigned j=0;j<line.length;++j){
        			int fid=line[j].cindex;
        			int partid=fid/step;
        			if (last!=partid){
        				result.push_elem(partid,Single(i));
        				last=partid;
					}
        		}
        	}
        }
        /*!\brief build index of user/item to observed records*/
        inline void buildIndex( PCDMatrixS<Single> &u2inst, const std::vector<int> &user, int numUser){
			u2inst.init_budget( numUser );
            for (size_t i=0;i<user.size();++i){
				u2inst.add_budget(user[i]);
			}
			u2inst.init_storage();
            for (size_t i=0;i<user.size();++i){
				u2inst.push_elem( user[i], (unsigned)i );
			}
		}
		/*!\brief initialization, laod feature matrices and train*/
        inline void initData( const char *userFeatureMatrix, const char *itemFeatureMatrix, const char *trainName ){
            long start = (long)time(0);
            if( !tryLoadBuffer( u2uf, userFeatureMatrix ) ){
                loadMatrix(u2uf,userFeatureMatrix,model.param.numUser,model.param.numFeatUser);
                saveBuffer( u2uf, userFeatureMatrix );
            }
            fprintf(stderr, "userFeature loaded!\n");fflush(stderr);
            if( !tryLoadBuffer( i2if, itemFeatureMatrix ) ){
                loadMatrix(i2if,itemFeatureMatrix,model.param.numItem,model.param.numFeatItem);
                saveBuffer( i2if, itemFeatureMatrix );
            }
            
            u2uf.sort();
            i2if.sort();
            
            fprintf(stderr, "itemFeature loaded!\n");fflush(stderr);
            transpose(u2uf,uf2u,model.param.numUser,model.param.numFeatUser);

            transpose(i2if,if2i,model.param.numItem,model.param.numFeatItem);           
            
            buildPart(u2uf,uf2u,p2u);
            buildPart(i2if,if2i,p2i);
            
            fprintf(stderr, "avg number of features for each user %lf\n",(double)u2uf.getSize()/u2uf.numRow());fflush(stderr);
            fprintf(stderr, "avg number of features for each item %lf\n",(double)i2if.getSize()/i2if.numRow());fflush(stderr);
                       
            if ( !tryLoadBuffer( userID, itemID, rate, weight, trainName ) ){
                FILE* in=fopen(trainName,"r");
                
                userID.clear();
                itemID.clear();
                rate.clear();
                weight.clear();
                float r, w;
                for (int uid,iid, inst=0;fgets(line,1000000,in);++inst){
					std::stringstream in(line);
					if (!(in >> uid >> iid >> r)){
						break;
					}
					if (!(in >> w)){
						w=1;
					}
					if (uid>=model.param.numUser){
						fprintf(stderr, "[Error] uid %d %d\n",uid,model.param.numUser);fflush(stderr);
					}else if (iid>=model.param.numItem){
						fprintf(stderr, "[Error] iid %d %d\n",iid,model.param.numItem);fflush(stderr);
					}
					apex_utils::assert_true(uid<model.param.numUser && iid<model.param.numItem,"conflict conf and train");
                    r /= model.param.scale;
                    userID.push_back(uid);
                    itemID.push_back(iid);
                    rate.push_back(r);
                    weight.push_back(w);
                }
                fprintf(stderr, "train loaded\n");
                
                saveBuffer( userID, itemID, rate, weight, trainName );
            }
            
            buildIndex(u2inst,userID,model.param.numUser);
            buildIndex(i2inst,itemID,model.param.numItem);
            
            // check balance
            u2uf.checkBalance( param.nthread, "u2uf" );
            uf2u.checkBalance( param.nthread, "uf2u" );
            i2if.checkBalance( param.nthread, "u2uf" );
            if2i.checkBalance( param.nthread, "uf2u" );
            //u2inst.checkBalance( model.param.nthread, "u2inst" );
            //i2inst.checkBalance( model.param.nthread, "i2inst" );
        }        
    public:
		/*!\brief initialization, laod config and then initData*/
        inline void initialize(char* configName,char* userFeatureMatrix,char* itemFeatureMatrix,char* trainName,char* modelpath)
        {
        	roundCounter = 0; //default
            apex_utils::ConfigIterator* conf = new apex_utils::ConfigIterator(configName);
            while (conf->next()){
                setParam(conf->name(),conf->val());
            }
            
            this->initData( userFeatureMatrix, itemFeatureMatrix, trainName );    
            
            PCDSolverCore::init();

            if (roundCounter>0){
            	char modelfile[1000];
            	sprintf(modelfile,"%s.%d",modelpath,roundCounter);
            	model.setParam("model",modelfile);
            }else{
	            model.initModel();
                PCDSolverCore::calcWU( model.WUser, model.PUFeat, u2uf );
                PCDSolverCore::calcWU( model.WItem, model.PIFeat, i2if );
            }
        }
        
        inline int startRound( void )
        {
        	return roundCounter;
        }
        
        inline void update_one_round( void )
        {
            PCDSolverCore::updateOneRound();
        }
        /*!\brief save the model to a binary file*/
        inline void finish(char* filename) const
        {
            FILE* out=apex_utils::fopen_check( filename, "wb" );
            model.saveModel(out);
            fclose( out );
        }       
    };
};
#endif

