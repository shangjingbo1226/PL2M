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
 * \file apex_pcd_train.h
 * \brief interface classes for PCD
 * \author JingboShang, Tianqi Chen: {shangjingbo,tqchen}@apex.sjtu.edu.cn
 */
#ifndef _APEX_PCD_TRAIN_H_
#define _APEX_PCD_TRAIN_H_

#include "apex_pcd_update.h"
#include <algorithm>
#include <vector>

namespace apex_svd{
    class IPCDTrainer{
    public:
        /*! 
         * \brief initialize the auxiliary space, called 
         */
        virtual void init( void ){}
        /*! 
         * \brief update PUFLine, keep grad, WULine up to date
         */
        virtual void update( apex_tensor::CTensor1D PUFLine,
                             apex_tensor::CTensor1D WULine,
                             const apex_tensor::CTensor1D WVLine,
                             std::vector<float> &grad,
                             std::vector<float> &pred,
                             const std::vector<int>   &itemID,
                             const std::vector<float> &rate,
                             const std::vector<float> &weight,
                             const PCDMatrixS<Single> &u2inst,
                             const PCDMatrixS<Pair> &u2uf,
                             const PCDMatrixS<Pair> &uf2u,
                             const PCDMatrixS<Single> &p2u,
                             const std::vector<float> &ufInsCnt,
                             const PCDTrainParam::Reg reg,
                             bool isBias 
                             ) = 0;
    };
};

namespace apex_svd{
    class PCDPPTrainer: public IPCDTrainer{
    private:
        const PCDModel::Param &loss;
        const PCDTrainParam   &param;        
        // local tmp space
        std::vector<float> guser, huser, backU;
        // updater to do update
        IPCDUpdater *updater;
    public:
        virtual void init( void ){
            if( guser.size() != 0 ) return;
            // malloc space, these space will not be changed, we know they are sufficient
            guser.resize( std::max( loss.numUser, loss.numItem ) );
            huser.resize( std::max( loss.numUser, loss.numItem ) );
            backU.resize( std::max( loss.numUser, loss.numItem ) );
            updater->init();
        }
        PCDPPTrainer( const PCDModel::Param &loss, const PCDTrainParam &param ):loss(loss),param(param){
            updater = new BlockPCDUpdater( loss, param );
        }
        virtual ~PCDPPTrainer( void ){
            delete updater;
        }
    private:
        // substeps of, get GH and back up deltaU
        virtual void getGH_BackU(  float *guser,
                                   float *huser,
                                   float *backU, 
                                   const apex_tensor::CTensor1D &WULine,
                                   const apex_tensor::CTensor1D &WVLine,
                                   const float *grad,
                                   const int *itemID,
                                   const float *weight,
                                   const PCDMatrixS<Single> &u2inst,
                                   bool isBias ){
            int rows = (int)u2inst.numRow();
            const int omp_chunk_size = std::max(rows/getBlock(param.nthread),1);
            if ( !isBias ){
                #pragma omp parallel for schedule(dynamic,omp_chunk_size) 
                for (int uid=0;uid<rows;++uid){
                    PCDMatrixS<Single>::RLine line=u2inst[uid];
                    pcd_sum_float sumg = 0.0f, sumh = 0.0f;
                    for (unsigned i=0;i<line.length;++i){
                        const unsigned instID = line[i].cindex;
                        const unsigned iid = itemID[ instID ];
                        const float v = WVLine[iid];
                        sumg += v * grad[instID];
                        sumh += sqr( v ) * weight[instID];
                    }
                    sumh *= loss.calcHess();
                    backU[ uid ] = WULine[ uid ];
                    guser[ uid ] = (float)sumg;
                    huser[ uid ] = (float)sumh;
                }
            }else{
                #pragma omp parallel for schedule(dynamic,omp_chunk_size) 
                for (int uid=0;uid<rows;++uid){
                    PCDMatrixS<Single>::RLine line=u2inst[uid];
                    pcd_sum_float sumg = 0.0f;
                    pcd_sum_float sumh = 0.0f;
                    for (unsigned i=0;i<line.length;++i){
                        const unsigned instID = line[i].cindex;
                        sumg += grad[instID];
                        sumh += weight[instID];
                    }
                    backU[ uid ] = WULine[ uid ];
                    guser[ uid ] = (float)sumg;
                    huser[ uid ] = (float)(loss.calcHess() * sumh);
//printf("%lf %lf %lf\n",sumg,sumh,loss.calcHess());fflush(stdout);
                }
            }
        }
        // substeps of, get updated WULine, and grad
        virtual void modify_U_Grad( apex_tensor::CTensor1D WULine,
                                    float *grad,
                                    float *pred,
                                    const apex_tensor::CTensor1D &PUFLine,
                                    const apex_tensor::CTensor1D &WVLine,
                                    const int *itemID,
                                    const float *rate,
                                    const float *weight,
                                    const float *backU,
                                    const PCDMatrixS<Single> &u2inst,
                                    const PCDMatrixS<Pair> &u2uf, bool isBias ){
            const int numUser = (int)u2inst.numRow();
            const int omp_chunk_size = std::max(numUser/getBlock(param.nthread),1);
            #pragma omp parallel for schedule(dynamic,omp_chunk_size)
            for (int uid=0;uid<numUser;++uid){
                // recalculate WULine 
				PCDMatrixS<Pair>::RLine fline=u2uf[uid];
				float s=0;
				for (unsigned j=0;j<fline.length;++j){
                    const unsigned fid=fline[j].cindex;
					const float value=fline[j].fvalue;
					s+=value*PUFLine[fid];
				}
				WULine[uid] = s;
                // update grad 
				const float delta = s-backU[uid];
                PCDMatrixS<Single>::RLine line=u2inst[uid];
                if (!isBias){
                    for (unsigned j=0;j<line.length;++j){
                        const unsigned i=line[j].cindex;
					    const int iid=itemID[i];
                        if( pred != NULL ){
                            pred[i]+=delta*WVLine[iid];
                            grad[i]=loss.calcGrad(pred[i],rate[i])*weight[i];
                        }else{
                            grad[i]+=loss.calcHess()*delta*WVLine[iid]*weight[i];                   
                        }
					}
                }else{
                    for (unsigned j=0;j<line.length;++j){
                        const unsigned i=line[j].cindex;
                        if( pred != NULL ){
                            pred[i]+=delta;
                            grad[i]=loss.calcGrad(pred[i],rate[i])*weight[i];
                        }else{
                            grad[i]+=loss.calcHess()*delta*weight[i];
                        }
					}
				}
            }
        }
    public:
        virtual void update( apex_tensor::CTensor1D PUFLine,
                             apex_tensor::CTensor1D WULine,
                             const apex_tensor::CTensor1D WVLine,
                             std::vector<float> &grad,
                             std::vector<float> &pred,
                             const std::vector<int>   &itemID,
                             const std::vector<float> &rate,
                             const std::vector<float> &weight,
                             const PCDMatrixS<Single> &u2inst,
                             const PCDMatrixS<Pair> &u2uf,
                             const PCDMatrixS<Pair> &uf2u,
                             const PCDMatrixS<Single> &p2u,
                             const std::vector<float> &ufInsCnt,
                             const PCDTrainParam::Reg reg,
                             bool isBias 
                             ) {
            this->getGH_BackU( &guser[0], &huser[0], &backU[0], 
                               WULine, WVLine,
                               &grad[0], &itemID[0], &weight[0],
                               u2inst, isBias );
            updater->update( PUFLine, &guser[0], &huser[0], ufInsCnt, u2uf, uf2u, p2u, reg );
            this->modify_U_Grad( WULine, &grad[0], 
                                 pred.size() == 0 ? NULL: &pred[0], 
                                 PUFLine, WVLine, &itemID[0], &rate[0], &weight[0], &backU[0], 
                                 u2inst, u2uf, isBias );
        }
    };
};
#endif
