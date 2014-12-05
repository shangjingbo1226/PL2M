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
 * \file apex_pcd_update.h
 * \brief interface classes for PCD
 * \author JingboShang, Tianqi Chen: {shangjingbo,tqchen}@apex.sjtu.edu.cn
 */
#ifndef _APEX_PCD_UPDATE_H_
#define _APEX_PCD_UPDATE_H_

#include <algorithm>
#include <vector>
#include <omp.h>

namespace apex_svd{
	inline int getBlock(int nthread) {
		if (nthread == 0) {
			return 64 * 32;
		}
		return nthread * 8;
	}
    class IPCDUpdater{
    public:
        /*! 
         * \brief initialize the auxiliary space
         */        
        virtual void init( void ){}        
        virtual void update( apex_tensor::CTensor1D pParam,
                             float *grad,
                             const float *hess, 
                             const std::vector<float> &insCnt,
                             const PCDMatrixS<Pair> &rowMatrix,
                             const PCDMatrixS<Pair> &colMatrix,
                             const PCDMatrixS<Single> &p2u,
                             const PCDTrainParam::Reg &reg ) = 0;
    };    
};

namespace apex_svd{
    /*!\brief simple implementation of CD Updater */
    class SimpleCDUpdater : public IPCDUpdater{
    protected:
        const PCDTrainParam &param;
    public:
        SimpleCDUpdater( const PCDTrainParam &param )
            :param(param){}
        virtual void update( apex_tensor::CTensor1D pParam,
                             float *grad,
                             const float *hess, 
                             const std::vector<float> &insCnt,
                             const PCDMatrixS<Pair> &rowMatrix,
                             const PCDMatrixS<Pair> &colMatrix,
                             const PCDMatrixS<Single> &p2u,
                             const PCDTrainParam::Reg &reg ){
            int numFeature = (int)colMatrix.numRow();
            for (int i=0;i<numFeature;++i){
                PCDMatrixS<Pair>::RLine line=colMatrix[i];
                pcd_sum_float sg=0.0f,sh=0.0f;
                for (unsigned j=0;j<line.length;++j){
                    const unsigned id = line[j].cindex;
                    const float value = line[j].fvalue;
                    sg += value*grad[id];
                    sh += sqr(value)*hess[id];
                }
                float &ref = pParam[ i ];
                float delta = reg.calcDelta( sg, sh, ref,insCnt[i] ) * param.learningRate;
                ref += delta * param.shrinkRate;
                for (unsigned j=0;j<line.length;++j){
                    const unsigned id = line[j].cindex;
                    const float value = line[j].fvalue;
                    grad[id] += delta * hess[id] * value;
                }
            }
        }
    };    
};
namespace apex_svd{
    /*!\brief simple implementation of CD Updater */
    class OpenMPCDUpdater: public SimpleCDUpdater{
    protected:
        const PCDModel::Param &loss;
        std::vector<float>    deltaP, sumConflict;
        std::vector<unsigned> updatePtr;
    public:
        OpenMPCDUpdater( const PCDModel::Param &loss, const PCDTrainParam &param )
            :SimpleCDUpdater(param), loss(loss){}
        virtual void update( apex_tensor::CTensor1D pParam,
                             float *grad,
                             const float *hess, 
                             const std::vector<float> &insCnt,
                             const PCDMatrixS<Pair> &rowMatrix,
                             const PCDMatrixS<Pair> &colMatrix,
                             const PCDMatrixS<Single> &p2u,
                             const PCDTrainParam::Reg &reg ){
            if( param.nthread == 0 ){
                SimpleCDUpdater::update( pParam, grad, hess, insCnt, rowMatrix, colMatrix, p2u, reg ); return;
            }
            this->updatePart( pParam, grad, &sumConflict[0], &deltaP[0], &updatePtr[0], 
                              hess, insCnt, rowMatrix, colMatrix, p2u, reg );
        }
        virtual void init( void ){
            sumConflict.resize( std::max( loss.numUser, loss.numItem ) );
            updatePtr.resize( std::max( loss.numUser, loss.numItem ) );
            deltaP.resize( std::max( loss.numFeatUser, loss.numFeatItem ) );
        }
    private:
        virtual void updatePart( apex_tensor::CTensor1D pParam,
                                 float *grad,
                                 float *sumConflict,
                                 float *deltaP,
                                 unsigned *ptr,
                                 const float *hess,           
                                 const std::vector<float> &insCnt,      
                                 const PCDMatrixS<Pair> &rowMatrix,
                                 const PCDMatrixS<Pair> &colMatrix,
                                 const PCDMatrixS<Single> &p2u,
                                 const PCDTrainParam::Reg &reg ){
            const int numIns = (int)rowMatrix.numRow();
            const int step = (int)param.sizePCDBlock;
            const int numFeature = (int)colMatrix.numRow();
            const int numPart = (numFeature+step-1)/step;
            
            //#pragma omp parallel for schedule( dynamic, PCD_INST_CHUNK_SIZE )
            #pragma omp parallel for schedule( static )
            for (int i=0;i<numIns;++i){
                ptr[i]=(unsigned)rowMatrix.rptr[i];
            }
            
            for (int part=0;part<numPart;++part){
                int st=part*step, ed=std::min(numFeature,st+step);                
                PCDMatrixS<Single>::RLine userSet=p2u[part];
                
                const int user_chunk_size = std::max((int)userSet.length/getBlock(param.nthread),1);
                #pragma omp parallel for schedule( dynamic, user_chunk_size )
                for (int k=0;k<(int)userSet.length;++k){
                    const int i = userSet[k].cindex;
                    float s = 0.0f;
                    unsigned end=(unsigned)rowMatrix.rptr[i+1];
                    for (unsigned j=ptr[i];j<end;++j){
                        int fid=rowMatrix.data[j].cindex;
                        if (fid>=ed) break;
                        s += fabsf( rowMatrix.data[j].fvalue );
                    }
                    sumConflict[ i ]=s;
                }

                const int userfeat_chunk_size = std::max((ed-st)/getBlock(param.nthread),1);
                #pragma omp parallel for schedule( dynamic, userfeat_chunk_size )
                for( int i = st; i < ed; i ++ ){
                    PCDMatrixS<Pair>::RLine line=colMatrix[i];
                    pcd_sum_float sg=0,sh=0;
                    for (unsigned j=0;j<line.length;++j){
                        const unsigned id = line[j].cindex;
                        const float value = line[j].fvalue;
                        sg+=value*grad[id];
                        sh+=fabsf( value )*sumConflict[id]*hess[id];
                    }
                    float &ref = pParam[ i ];
                    float delta = reg.calcDelta(sg,sh,ref,insCnt[i]) * param.learningRate;
                    ref += delta * param.shrinkRate;
                    deltaP[ i ] = delta;
                }
                
                #pragma omp parallel for schedule( dynamic, user_chunk_size )
                for (int k=0;k<(int)userSet.length;++k){
                    int i=userSet[k].cindex;
                    float ds=0;                    
                    const unsigned end=(unsigned)rowMatrix.rptr[i+1];
                    unsigned j=ptr[i];
                    for (;j<end;++j){
                        int id = rowMatrix.data[j].cindex;
                        if (id>=ed) break;                        
                        const float value = rowMatrix.data[j].fvalue;                        
                        const float delta = deltaP[id];                        
                        ds += delta*value;
                    }
                    ptr[i]=j;
                    grad[i] += hess[i] * ds;
                }
            }                      
        }
    };
};


namespace apex_svd{
    /*!\brief implementation using block update */
    class BlockPCDUpdater: public OpenMPCDUpdater{
    private:
        // step of max instance
        int maxInst;
        struct ThreadData{
            // pointer to private gradient and hess
            // hess record hess[i] * (sum_{t\in S} |X_{it}|) / (sum_{t\in S_r} |X_{it}|), where r is thread id
            float grad, hess;
        };
        // private storage for gradient, hessian statistics
        std::vector<ThreadData> tdata;

        inline ThreadData* getThreadDArray( int tid ){
            return &tdata[ tid * maxInst ];
        }
        inline ThreadData& getThreadDElem( int tid, int instID ){
            return tdata[ tid * maxInst + instID];
        }
    public:
        BlockPCDUpdater( const PCDModel::Param &loss, const PCDTrainParam &param )
            :OpenMPCDUpdater( loss, param ){}

        virtual void update( apex_tensor::CTensor1D pParam,
                             float *grad,
                             const float *hess, 
                             const std::vector<float> &insCnt,
                             const PCDMatrixS<Pair> &rowMatrix,
                             const PCDMatrixS<Pair> &colMatrix,
                             const PCDMatrixS<Single> &p2u,
                             const PCDTrainParam::Reg &reg ){
            if( param.blockUpdate == 0 || param.nthread == 0 ){
                OpenMPCDUpdater::update( pParam, grad, hess, insCnt, rowMatrix, colMatrix, p2u, reg ); return;
            }
            this->updateBlock( pParam, grad, &sumConflict[0], &deltaP[0], &updatePtr[0], 
                              hess, insCnt, rowMatrix, colMatrix, p2u, reg );

        }
        virtual void init( void ){
            if( param.blockUpdate == 0 || param.nthread == 0 ){
                OpenMPCDUpdater::init();
            }else{
                this->maxInst = std::max( loss.numUser, loss.numItem );
                sumConflict.resize( maxInst );
                updatePtr.resize( maxInst );
                deltaP.resize( maxInst );
                tdata.resize( maxInst * param.nthread );
            }
        }
    private:
        virtual void updateBlock( apex_tensor::CTensor1D pParam,
                                  float *grad,
                                  float *sumConflict,
                                  float *deltaP,
                                  unsigned *ptr,
                                  const float *hess,           
                                  const std::vector<float> &insCnt,
                                  const PCDMatrixS<Pair> &rowMatrix,
                                  const PCDMatrixS<Pair> &colMatrix,
                                  const PCDMatrixS<Single> &p2u,
                                  const PCDTrainParam::Reg &reg ){
            const int numIns = (int)rowMatrix.numRow();
            const int step = (int)param.sizePCDBlock;
            const int numFeature = (int)colMatrix.numRow();
            const int numPart = (numFeature+step-1)/step;
            
            #pragma omp parallel for schedule( static )
            for (int i=0;i<numIns;++i){
                ptr[i]=(unsigned)rowMatrix.rptr[i];
            }
            
            for (int part=0;part<numPart;++part){
                int st=part*step, ed=std::min(numFeature,st+step);

                // step for each thread in this part
                const int tstep = ( ed - st + param.nthread -1 ) / param.nthread;                
                PCDMatrixS<Single>::RLine userSet=p2u[part];
                
                const int user_chunk_size = std::max((int)userSet.length/getBlock(param.nthread),1);
                #pragma omp parallel for schedule( dynamic, user_chunk_size )
                for (int k=0;k<(int)userSet.length;++k){
                    const int i = userSet[k].cindex;
                    const unsigned end=(unsigned)rowMatrix.rptr[i+1];
                    float sumConfAll = 0.0f;
                    for (unsigned j=ptr[i];j<end;++j){
                        const int fid=rowMatrix.data[j].cindex;
                        if ( fid >= ed ) break;
                        sumConfAll += fabsf( rowMatrix.data[j].fvalue );
                    }
                    // calculate sumConflict in each region
                    int tid = 0; 
                    float ts = 0.0f;
                    for ( unsigned j=ptr[i];j<end;++j ){
                        const int fid=rowMatrix.data[j].cindex;
                        if ( fid >= ed ) break;
                        // a new tid arrived
                        const int ctid = ( fid - st )/ tstep;
                        if( ctid != tid ){
                            // record statistics for old tid
                            ThreadData &d = this->getThreadDElem( tid, i );
                            d.grad = grad[ i ]; 
                            d.hess = ts > eps ? hess[ i ] * sumConfAll / ts : 0.0f;
                            tid = ctid; ts = 0.0f;
                        }
                        ts += fabsf( rowMatrix.data[j].fvalue );
                    }
                    {// remember last tid
                        ThreadData &d = this->getThreadDElem( tid, i );
                        d.grad = grad[ i ]; d.hess = ts > eps ? hess[ i ] * sumConfAll / ts : 0.0f;
                    }                          
                }
                                
                #pragma omp parallel
                {   // get thread id, this area is ran by each thread
                    const int tid = omp_get_thread_num();
                    const int tstart = st + tid * tstep;
                    const int tend   = std::min( tstart + tstep, ed );
                    ThreadData *tdata = this->getThreadDArray( tid );
                    
                    for( int i = tstart; i < tend; i ++ ){
                        PCDMatrixS<Pair>::RLine line=colMatrix[i];
                        pcd_sum_float sg=0,sh=0;
                        for (unsigned j=0;j<line.length;++j){
                            const unsigned id = line[j].cindex;
                            const float value = line[j].fvalue;
                            sg += value * tdata[id].grad;
                            sh += sqr( value ) * tdata[id].hess;
                        }
                        float &ref = pParam[ i ];
                        float delta = reg.calcDelta(sg,sh,ref,insCnt[i]) * param.learningRate;
                        ref += delta * param.shrinkRate;
                        deltaP[ i ] = delta;
                        for (unsigned j=0;j<line.length;++j){
                            const unsigned id = line[j].cindex;
                            const float value = line[j].fvalue;
                            tdata[ id ].grad += delta * value * tdata[ id ].hess;
                        }
                    }
                }
                
                // as old ways, update gradient 
                #pragma omp parallel for schedule( dynamic, user_chunk_size )
                for (int k=0;k<(int)userSet.length;++k){
                    int i=userSet[k].cindex;
                    float ds=0;                    
                    const unsigned end=(unsigned)rowMatrix.rptr[i+1];
                    unsigned j=ptr[i];
                    for (;j<end;++j){
                        int id = rowMatrix.data[j].cindex;
                        if (id>=ed) break;                        
                        const float value = rowMatrix.data[j].fvalue;                        
                        const float delta = deltaP[id];
                        ds += delta*value;
                    }
                    ptr[i]=j;
                    grad[i] += hess[i] * ds;
                }
            }                        
        }
    };
};
#endif
