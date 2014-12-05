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
 * \file apex_pcd_model.h
 * \brief model file for PCD
 * \author Jingbo Shang, Tianqi Chen: {shangjingbo,tqchen}@apex.sjtu.edu.cn
 */
#ifndef _APEX_PCD_MODEL_H_
#define _APEX_PCD_MODEL_H_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <algorithm>
#ifdef _MSC_VER
#include <cfloat>
#endif

#include "../apex-utils/apex_utils.h"
#include "../apex-tensor/apex_tensor.h"

namespace apex_svd
{
    class PCDModel
    {
    public:
        struct Param
        {
            /*! \brief the number of users*/
            int numUser;
            /*! \brief the number of items*/
            int numItem;
            /*! \brief the dimension of latent factor*/
            int numDimension;
            /*! \brief the number of user features*/
            int numFeatUser;
            /*! \brief the number of item features*/
            int numFeatItem;
            /*! \brief the type of loss function */
            int lossType;
            /*! \brief standard variance of initialization */
            float initSigma;
            /*! \brief mean value */            
            float baseScore;
            /*! \brief scale of input score */
            float scale;
            int reserved[252];
            
            const static int SquareLoss     =   0xa0000;            
            const static int LogisticLoss   =   0xa0001;
            
            Param( void ){
                numUser=numItem=numDimension=0;
                numFeatUser=numFeatItem=0;
                initSigma = 0.005f;
                this->lossType=SquareLoss;
                baseScore = 0.5f;
            }
            /*!\brief return the gradient of loss function*/
            inline float calcGrad(float pred,float rate) const
            {
                switch( lossType ){
                case SquareLoss: return pred-rate;
                case LogisticLoss: return 1/(1+expf(-pred))-rate;
                default: apex_utils::error("loss type unkown"); return 0.0f;
                }
            }
            /*!\brief return the uniform upper bound of hessian of loss function*/
            inline float calcHess( void ) const
            {
                switch (lossType){
                case SquareLoss: return 1.0f;
                case LogisticLoss: return 0.25f;
                default: apex_utils::error("loss type unkown"); return 0.0f;
                }
            }
            
            inline void setParam(const char* name,const char* value){
                std::stringstream in(value);
                if (!strcmp(name,"numUser")){
                    in >> numUser;
                    return;
                }
                if (!strcmp(name,"numItem")){
                    in >> numItem;
                    return;
                }
                if (!strcmp(name,"numDimension")){
                    in >> numDimension;
                    return;
                }
                if (!strcmp(name,"numFeatUser")){
                    in >> numFeatUser;
                    return;
                }
                if (!strcmp(name,"numFeatItem")){
                    in >> numFeatItem;
                    return;
                }
                if (!strcmp(name,"initSigma")){
                    in >> initSigma;
                    return;
                }
                if (!strcmp(name,"lossType")){
                    if( !strcmp(value,"square") ){
                        lossType = SquareLoss;
                    } else if (!strcmp(value,"logistic")){
                        lossType = LogisticLoss;
                    }else{
                        apex_utils::error("lossType not Supported");
                    }
                    return;
                }                
                if (!strcmp(name,"baseScore")){
                    in >> baseScore;
                    return;
                }
                if (!strcmp(name,"scale")){
                    in >> scale;
                    return;
                }
            }
            /*!\brief scale and transform the base score*/
            inline void initBaseScore( void ){
                baseScore/=scale;
                if (lossType==LogisticLoss){
                    apex_utils::assert_true( baseScore < 1.0f && baseScore > 0.0f,"logistic loss base must be in (0,1)");
                    baseScore = -logf( 1.0f/baseScore - 1.0f );
                }
            } 
        };
    public:
        /*!\brief model paramaters */
        Param param;
        // ----tmp buffer param----
        // NOTE: for all MATRIX, the maximum dimension is treat as bias, can be dropped
        /*!\brief latent factor of user */
        apex_tensor::CTensor2D WUser;
        /*!\brief latent factor of item */
        apex_tensor::CTensor2D WItem;
        /*!\brief projection matrix P */
        apex_tensor::CTensor2D PUFeat;
        /*!\brief projection matrix I*/
        apex_tensor::CTensor2D PIFeat;
    	/*!\brief return the prediction for <uid,iid>*/
        inline float predict(unsigned uid,unsigned iid) const{            
            float score = param.baseScore + WUser[ param.numDimension ][ uid ] + WItem[ param.numDimension ][ iid ];            
            for( int k = 0; k < param.numDimension; k ++ ){
                score += WUser[ k ][ uid ] * WItem[ k ][ iid ];
            }            
            return score;
        }
        /*!\brief return the real score of prediction <uid,iid>, scale and transform back*/
        inline float realScore(unsigned uid,unsigned iid) const{
			float score=predict(uid,iid);
			if (param.lossType==param.LogisticLoss){
                score = 1.0f / (1.0f + expf(-score) );
            }
            score*=param.scale;
            return score;
		}
        
    private:
		/*!\brief allocate space for model*/
        inline void allocSpace( void ){
            if( WUser.elem != NULL ) return;
            WUser.set_param( param.numDimension+1, param.numUser );
            WItem.set_param( param.numDimension+1, param.numItem );
            PUFeat.set_param( param.numDimension+1, param.numFeatUser );
            PIFeat.set_param( param.numDimension+1, param.numFeatItem);
            apex_tensor::tensor::alloc_space( WUser );
            apex_tensor::tensor::alloc_space( WItem );
            apex_tensor::tensor::alloc_space( PUFeat );
            apex_tensor::tensor::alloc_space( PIFeat );
        }
    public:
        PCDModel( void ){
            WUser.elem = NULL;
        }
        ~PCDModel( void ){
            if( WUser.elem != NULL ){
                apex_tensor::tensor::free_space( WUser );
                apex_tensor::tensor::free_space( WItem );
                apex_tensor::tensor::free_space( PUFeat );
                apex_tensor::tensor::free_space( PIFeat );
            }
        }
        inline void setParam( const char *name, const char *value ){
            if( WUser.elem == NULL ) param.setParam( name, value );
        }
        /*!\brief load model from a binary file*/
        inline void loadModel( FILE *fi ){
            apex_utils::assert_true( fread( &param, sizeof(Param), 1, fi ) != 0, "BUG");
            this->allocSpace();
            apex_tensor::cpu_only::load_from_file( WUser, fi, true );
            apex_tensor::cpu_only::load_from_file( WItem, fi, true );
            apex_tensor::cpu_only::load_from_file( PUFeat, fi, true );
            apex_tensor::cpu_only::load_from_file( PIFeat, fi, true );
        }
        /*!\brief save model to a binary file*/
        inline void saveModel( FILE *fo ) const{
            fwrite( &param, sizeof(Param), 1, fo );
            apex_tensor::cpu_only::save_to_file( WUser, fo );
            apex_tensor::cpu_only::save_to_file( WItem, fo );
            apex_tensor::cpu_only::save_to_file( PUFeat, fo );
            apex_tensor::cpu_only::save_to_file( PIFeat, fo );
        }
        /*!\brief allocate space, initialize the latent vectors and bias*/
        inline void initModel( void ){
            allocSpace();
            PUFeat = 0.0f;
            //apex_tensor::tensor::sample_gaussian( PUFeat, param.initSigma );
			//set PUFeat to 0, avoid the effects of the initial random score
            apex_tensor::tensor::sample_gaussian( PIFeat, param.initSigma );
            // set bias to 0
            PUFeat[ param.numDimension ] = 0.0f; PIFeat[ param.numDimension ] = 0.0f;
            param.initBaseScore();
        }       
    };
    
    // bound of all weight
    static int counterNan, counterInf, reportRound;        
	
	static const double eps = 1e-8;
	struct PCDTrainParam
    {
        struct Reg
        {
            float alpha;
            float lambda;            
            Reg(float alpha, float lambda):alpha(alpha),lambda(lambda){
            }
            /*!\brief Inf and Nan is not allowed*/
            inline void recordNan( void ) const{
                counterNan ++;
                if( counterNan % reportRound == 1 ){
                    fprintf( stderr, "counterNan=%d\n", counterNan );
                }
            }
            /*!\brief Inf and Nan is not allowed*/
            inline void recordInf( void ) const{
                counterInf ++;
                if( counterInf % reportRound == 1 ){
                    fprintf( stderr, "counterInf=%d\n", counterInf );
                }
            }
            /*!\brief return the delta according to the loss and regularizaiton function*/
            inline double calcDeltaInner(double x,double y,float w,float cnt) const{
                if( y < eps ) return 0.0f;
                double lambda=this->lambda*cnt;
                if( alpha < eps ){
                    return -(x+lambda*w)/(y+lambda);
                }
                double tmp=w-(x+lambda*w)/(y+lambda);
                if ( tmp >=0 ){
                    return std::max(-(x+lambda*w+alpha)/(y+lambda),-(double)w);
                }else{
                    return std::min(-(x+lambda*w-alpha)/(y+lambda),-(double)w);
                }
            }
            /*!\brief return delta, and check Nan and Inf*/
            inline float calcDelta( double x, double y, float w, float cnt ) const{
                float dw = static_cast<float>( calcDeltaInner( x, y, w, cnt ) );                
                const float wnew = w + dw;

#ifdef _MSC_VER
				if( _isnan( wnew ) ){
                    this->recordNan(); 
                    apex_utils::assert_true( _isnan( w ) == 0, "nan occured" );
                    return 0.0f;
                }
#else
				if( std::isnan( wnew ) ){
                    this->recordNan(); 
                    apex_utils::assert_true( !std::isnan( w ), "nan occured" );
                    return 0.0f;
                }
                if( std::isinf( wnew ) ){
                    this->recordInf(); 
                    apex_utils::assert_true( !std::isinf( w ), "inf occured" );
                    return 0.0f;
                }
#endif
                return dw;
            }
        };
        /*!\brief L1-norm coef*/
        float alphaUser, alphaItem;
        /*!\brief L2-norm coef*/
        float lambdaUser,lambdaItem,lambdaBias;
        /*! \brief learning rate parameter */
        float learningRate;
        /*! \brief shrinkage rate parameter */
        float shrinkRate;
        /*! \brief increase shrink rate by the ratio after each round */
        float incSR;
        /*! \brief max shrink rate we can get */
        float maxSR;
        /*! \brief burn in rounds of shrink Rate */
        int   burnSRRound;
        /*! \brief counter which round we have reached */
        int roundCounter;
        /*! \brief bias only round */
        int biasOnlyRound;
        /*! \brief flag used to help debug */
        int debugFlag;
        /*! \brief size of parallel CD block  */        
        int sizePCDBlock;
        /*! \brief which kind of update we are using  */        
        int updateMethod;
        /*! \brief number of parallel thread  */ 
        int nthread;
        /*! \brief use hess or not*/
        int useHess;
        /*! \brief shuffle dimension in each update */
        int shuffleDim;
        /*! \brief which kind of wise update we use */
        int updateWise;
        /*! \brief wheher use instance cnt */
        int useInsCnt;
        /*! \brief maximum number of round */
        int maxRound;
        /*! \brief whether to use block update */
        int blockUpdate;
        /*! \brief skip user bias or not */
        int skipUserBias;
        PCDTrainParam( void ){
            alphaUser = alphaItem = 0.0f;
            lambdaUser = lambdaItem = 1.0f;            
            roundCounter = 0;
            learningRate = 1.0f;
            shrinkRate = 1.0f;   maxSR = 0.2f; incSR = 0.0f; burnSRRound = 0;
            biasOnlyRound = 0; debugFlag = 0;
            sizePCDBlock = 1; nthread= 0; useHess = 1; updateMethod = 0; 
            shuffleDim = 1; updateWise = 0; useInsCnt = 0; maxRound = 40;
            counterNan = 0; counterInf = 0; reportRound = 10;
            skipUserBias = 0; blockUpdate = 0;
        }
        inline int getBlock() {
        	if (nthread == 0) {
        		return 64;
        	}
        	return abs(nthread) * 2 + 1;
        }
        /*!\brief return Reg for bias, L1-norm set to 0*/
        inline Reg getRegBias( void ) const{
            return Reg( 0.0f, lambdaBias );
        }
        /*!\brief return Reg for user*/
        inline Reg getRegUFeat( void ) const{
            return Reg( alphaUser, lambdaUser );
        }
        /*!\brief return Reg for item*/
        inline Reg getRegIFeat( void ) const{
            return Reg( alphaItem, lambdaItem );
        }
        inline void setParam(const char* name,const char* value)
        {
            std::stringstream in(value);
            if (!strcmp(name,"lambdaUserL1")){
                in >> alphaUser;
                return;
            }
            if (!strcmp(name,"lambdaUserL2")){
                in >> lambdaUser;
                return;
            }
            if (!strcmp(name,"lambdaItemL1")){
                in >> alphaItem;
                return;
            }
            if (!strcmp(name,"lambdaItemL2")){
                in >> lambdaItem;
                return;
            }
            if (!strcmp(name,"lambdaL2")){
                in >> lambdaUser;
                lambdaItem = lambdaUser;
                return;
            }
            if (!strcmp(name,"lambdaBias")){
                in >> lambdaBias;
                return;
            }
            if (!strcmp(name,"learningRate")){
                in >> learningRate;
                return;
            }
            if (!strcmp(name,"shrinkRate")){
                in >> shrinkRate;
                return;
            }
            if (!strcmp(name,"incSR")){
                in >> incSR;
                return;
            }
            if (!strcmp(name,"maxSR")){
                in >> maxSR;
                return;
            }
            if (!strcmp(name,"burnSRRound")){
                in >> burnSRRound;
                return;
            }
            if (!strcmp(name,"biasOnlyRound")){
                in >> biasOnlyRound;
                return;
            }
            if (!strcmp(name,"debugFlag")){
                in >> debugFlag;
                return;
            }
            if (!strcmp(name,"sizePCDBlock")){
                in >> sizePCDBlock;
                return;
            }
            if (!strcmp(name,"nthread")){
                in >> nthread;
                return;
            }
            if (!strcmp(name,"useHess")){
            	in >> useHess;
            	return;
            }
            if (!strcmp(name,"updateMethod")){
            	in >> updateMethod;
            	return;
            }
            if (!strcmp(name,"shuffleDim")){
            	in >> shuffleDim;
            	return;
            }
            if (!strcmp(name,"updateWise")){
            	in >> updateWise;
            	return;
            }
            if (!strcmp(name,"useInsCnt")){
            	in >> useInsCnt;
            	return;
            }
            if (!strcmp(name,"maxRound")){
            	in >> maxRound;
            	return;
            }
            if (!strcmp(name,"blockUpdate")){
				in >> blockUpdate;
				return;
			}
            if (!strcmp(name,"skipUserBias")){
				in >> skipUserBias;
				return;
			}
        }
        /*! \brief call at begining of each round to get round specific param right */
        inline void setRound( int round ){
            if( burnSRRound != 0  && incSR < 1e-6f ){
                incSR = ( maxSR - shrinkRate ) / burnSRRound;
            } 
            while( roundCounter < round ){
                roundCounter ++;
                if( incSR > 0.0f ) this->shrinkRate += this->incSR;
            }
            if( this->shrinkRate > maxSR ) this->shrinkRate = maxSR;
        }
    };
};

#endif

