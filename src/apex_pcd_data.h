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
 * \file apex_pcd_data.h
 * \brief data structure for PCD
 * \author Jingbo Shang, Tianqi Chen: {shangjingbo,tqchen}@apex.sjtu.edu.cn
 */
#ifndef _APEX_PCD_DATA_H_
#define _APEX_PCD_DATA_H_

#include <cstdlib>
#include <vector>
#include <algorithm>
#include "../apex-utils/apex_matrix_csr.h"

namespace apex_svd{	
	/*!\brief the varible type used for sum*/
    typedef double pcd_sum_float;

	/*!\brief template square function*/
    template<class T>
    inline T sqr(T x){
        return x*x;
    }
    
    /*!\brief Matrix Entry Type: Pair*/
    struct Pair{
        /*!\brief column index */
        unsigned cindex;
        /*!\brief content value */
        float    fvalue;
        
        Pair(){}
        Pair(unsigned i,float v):cindex(i),fvalue(v){}
        
        /*!\brief ordered by index*/
        inline bool operator <(const Pair &other) const{
        	return this->cindex<other.cindex;
        }
    };
    
    /*!\brief Matrix Entry Type: Single*/
    struct Single{
        /*!\brief column index */
        unsigned cindex;
        
        Single(){}
        Single(unsigned i):cindex(i){}
        
        /*!\brief orderd by index*/
        inline bool operator <(const Single &other) const{
        	return this->cindex<other.cindex;
        }
    };
    
    /*!\brief sparse matrix class */
    template<class Entry>
    class PCDMatrixS{
    public:
        /*!\brief one row */
        struct RLine{
            const Entry *pEntry;
            unsigned length;                        
            inline const Entry &operator[]( size_t i ) const{
                return pEntry[ i ];
            }
        };
        /*!\brief sparse matrix class */
        std::vector<size_t> rptr;
        /*!\brief data content */
        std::vector<Entry>  data;
        /*!\brief matrix builder */
        apex_utils::SparseCSRMBuilder<Entry> *builder;
    public:
        PCDMatrixS(){
            builder=new apex_utils::SparseCSRMBuilder<Entry>( rptr, data );
        }
        
        ~PCDMatrixS(){
            delete builder;
        }
        
        inline size_t getSize() const{
            return data.size();
        }
    
        inline size_t numRow( void ) const{
            return rptr.size() - 1;
        }
        
        inline RLine operator[]( size_t i ) const{
            apex_utils::assert_true( i + 1 < rptr.size(), "row index exceed bound" );
            RLine l;
            l.pEntry = &data[ rptr[i] ];
            l.length = static_cast<unsigned>( rptr[i+1] - rptr[i] );
            return l;
        }
        /*!\brief initialize the number of rows*/
        inline void init_budget(int row){
            builder->init_budget(row);
        }
        /*!\brief add a slot at a row*/
        inline void add_budget(int rowid){
            builder->add_budget(rowid);
        }
        /*!\brief allocate space for all entries after added*/
        inline void init_storage(){
            builder->init_storage();
        }
        /*!\brief push an element, should be done after init_storage*/
        inline void push_elem(int rowid,Entry e){
            builder->push_elem(rowid,e);
        }
        /*!\brief load the matrix from a binary file*/
        inline void loadBinary( FILE *fi ){
            size_t nrows, nentry;            
            apex_utils::assert_true( fread( &nrows,sizeof(size_t), 1, fi ) > 0, "load" );
            apex_utils::assert_true( fread( &nentry,sizeof(size_t), 1, fi ) > 0, "load" ); 
            rptr.resize( nrows + 1 );
            data.resize( nentry );
            if( nentry > 0 ){
                apex_utils::assert_true( fread( &rptr[0],sizeof(size_t), rptr.size(), fi ) > 0, "load" );
                apex_utils::assert_true( fread( &data[0],sizeof(Entry), data.size(), fi ) > 0, "load" );
            }
        }
		/*!\brief save the matrix to a binary file*/
        inline void writeBinary( FILE *fo ) const{
            size_t nrows = numRow(), nentry = getSize();            
            fwrite( &nrows,sizeof(size_t), 1, fo );
            fwrite( &nentry,sizeof(size_t), 1, fo );
            if( nentry > 0 ){
                fwrite( &rptr[0],sizeof(size_t), rptr.size(), fo );
                fwrite( &data[0],sizeof(Entry), data.size(), fo );
            }
        }
        /*!\brief check the balance to see the approximate upper bound of the speedup of nthreads*/
        inline void checkBalance( int nthreads, const char *name ){
        	if (nthreads==0){
        		return;
        	}
            int maxe = 0;
            int step = (int)(numRow()+nthreads)/nthreads;
            for( int i = 0; i < nthreads; i ++ ){
                int l = i*step, r = std::min( (i+1)*step, (int)numRow() );
                int nentry = static_cast<int>(rptr[ r ] - rptr[ l ]);
                if( maxe < nentry ) maxe = nentry;
            }
            printf("balance[%s]:%lf vs %d\n", name, (((float)(rptr.back()- rptr[0]))/maxe), nthreads );
        }
        /*!\brief shuffle entries in the same row. rand seed is set to 0 defaultly*/
        inline void shuffle(unsigned seed=0){
        	srand(seed);
        	for (size_t i=0;i+1<rptr.size();++i){
        		std::random_shuffle(data.begin()+rptr[i],data.begin()+rptr[i+1]);
        	}
		}
		/*!\brief sort entries in the same row by col-index*/
		inline void sort(){
			for (size_t i=0;i+1<rptr.size();++i){
        		std::sort(data.begin()+rptr[i],data.begin()+rptr[i+1]);
        	}
		}
    };
};
#endif
