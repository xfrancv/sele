/*=================================================================
 * [Y,F] = maxsum_feasible(x,model)
 *
 * It uses a brutal force to find all feasible labelings (> -inf) of 
 * input maxsum problem. It works only for a toy probles otherwise
 * it quickly runs out of memmory.
 *
 * History:
 * 02-mar-07, VF
 * 19-jul-06, VF
 *  
 *=================================================================*/

#include <stdio.h>
#include <string.h>
#include "mex.h"

#define INDEX2(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define INDEX3(ROW,COL,DEPTH,NUM_ROWS,NUM_COLS) ((DEPTH)*(NUM_ROWS)*(NUM_COLS) + (COL)*(NUM_ROWS)+(ROW))
#define MAX(A,B) ((A) < (B) ? (B) : (A))

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{
   uint32_T   *x, *E, *q, *Y, *new_Y, *tmp_uint32_ptr; 
   uint32_T   nT, nY, nX, nG, nQ, nE, y;
   uint32_T   *nhood1, *nhood2;
   uint32_T   num_nhood1, num_nhood2;
   uint32_T   i,j,cnt,t,num_labs,edge,tt;
   mxArray    *tmp;
   int        ndims; 
   int        dims[2];
   const int  *tmp_dims;
   double     *G, *Q, *F, *new_F, *tmp_double_ptr, maxsum;
   
   
   x = (uint32_T*)mxGetPr(prhs[0]);
   
   nT = MAX(mxGetN( prhs[0] ),mxGetM( prhs[0] ));
   
   tmp = mxGetField(prhs[1],0,"E");
   E = (uint32_T*)mxGetPr( tmp );
   nE = mxGetN(tmp);
   
   tmp = mxGetField(prhs[1],0,"G");
   ndims = mxGetNumberOfDimensions(tmp);
   if( ndims == 2) nG = 1;
   else {tmp_dims = mxGetDimensions(tmp); nG = tmp_dims[2];}
   G = (double*)mxGetPr( tmp );

   tmp = mxGetField(prhs[1],0,"Q");
   ndims = mxGetNumberOfDimensions(tmp);
   tmp_dims = mxGetDimensions(tmp);
   if( ndims == 2) nQ = 1; else nQ = tmp_dims[2];
   Q = (double*)mxGetPr( tmp );
   nX = tmp_dims[0];
   nY = tmp_dims[1];
   
   tmp = mxGetField(prhs[1],0,"q");
   q = (uint32_T*)mxGetPr( tmp );
   
/*   mexPrintf("nX=%d, nY=%d, nG=%d, nQ=%d, nE=%d, nT=%d\n", nX, nY, nG, nQ, nE, nT);*/

   /*--------------------------------------------------*/
   nhood1 = mxCalloc(nE,sizeof(uint32_T));
   nhood2 = mxCalloc(nE,sizeof(uint32_T));
   
   Y = mxCalloc(nY,sizeof(uint32_T));
   F = mxCalloc(nY,sizeof(double));
   if( Y==NULL || F==NULL) mexErrMsgTxt("Not enough memory");

   cnt = 0;
   for( y = 1; y <= nY; y++ )
   {
      if( Q[INDEX3(x[0]-1,y-1,q[0]-1,nX,nY)] > -mxGetInf() ) 
      {
         F[cnt] = Q[INDEX3(x[0]-1,y-1,q[0]-1,nX,nY)];
         Y[cnt] = y;
         cnt ++;
      }
   }
  
   for(t=2; t <= nT; t++ )
   {
      num_labs = cnt;
       
      new_Y = mxCalloc(nY*num_labs*t,sizeof(uint32_T));
      new_F = mxCalloc(nY*num_labs,sizeof(double));
      if( Y==NULL || F==NULL) mexErrMsgTxt("Not enough memory");
      
      cnt = 0;
      
      num_nhood1 = 0; num_nhood2 = 0;
      for(i=0; i < nE; i++ ) 
      { 
        if( E[INDEX2(0,i,3)] == t && E[INDEX2(1,i,3)] < t ) nhood1[num_nhood1++] = i; 
        if( E[INDEX2(1,i,3)] == t && E[INDEX2(0,i,3)] < t ) nhood2[num_nhood2++] = i; 
      }
      
      for(j = 0; j < num_labs; j++ )
      {
         
         for(y=1; y <= nY; y++ )
         {
            maxsum = F[j] + Q[INDEX3(x[t-1]-1,y-1,q[t-1]-1,nX,nY)];

            for(i = 0; i < num_nhood1 && maxsum > -mxGetInf(); i++ )
            {
               edge = E[INDEX2(2,nhood1[i],3)];
               tt = E[INDEX2(1,nhood1[i],3)];
                 
                maxsum = maxsum + G[INDEX3(y-1,Y[INDEX2(j,tt-1,num_labs)]-1,edge-1,nY,nY)];
            }

            for(i = 0; i < num_nhood2 && maxsum > -mxGetInf(); i++ )
            {
               edge = E[INDEX2(2,nhood2[i],3)];
               tt = E[INDEX2(0,nhood2[i],3)];
                  
               maxsum = maxsum + G[INDEX3(Y[INDEX2(j,tt-1,num_labs)]-1,y-1,edge-1,nY,nY)];
            }

            if(maxsum > -mxGetInf())
            {
               for(i=0; i < t-1; i++ ) 
               { 
                  new_Y[INDEX2(cnt,i,nY*num_labs)] = Y[INDEX2(j,i,num_labs)]; 
               }
               
               new_Y[INDEX2(cnt,t-1,nY*num_labs)] = y;
               
               new_F[cnt] = maxsum;
               
               cnt++;
            }
            
         }

          
      } /* for(j = 0; j < num_labs; j++ ) */
      
      
      mxFree(Y); mxFree(F);
      
      if(cnt == 0) {
         plhs[0] = mxCreateDoubleMatrix(0, 0, 0);
         plhs[1] = mxCreateDoubleMatrix(0, 0, 0);
         return;
      }
      
      Y = mxCalloc(cnt*t,sizeof(uint32_T));
      F = mxCalloc(cnt,sizeof(double));
      if( Y==NULL || F==NULL) mexErrMsgTxt("Not enough memory");
      
      for(j=0; j < cnt; j++ )
      {
         for(i=0; i < t; i++)
         {  
           Y[INDEX2(j,i,cnt)] = new_Y[INDEX2(j,i,nY*num_labs)]; 
         }
         
         F[j] = new_F[j];
      }      
   }
   
   
   /*--------------------------------------------*/ 
   
   dims[0] = cnt; dims[1] = nT;
   plhs[0] = mxCreateNumericArray(2, dims, mxUINT32_CLASS, mxREAL);
   tmp_uint32_ptr = (uint32_T*) mxGetPr( plhs[0] );

   plhs[1] = mxCreateDoubleMatrix(cnt, 1, 0);
   tmp_double_ptr = (double*)mxGetPr(plhs[1]);
   
   for( j = 0; j < cnt; j++ ) 
   { 
      for(i=0; i < nT; i++)
      {
         tmp_uint32_ptr[INDEX2(j,i,cnt)] = Y[INDEX2(j,i,cnt)]; 
      }
      
      tmp_double_ptr[j] = F[j];
   }
   
   return;
}
