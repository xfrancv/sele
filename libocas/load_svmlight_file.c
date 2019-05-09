/*=================================================================
 *  [feat,labels] = load_svmlight_format(file_name)
 *  [feat,labels] = load_svmlight_format(file_name,verb)
 *
 *  This function reads examples from a file complaying to SVM^light 
 *  format. 
 *
 *  
 * 
 *
 *=================================================================*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <mex.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>

#if !defined(MX_API_VER) || MX_API_VER<0x07040000
#define mwSize int
#define INDEX_TYPE_T int
#define mwIndex int
#else
#define INDEX_TYPE_T mwSize
#endif

#include "lib_svmlight_format.h"

#define MaxExamples   50000000

#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define ABS(A) ((A) < 0 ? -(A) : (A))
#define INDEX2(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))


/*======================================================================
  Main code plus interface to Matlab.
========================================================================*/

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{
  char fname[1000];
  FILE *fid;
  char *line;
  double *feat_val;
  uint32_t *feat_idx;
  long nnzf;
  mxArray *W;
  long nDims = 0;
  long nData;
  long i, j;
  long nnz = 0;
  double max_data_norm2 = -mxGetInf();
  mwIndex *irs, *jcs;
  double *sr;
  int verb=0;

  if( nrhs < 1 )
    mexErrMsgTxt("At least one input argument required.\n"
                 "Synopsis:\n"
                 " [feat,labels] = load_svmlight_format(file_name)\n"
                 " [feat,labels] = load_svmlight_format(file_name,verb)\n"
                 " \n"
                 );
  if( nrhs >= 2)
    verb = (int)mxGetScalar(prhs[1]);
  else
    verb = 1;

  /* get input arguments */
  mxGetString(prhs[0], fname, 1000);

  if(verb)
    mexPrintf("Input file: %s\n", fname);

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error: ");
    mexErrMsgTxt("Cannot open input file.");
  }

  /**********************************/
  line = mxCalloc(LIBSLF_MAXLINELEN, sizeof(char));
  if( line == NULL )
    mexErrMsgTxt("Not enough memmory to allocate line buffer.");

  feat_idx = mxCalloc(LIBSLF_MAXLINELEN, sizeof(uint32_t));
  if( feat_idx == NULL )
    mexErrMsgTxt("Not enough memmory to allocate feat_idx.");

  feat_val = mxCalloc(LIBSLF_MAXLINELEN, sizeof(double));
  if( feat_val == NULL )
    mexErrMsgTxt("Not enough memmory to allocate feat_val.");



  /*********************************************/
  /* Main code                                 */
  /*********************************************/

  if(verb)
    mexPrintf("Analysing input data...");

  double label;
  int go = 1;
  long line_cnt = 0;

  while(go) {
    
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL ) 
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0) 
          mexPrintf(" %d", line_cnt);
        mexPrintf(" EOF.\n");
      }

    }
    else
    {
      line_cnt ++;
      nnzf = svmlight_format_parse_line_doubley(line, &label, feat_idx, feat_val);
      
      if(nnzf == -1) 
      {
         mexPrintf("Parsing error on line %d .\n", line_cnt);
         mexErrMsgTxt("Defective input file.");
      }

      double norm2 = 0;
      for(j = 0; j < nnzf; j++)
        norm2 += feat_val[j]*feat_val[j];

      max_data_norm2 = MAX(max_data_norm2,norm2);

      nDims = MAX(nDims,feat_idx[nnzf-1]);

      nnz += nnzf;
      
      if( (line_cnt % 1000) == 0) {
        if(verb)
        {
          mexPrintf(" %d", line_cnt);
          fflush(NULL);
        }
      }
    }
  }

  nData = line_cnt;

  fclose(fid);  
  if(verb)
  {
    mexPrintf("Number of examples: %d\n", nData);
    mexPrintf("Dimensions: %d\n", nDims);
    mexPrintf("nnz: %d, density: %f%%\n", nnz, 100*(double)nnz/((double)nDims*(double)nData) );
    mexPrintf("max_i ||x_i||^2: %f\n", max_data_norm2);
  }

  /*---------------------------------------------*/


  mxArray* sp_mat_X = mxCreateSparse(nDims, nData, nnz, mxREAL);
  if( sp_mat_X == NULL)
    mexErrMsgTxt("Not enough memory to allocate sp_mat_X");
  plhs[0] = sp_mat_X;

  plhs[1] = mxCreateDoubleMatrix(nData,1,mxREAL);
  if( plhs[1] == NULL)
    mexErrMsgTxt("Not enough memory to allocate vec_y.");
  double *vec_y = mxGetPr(plhs[1]);

  sr  = mxGetPr(sp_mat_X);
  irs = mxGetIr(sp_mat_X);
  jcs = mxGetJc(sp_mat_X);

  fid = fopen(fname, "r");
  if(fid == NULL) {
    perror("fopen error: ");
    mexErrMsgTxt("Cannot open input file.");
  }

  if(verb)
    mexPrintf("Reading examples...");
  
  go = 1;
  line_cnt = 0;
  long k=0;
  while(go) {
    if(fgets(line,LIBSLF_MAXLINELEN, fid) == NULL ) 
    {
      go = 0;
      if(verb)
      {
        if( (line_cnt % 1000) != 0) 
          mexPrintf(" %d", line_cnt);
        mexPrintf(" EOF.\n");
      }
    }
    else
    {
      line_cnt ++;
      nnzf = svmlight_format_parse_line_doubley(line, &label, feat_idx, feat_val);
      
      if(nnzf == -1) 
      {
         mexPrintf("Parsing error on line %d .\n", line_cnt);
         mexErrMsgTxt("Defective input file.");
      }

      vec_y[line_cnt-1] = (double)label;

      jcs[line_cnt-1] = k;

      for(j = 0; j < nnzf; j++) {
        sr[k] = feat_val[j];
        irs[k] = feat_idx[j]-1;
        k++;
      }
      
      if(verb)
      {
        if( (line_cnt % 1000) == 0) {
          mexPrintf(" %d", line_cnt);
          fflush(NULL);
        }
      }
    }
  }
  jcs[line_cnt] = k;

  plhs[0] = sp_mat_X;

  fclose(fid);

  return;
}

