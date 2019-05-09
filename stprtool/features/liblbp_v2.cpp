#include "liblbp_v2.h"


const int kUniformLBP[] = {
  1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 
  15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 
  20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 
  26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 
  48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58
};


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_ltp_pyr_features(char *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols, uint8_t th )
{
  uint32_t offset, ww, hh, x, y,center,j ;
  uint8_t pattern1;
  uint8_t pattern2;

  offset=0;
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        center = img[LIBLBP_INDEX(y,x,img_nRows)];

        pattern1 = 0;
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center-th) pattern1 = pattern1 | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center-th)   pattern1 = pattern1 | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center-th) pattern1 = pattern1 | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center-th)   pattern1 = pattern1 | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center-th)   pattern1 = pattern1 | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center-th) pattern1 = pattern1 | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center-th)   pattern1 = pattern1 | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center-th) pattern1 = pattern1 | 0x80;

        pattern2 = 0;
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] > center+th) pattern2 = pattern2 | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] > center+th)   pattern2 = pattern2 | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] > center+th) pattern2 = pattern2 | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] > center+th)   pattern2 = pattern2 | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] > center+th)   pattern2 = pattern2 | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] > center+th) pattern2 = pattern2 | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] > center+th)   pattern2 = pattern2 | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] > center+th) pattern2 = pattern2 | 0x80;

        vec[offset+pattern1]++;
        offset += 256; 
        vec[offset+pattern2]++;
        offset += 256; 

      }
    }
    if(vec_nDim <= offset) 
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

  return;
}


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_ltp_pyr_features_sparse(uint32_t *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols, uint8_t th )
{
  uint32_t offset, ww, hh, x, y,center,j ;
  uint8_t pattern1;
  uint8_t pattern2;
  uint32_t idx;

  idx = 0;
  offset=0;
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        center = img[LIBLBP_INDEX(y,x,img_nRows)];

        pattern1 = 0;
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center-th) pattern1 = pattern1 | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center-th)   pattern1 = pattern1 | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center-th) pattern1 = pattern1 | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center-th)   pattern1 = pattern1 | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center-th)   pattern1 = pattern1 | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center-th) pattern1 = pattern1 | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center-th)   pattern1 = pattern1 | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center-th) pattern1 = pattern1 | 0x80;

        pattern2 = 0;
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] > center+th) pattern2 = pattern2 | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] > center+th)   pattern2 = pattern2 | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] > center+th) pattern2 = pattern2 | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] > center+th)   pattern2 = pattern2 | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] > center+th)   pattern2 = pattern2 | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] > center+th) pattern2 = pattern2 | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] > center+th)   pattern2 = pattern2 | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] > center+th) pattern2 = pattern2 | 0x80;

        vec[idx++] = offset+pattern1;
        /*        vec[offset+pattern1]++;*/
        offset += 256; 
        /*        vec[offset+pattern2]++;*/
        vec[idx++] = offset+pattern2;
        offset += 256; 

      }
    }
    if(vec_nDim <= idx)
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

  return;
}



/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_uniform_pyr_features(char *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols )
{
  uint32_t offset, ww, hh, x, y,center,j ;
  uint8_t pattern;

  offset=0;
/*  ww=win_W;*/
/*  hh=win_H;*/
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        pattern = 0;
        center = img[LIBLBP_INDEX(y,x,img_nRows)];
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;
        
        vec[offset+kUniformLBP[pattern]]++;
        offset += 59; 

      }
    }
    if(vec_nDim <= offset) 
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

  return;
}


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_pyr_features_sparse(uint32_t* vec, uint32_t vec_nDim, uint32_t* img, uint16_t img_nRows, uint16_t img_nCols)
{
    uint32_t offset, ww, hh, x, y, center, j, idx;
    uint8_t pattern;

    idx = 0;
    offset = 0;
    ww = img_nCols;
    hh = img_nRows;
    while(1)
    {
        for(x = 1; x < ww-1; x++)
        {
            for(y = 1; y< hh-1; y++)
            {
                pattern = 0;
                center = img[LIBLBP_INDEX(y,x,img_nRows)];
                if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
                if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
                if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
                if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
                if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
                if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
                if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
                if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;

                vec[idx++] = offset+pattern;
                offset += 256;
            }
        }
        if(vec_nDim <= idx)
          return;

        if(ww % 2 == 1) ww--;
        if(hh % 2 == 1) hh--;

        ww = ww/2;
        
        for(x=0; x < ww; x++)
          for(j=0; j < hh; j++)
            img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
              img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

        hh = hh/2;
        
        for(y=0; y < hh; y++)
          for(j=0; j < ww; j++)
            img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
              img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    }
    return;
}


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_pyr_features(char *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols )
{
  uint32_t offset, ww, hh, x, y,center,j ;
  uint8_t pattern;

  offset=0;
/*  ww=win_W;*/
/*  hh=win_H;*/
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        pattern = 0;
        center = img[LIBLBP_INDEX(y,x,img_nRows)];
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;

        vec[offset+pattern]++;
        offset += 256; 

      }
    }
    if(vec_nDim <= offset) 
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

  return;
}


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
double liblbp_pyr_dotprod(double *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols)
{
  double dot_prod = 0;
  uint32_t offset=0;
  uint32_t ww, hh, center, x, y, j;
  uint8_t pattern;
  
/*  ww=win_W;*/
/*  hh=win_H;*/
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        pattern = 0;
        center = img[LIBLBP_INDEX(y,x,img_nRows)];
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;

        dot_prod += vec[offset+pattern];
        offset += 256; 


      }
    }
    if(vec_nDim <= offset) 
      return(dot_prod);


    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
                                          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
                                           img[LIBLBP_INDEX(2*y+1,j,img_nRows)];    
  }
 
  
}


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_pyr_addvec(int64_t *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols)
{
  uint32_t offset, ww, hh, x, y, center,j ;
  uint8_t pattern;

  offset=0;
/*  ww=win_W;*/
/*  hh=win_H;*/
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        pattern = 0;
        center = img[LIBLBP_INDEX(y,x,img_nRows)];
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;

        vec[offset+pattern]++;
        offset += 256; 

      }
    }
    if(vec_nDim <= offset) 
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
             img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

  return;
}



/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/
void liblbp_pyr_subvec(int64_t *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols)
{
  uint32_t offset, ww, hh, x, y,center,j ;
  uint8_t pattern;

  offset=0;
/*  ww=win_W;*/
/*  hh=win_H;*/
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        pattern = 0;
        center = img[LIBLBP_INDEX(y,x,img_nRows)];
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;

        vec[offset+pattern]--;
        offset += 256; 

      }
    }
    if(vec_nDim <= offset) 
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

  return;
}


/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/

uint32_t liblbp_pyr_get_dim(uint16_t img_nRows, uint16_t img_nCols, uint16_t nPyramids)
{
  uint32_t w, h, N, i;

  for(w=img_nCols, h=img_nRows, N=0, i=0; i < nPyramids && LIBLBP_MIN(w,h) >= 3; i++)
  {
    N += (w-2)*(h-2);

    if(w % 2) w--;
    if(h % 2) h--;
    w = w/2;
    h = h/2;
  }
  return(256*N);
}

/*-----------------------------------------------------------------------
  -----------------------------------------------------------------------*/

uint32_t liblbp_uniform_pyr_get_dim(uint16_t img_nRows, uint16_t img_nCols, uint16_t nPyramids)
{
  uint32_t w, h, N, i;

  for(w=img_nCols, h=img_nRows, N=0, i=0; i < nPyramids && LIBLBP_MIN(w,h) >= 3; i++)
  {
    N += (w-2)*(h-2);

    if(w % 2) w--;
    if(h % 2) h--;
    w = w/2;
    h = h/2;
  }
  return(59*N);
}
