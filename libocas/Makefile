MEXDETECTED := $(shell mex -h >/dev/null 2>&1 && echo yes)
MEX := mex
MEXFLAGS := $(shell if uname -m | grep -q x86_64 ; then echo -largeArrayDims ; fi)
MEXSUFFIX := $(shell if uname -m | grep -q x86_64 ; then echo mexa64 ; else echo mexglx ; fi)
CC := gcc
#CFLAGS := -g -lm -Wall -pthread
CFLAGS := -msse -O3 -fPIC -fstrict-aliasing -fomit-frame-pointer -Wall -pthread
CLIBS := -lm
#CFLAGS := -lm -msse -O3 -fPIC -fstrict-aliasing -fomit-frame-pointer -Wall
#CFLAGS := -lm -msse -O3 -fPIC -fopenmp -fstrict-aliasing -fomit-frame-pointer -Wall


ifeq (yes,$(MEXDETECTED))
all: svmocas_nnw.$(MEXSUFFIX) svmocas.$(MEXSUFFIX) svmocas_light.$(MEXSUFFIX) linclassif_light.$(MEXSUFFIX) libocas.so svmocas msvmocas linclassif msvmocas.$(MEXSUFFIX) msvmocas_light.$(MEXSUFFIX) compute_auc.$(MEXSUFFIX) svmocas_lbp.$(MEXSUFFIX) svmocas_bool.$(MEXSUFFIX) lbppyr_features.$(MEXSUFFIX) load_svmlight_file.$(MEXSUFFIX)

compute_auc.$(MEXSUFFIX): compute_auc_mex.c libocas.h ocas_helper.h ocas_helper.c version.h
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output compute_auc.$(MEXSUFFIX) compute_auc_mex.c ocas_helper.c lib_svmlight_format.c

svmocas.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h svmocas_mex.c lib_svmlight_format.c lib_svmlight_format.h ocas_helper.c ocas_helper.h features_int8.h features_int8.c features_double.h features_double.c features_single.h features_single.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output svmocas.$(MEXSUFFIX) svmocas_mex.c ocas_helper.c lib_svmlight_format.c features_int8.c features_double.c features_single.c libocas.c libqp_splx.c

svmocas_nnw.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h svmocas_nnw_mex.c lib_svmlight_format.c lib_svmlight_format.h ocas_helper.c ocas_helper.h features_int8.h features_int8.c features_double.h features_double.c features_single.h features_single.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output svmocas_nnw.$(MEXSUFFIX) svmocas_nnw_mex.c lib_svmlight_format.c ocas_helper.c features_int8.c features_double.c features_single.c libocas.c libqp_splx.c

svmocas_light.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h svmocas_light_mex.c lib_svmlight_format.c lib_svmlight_format.h ocas_helper.c ocas_helper.h features_double.h features_double.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output svmocas_light.$(MEXSUFFIX) svmocas_light_mex.c lib_svmlight_format.c ocas_helper.c features_double.c libocas.c libqp_splx.c
		
linclassif_light.$(MEXSUFFIX): linclassif_light_mex.c lib_svmlight_format.c lib_svmlight_format.h ocas_helper.c ocas_helper.h
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output linclassif_light.$(MEXSUFFIX) linclassif_light_mex.c lib_svmlight_format.c ocas_helper.c

msvmocas.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h msvmocas_mex.c lib_svmlight_format.c lib_svmlight_format.h ocas_helper.c ocas_helper.h features_double.h features_double.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output msvmocas.$(MEXSUFFIX) msvmocas_mex.c lib_svmlight_format.c ocas_helper.c features_double.c libocas.c libqp_splx.c

msvmocas_light.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h msvmocas_light_mex.c lib_svmlight_format.c lib_svmlight_format.h ocas_helper.c ocas_helper.h features_double.h features_double.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output msvmocas_light.$(MEXSUFFIX) msvmocas_light_mex.c lib_svmlight_format.c ocas_helper.c features_double.c libocas.c libqp_splx.c
		
lbppyr.$(MEXSUFFIX): lbppyr_mex.c
		$(MEX) $(MEXFLAGS) -O -output lbppyr.$(MEXSUFFIX) lbppyr_mex.c 	

svmocas_lbp.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h svmocas_lbp_mex.c ocas_lbp_helper.c ocas_lbp_helper.h liblbp.h liblbp.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output svmocas_lbp.$(MEXSUFFIX) svmocas_lbp_mex.c ocas_lbp_helper.c  libocas.c libqp_splx.c liblbp.c

svmocas_bool.$(MEXSUFFIX): libocas.c libocas.h libqp_splx.c libqp.h svmocas_bool_mex.c features_bool.c features_bool.h liblbp.h liblbp.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output svmocas_bool.$(MEXSUFFIX) svmocas_bool_mex.c features_bool.c ocas_helper.c lib_svmlight_format.c libocas.c libqp_splx.c 

lbppyr_features.$(MEXSUFFIX): lbppyr_features_mex.c liblbp.c liblbp.h
		$(MEX) -g $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output lbppyr_features.$(MEXSUFFIX) lbppyr_features_mex.c liblbp.c

lbpfilter.$(MEXSUFFIX): lbpfilter_mex.c
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output lbpfilter.$(MEXSUFFIX) lbpfilter_mex.c

load_svmlight_file.$(MEXSUFFIX): load_svmlight_file.c lib_svmlight_format.c lib_svmlight_format.h
		$(MEX) $(MEXFLAGS) -DLIBOCAS_MATLAB -O -output load_svmlight_file.$(MEXSUFFIX) load_svmlight_file.c lib_svmlight_format.c

else 
all: libocas.so svmocas msvmocas linclassif
endif

svmocas:	svmocas.c svmocas.c lib_svmlight_format.c sparse_mat.c ocas_helper.c ocas_helper.h libocas.h sparse_mat.h libocas.c features_double.h features_double.c version.h
		$(CC) $(CFLAGS) -o $@ svmocas.c lib_svmlight_format.c sparse_mat.c ocas_helper.c features_double.c libocas.c libqp_splx.c $(CLIBS)

msvmocas:	msvmocas.c lib_svmlight_format.c sparse_mat.c ocas_helper.c ocas_helper.h libocas.h sparse_mat.h libocas.c features_double.h features_double.c version.h
		$(CC) $(CFLAGS) -o $@ msvmocas.c lib_svmlight_format.c sparse_mat.c ocas_helper.c features_double.c libocas.c libqp_splx.c $(CLIBS)

linclassif:	linclassif.c lib_svmlight_format.c libocas.h version.h
		$(CC) $(CFLAGS) -o $@ linclassif.c lib_svmlight_format.c $(CLIBS)


libocas.so:	libocas.c libocas.h libqp_splx.c libqp.h
		$(CC) $(CFLAGS) -shared -o $@ libocas.c libqp_splx.c $(CLIBS)

clean: 
		rm -f *~ svmocas.$(MEXSUFFIX) svmocas_nnw.$(MEXSUFFIX) svmocas_bool.$(MEXSUFFIX) svmocas_light.$(MEXSUFFIX) linclassif_light.$(MEXSUFFIX) svmocas msvmocas linclassif libocas.so msvmocas.$(MEXSUFFIX) msvmocas_light.$(MEXSUFFIX) lbppyr_features.$(MEXSUFFIX) svmocas_lbp.$(MEXSUFFIX) compute_auc.$(MEXSUFFIX) load_svmlight_file.$(MEXSUFFIX)
