#!/bin/bash
#

#
# qstat
# qhost
# qdel
# qdel -r conftrain
#

for run in {1..20}
do
    qsub -N conftrain sge_superfastjobs_job.sh "/usr/local/matlab90/bin/matlab -nodisplay -nosplash -singleCompThread -r run_all_train_conf"
done
