#!/bin/bash
#

# Request Bash shell as shell for job
#$ -S /bin/bash

# Execute the job from the current working directory.
#$ -cwd

# Defines  or  redefines  the  path used for the standard error stream of the job.
#$ -e ./jobsLogs/

# The path used for the standard output stream of the job.
#$ -o ./jobsLogs/

# Select offline queue.
#$ -q offline

date
$1
date
