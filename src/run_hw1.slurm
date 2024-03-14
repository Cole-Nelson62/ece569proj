#!/bin/bash

### Set the job name
#SBATCH --job-name=devicequery

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=standard

#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:00:05

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules
###module load openmpi3
###module load cuda11/11.0


### set directory for job execution, ~netid = home directory path
cd ~akoglu/class/ece569/build_dir

###
###setenv MPI_DSM_DISTRIBUTE

### run your executable program with begin and end date and time output
date
###/usr/bin/time mpirun -np 1 
./DeviceQuery_Solution > devicequery_output.txt
date
