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
folder_path=~nelso680/ece569/finalproj
hw_path=$folder_path/build
testpics_path=$folder_path/testpics

cd $hw_path 
pwd

mkdir output
mkdir output/img1
mkdir output/img2
mkdir output/img3

im1_path=$hw_path/output/img1
im2_path=$hw_path/output/img3
im3_path=$hw_path/output/img3
###
###setenv MPI_DSM_DISTRIBUTE

### run your executable program with begin and end date and time output
date
###/usr/bin/time mpirun -np 1 
./shadowRemoval_Solution $testpics_path/plt4.jpg $im1_path/CI.jpg $im1_path/Grayscale.jpg $im1_path/UComponent.jpg $im1_path/ConvolutionOut.jpg
date
