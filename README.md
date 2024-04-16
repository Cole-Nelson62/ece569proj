# ece569proj
ECE 569 final project

From the home folder of the ECE569 final project folder run the following commands
mkdir build
mv run_final.slurm build/.

To build the program run the following and from the home directory of the project folder cmakelist will be in mail

cd build
module load cuda11/11.0
CC=gcc cmake3 ../.
make

below is a example on how to run the command manually but it is meant to run the command with the slurm underneath
example  ./shadowRemoval_Solution plt4.jpg CI.jpg Grayscale.jpg UComponent.jpg

To run the slurm edit the folder path variable to whatever project folder directory is being used. example below
folder_path=~nelso680/ece569/finalproj

from the build directory run the line below
srun run_finalproj.slurm