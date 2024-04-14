# ece569proj
ECE 569 final project

mkdir build
cp -r testpics build

$module load cuda11/11.0
$CC=gcc cmake3 ../.
make

example  ./shadowRemoval_Solution plt4.jpg CI.jpg Grayscale.jpg UComponent.jpg

srun run_finalproj.slurm