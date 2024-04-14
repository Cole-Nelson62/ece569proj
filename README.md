# ece569proj
ECE 569 final project

mkdir build
cp -r testpics build

$module load cuda11/11.0
$CC=gcc cmake3 ../.

srun run_finalproj.slurm