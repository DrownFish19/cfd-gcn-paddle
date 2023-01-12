export BATCH_SIZE=16

export SU2_RUN="/root/autodl-tmp/cfd-gcn-paddle/SU2"
export SU2_HOME="/root/autodl-tmp/SU2"
export PATH=$PATH:$SU2_RUN
export PYTHONPATH=$PYTHONPATH:$SU2_RUN

# Prediction experiments

mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --su2-config config/coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-4 --data-dir data/NACA0012_interpolate --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 -e cfd_gcn_interp > /dev/null

mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-4 --data-dir data/NACA0012_interpolate/ -e gcn_interp

# Generalization experiments

mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --su2-config config/coarse.cfg --model cfd_gcn --hidden-size 512 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-4 --data-dir data/NACA0012_machsplit_noshock --coarse-mesh meshes/mesh_NACA0012_xcoarse.su2 -e cfd_gcn_gen > /dev/null

mpirun -np $((BATCH_SIZE+1)) --oversubscribe python main.py --batch-size $BATCH_SIZE --gpus 1 -dw 1 --model gcn --hidden-size 512 --num-layers 6 --optim adam -lr 5e-4 --data-dir data/NACA0012_machsplit_noshock/ -e gcn_gen