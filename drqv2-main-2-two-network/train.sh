export MUJOCO_GL=osmesa
export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=1, python train.py task=acrobot_swingup

#walker_run