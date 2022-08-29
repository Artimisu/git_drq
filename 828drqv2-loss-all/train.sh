export MUJOCO_GL=osmesa
export HYDRA_FULL_ERROR=1
task_list=(humanoid_stand humanoid_run humanoid_walk)
alpha_list="5.0 15.0"

for alpha in $alpha_list

do
    rlaunch --charged-group research_model --preemptible=no  --cpu=16 --gpu=1 --memory=32000 -- /home/huyang02/anaconda3/envs/gfe/bin/python train.py \
        task=${task_list[0]} alpha=$alpha &

    rlaunch --charged-group research_model --preemptible=no  --cpu=16 --gpu=1 --memory=32000 -- /home/huyang02/anaconda3/envs/gfe/bin/python train.py \
        task=${task_list[1]} alpha=$alpha &

    rlaunch --charged-group research_model --preemptible=no  --cpu=16 --gpu=1 --memory=32000 -- /home/huyang02/anaconda3/envs/gfe/bin/python train.py \
        task=${task_list[2]} alpha=$alpha &

done  
    

# rlaunch --charged-group research_model --preemptible=no  --cpu=16 --gpu=1 --memory=32000 -- /home/huyang02/anaconda3/envs/gfe/bin/python train.py \
# task=${task_list[0]} alpha=5 &
