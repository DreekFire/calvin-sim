# export DIFFUSION_MODEL_CHECKPOINT=<path_to_checkpoint>
# export JAXRL_M_POLICY_CHECKPOINT=gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407/checkpoint_2000000000
export NUM_SEQUENCES=50
export TFHUB_CACHE_DIR=~/tf_models
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# name="ddpmiql_ABC_D10_all_mixing0.9_expectile0.7_res18_repeat5_20240114_073709"
# name="ddpmiql_ABC_D10_forward_mixing0.9_expectile0.7_res18_repeat5_20240114_073855"
name="ddpmiql_ABC_D10_only_real_mixing0.9_expectile0.7_res18_repeat5_20240114_073837"

python calvin_models/calvin_agent/evaluation/evaluate_policy_diffusion_lc.py --dataset_path "/home/mitsuhiko/calvin-sim/mini_dataset" --custom_model \
--checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/CALVIN_IQL_fixed/$name/checkpoint_250000" \
--wandb_run_name "mitsuhiko/CALVIN_IQL_fixed/$name" \
--agent "ddpmlc"


# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_forward_mixing0.9_act1_20240107_211457/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_forward_mixing0.9_act1_20240107_211457"

# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407"

# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_20231227_191722/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_20231227_191722"

# --checkpoint_path "/nfs/kun2/users/pranav/checkpoints/diffusion_policy_checkpoints/lcbc/checkpoint_188000"  \
# --wandb_run_name "pranavatreya/jaxrl_m_calvin_lcbc/lcbc_diffusion_policy_20231014_164920"