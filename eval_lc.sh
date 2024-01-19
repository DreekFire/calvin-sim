# export DIFFUSION_MODEL_CHECKPOINT=<path_to_checkpoint>
# export JAXRL_M_POLICY_CHECKPOINT=gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407/checkpoint_2000000000
export NUM_SEQUENCES=100
export TFHUB_CACHE_DIR=~/tf_models
export CUDA_VISIBLE_DEVICES=5
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# name="lciql_ABC_D10_all_res18__expectile0.7_20240114_073713"
# name="lciql_ABC_D10_forward_res18__expectile0.7_20240114_074055"
# name="lciql_ABC_D10_only_real_res18__expectile0.7_20240114_074037"

# name="lciql_ABC_D10_forward_res18__expectile0.7_repeat1_posreward_20240115_011517"
# name="lciql_ABC_D10_only_real_res18__expectile0.7_repeat1_posreward_20240115_011614"

project="CALVIN_IQL_fixed_encoder_and_goals"
# name="lcbc_ABCD_b256_tanhFalse_learnedstd_20240117_224749"
name="lciql_ABCD_expectile0.7_temperature1_b256_fixed_discount0.98_tanhFalse_shared_encoder_20240118_232422"
python calvin_models/calvin_agent/evaluation/evaluate_policy_diffusion_lc.py --dataset_path "/home/mitsuhiko/calvin-sim/mini_dataset" --custom_model \
--checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/$project/$name/checkpoint_150000" \
--wandb_run_name "mitsuhiko/$project/$name" \
--agent "lc"


# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_forward_mixing0.9_act1_20240107_211457/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_forward_mixing0.9_act1_20240107_211457"

# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407"

# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_20231227_191722/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_20231227_191722"

# --checkpoint_path "/nfs/kun2/users/pranav/checkpoints/diffusion_policy_checkpoints/lcbc/checkpoint_188000"  \
# --wandb_run_name "pranavatreya/jaxrl_m_calvin_lcbc/lcbc_diffusion_policy_20231014_164920"