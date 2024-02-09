# export DIFFUSION_MODEL_CHECKPOINT=<path_to_checkpoint>
# export JAXRL_M_POLICY_CHECKPOINT=gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407/checkpoint_2000000000
export NUM_SEQUENCES=100
export TFHUB_CACHE_DIR=~/tf_models
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# name="lciql_ABC_D10_all_res18__expectile0.7_20240114_073713"
# name="lciql_ABC_D10_forward_res18__expectile0.7_20240114_074055"
# name="lciql_ABC_D10_only_real_res18__expectile0.7_20240114_074037"

# name="lciql_ABC_D10_forward_res18__expectile0.7_repeat1_posreward_20240115_011517"
# name="lciql_ABC_D10_only_real_res18__expectile0.7_repeat1_posreward_20240115_011614"

# bucket="rail-tpus-mitsuhiko"
bucket="rail-tpus-mitsuhiko-central2"

entity="mitsuhiko"
project="230126_PTR_calvin"
# name="lcbc_ABCD_b256_tanhFalse_learnedstd_20240117_224749"
name="IQL_Video_ABC_DTargetChain_neg1_mixing0.98_2024_02_07_16_04_50_0000--s-42"
# name="CQL_alpha5_rescaleFalse_2024_01_28_17_04_01_0000--s-42"

python calvin_models/calvin_agent/evaluation/evaluate_policy_jaxrl2.py \
--dataset_path "/home/mitsuhiko/calvin-sim/mini_dataset" --custom_model \
--checkpoint_path "gs://$bucket/logs/ptr_exp/$project/$name/checkpoint150000" \
--wandb_run_name "$entity/$project/$name" \
--agent "lc" \
--single_chain 1


# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_forward_mixing0.9_act1_20240107_211457/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_forward_mixing0.9_act1_20240107_211457"

# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_ABC_D10_only_real_mixing0.9_act1_20240107_210407"

# --checkpoint_path "gs://rail-tpus-mitsuhiko-central2/logs/jaxrl_m_calvin_lcbc/lcbc_20231227_191722/checkpoint_200000" \
# --wandb_run_name "mitsuhiko/jaxrl_m_calvin_lcbc/lcbc_20231227_191722"

# --checkpoint_path "/nfs/kun2/users/pranav/checkpoints/diffusion_policy_checkpoints/lcbc/checkpoint_188000"  \
# --wandb_run_name "pranavatreya/jaxrl_m_calvin_lcbc/lcbc_diffusion_policy_20231014_164920"