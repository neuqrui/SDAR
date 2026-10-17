set -x


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

# Keep this short: Ray AF_UNIX sockets must stay under 107 bytes.
export RAY_TMPDIR=/data/qirui/SDAR/tmp/ray_af_7b_sft
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

ENGINE=${1:-vllm}
export CUDA_VISIBLE_DEVICES=4,5,2,3
num_cpus_per_env_worker=0.1
max_steps=50
# LENGTH_REWARD=none     : episode reward stays 10 or 0.
# LENGTH_REWARD=adv      : task-relative length advantage after GRPO, coef 1. Episode reward stays 10 or 0.
# LENGTH_REWARD=bonus    : add 5 * (max_steps - steps) / max_steps on successes only.
# LENGTH_REWARD=pairwise : soft pairwise efficiency, tau=0.1. Score is 10+5*b on successes, then GRPO.
LENGTH_REWARD=${LENGTH_REWARD:-pairwise}
case "$LENGTH_REWARD" in
    none)
        success_length_adv_coef=0
        success_length_bonus_coef=0
        success_length_pairwise_tau=0
        ;;
    adv)
        success_length_adv_coef=1.0
        success_length_bonus_coef=0
        success_length_pairwise_tau=0
        ;;
    bonus)
        success_length_adv_coef=0
        success_length_bonus_coef=5.0
        success_length_pairwise_tau=0
        ;;
    pairwise)
        success_length_adv_coef=0
        success_length_bonus_coef=0
        success_length_pairwise_tau=0.1
        ;;
    *)
        echo "LENGTH_REWARD must be none, adv, bonus, or pairwise, got ${LENGTH_REWARD}" >&2
        exit 1
        ;;
esac

# GRPO on every step. OPSD coverage follows skill_source.
# ana_loss_mode=none: the student writes the skill for OPSD and is not updated on it.
# No judge and no external analyzer API. The skill comes from policy_vllm.
# L = L_GRPO + opsd_coef * L_OPSD(marked steps) + kl_coef * L_KL
opsd_coef=0.01
ana_pg_coef=0
ana_loss_mode=${ANA_LOSS_MODE:-none}
if [[ "$ana_loss_mode" != "none" ]]; then
    echo "This script has no judge or analyzer API. ANA_LOSS_MODE must stay none, got ${ana_loss_mode}" >&2
    exit 1
fi
opsd_objective=sampled_kl
# gap = stopgrad(log pi_T - log pi_S). OPSD_GATE selects the bound:
#   sdar / sdar-gate: L = sigmoid(beta * gap) * (log pi_T - log pi_S), gradient -g
#   clip:             L = -clip(gap, -c, c) * log pi_S
#   tanh:             L = -(c * tanh(beta * gap / c)) * log pi_S
# c is OPSD_ADV_CLIP. Default tanh uses beta=2.5, c=1, so a = tanh(2.5 * gap) in (-1, 1).
opsd_gate=${OPSD_GATE:-tanh}
opsd_gate_beta=${OPSD_GATE_BETA:-2.5}
opsd_adv_clip=${OPSD_ADV_CLIP:-1}
skill_all=false
# step_skill: one skill shared by each target_steps group, OPSD only on those steps.
skill_source=${SKILL_SOURCE:-step_skill}
ana_backend=policy_vllm
# Student responses stay at 512, so the analyzer completion is capped there too.
ana_max_prompt_length=28672
ana_max_completion_tokens=512
ana_max_model_len=$((ana_max_prompt_length + ana_max_completion_tokens))

train_data_size=16
val_data_size=128
group_size=8
total_epochs=180
# 3-epoch ALFWorld analyzer SFT. Do not inherit .env MODEL_PATH (Qwen2.5-3B).
model_path="${SFT_MODEL_PATH:-/home/liuqirui/SDAR/checkpoints/sft/alfworld_analyzer_qwen2.5_7b/global_step_261}"
if [[ ! -f "$model_path/config.json" ]]; then
    echo "ALFWorld 7B SFT checkpoint not found: $model_path" >&2
    exit 1
fi
case "$opsd_gate" in
    sdar|sdar-gate|sdar_gate)
        opsd_gate_tag="_gatesdar_beta${opsd_gate_beta}"
        ;;
    clip|tanh)
        opsd_gate_tag="_gate${opsd_gate}_c${opsd_adv_clip}"
        ;;
    none)
        opsd_gate_tag=""
        ;;
    *)
        echo "OPSD_GATE must be sdar-gate, clip, or tanh (got ${opsd_gate})" >&2
        exit 1
        ;;
esac
experiment_name="grpo_opsd_kl_qwen2.5_7b_sft_coef${opsd_coef}_${opsd_objective}${opsd_gate_tag}_${skill_source}_${ana_loss_mode}_length_${LENGTH_REWARD}"
# wandb replaced by swanlab; see .env SWANLAB_API_KEY

python3 -m examples.data_preprocess.prepare \
    --local_dir "$DATA_ROOT/verl-agent/" \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_opsd_kl \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_ROOT/verl-agent/text/train.parquet \
    data.val_files=$DATA_ROOT/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=$ana_max_model_len \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.opsd_kl.opsd_coef=$opsd_coef \
    algorithm.opsd_kl.objective=$opsd_objective \
    algorithm.opsd_kl.gate=$opsd_gate \
    algorithm.opsd_kl.gate_beta=$opsd_gate_beta \
    algorithm.opsd_kl.advantage_clip=$opsd_adv_clip \
    +algorithm.opsd_kl.skills_dir=skills/alfworld \
    +algorithm.opsd_kl.skill_all=$skill_all \
    +algorithm.opsd_kl.skill_source=$skill_source \
    +algorithm.opsd_kl.ana_backend=$ana_backend \
    +algorithm.opsd_kl.ana_max_prompt_length=$ana_max_prompt_length \
    +algorithm.opsd_kl.ana_max_completion_tokens=$ana_max_completion_tokens \
    +algorithm.opsd_kl.ana_pg_coef=$ana_pg_coef \
    +algorithm.opsd_kl.ana_loss_mode=$ana_loss_mode \
    +algorithm.opsd_kl.ana_alpha_pos=1 \
    +algorithm.opsd_kl.ana_alpha_neg=1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$max_steps \
    reward_model.success_length_bonus_coef=$success_length_bonus_coef \
    reward_model.success_length_adv_coef=$success_length_adv_coef \
    reward_model.success_length_pairwise_tau=$success_length_pairwise_tau \
    reward_model.success_length_max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=4 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=$CKPT_ROOT/rollouts/${experiment_name} \
    trainer.validation_data_dir=$CKPT_ROOT/rollouts_val/${experiment_name} \
    trainer.trajectory_dump.freq=1 \
    trainer.trajectory_dump.max_episodes=2 \
    trainer.trajectory_dump.max_teacher_examples=2 \
    trainer.trajectory_dump.write_markdown=true \
    trainer.trajectory_dump.write_html=true $@
