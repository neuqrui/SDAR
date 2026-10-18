# ALFWorld 7B SFT step-skill, step 180

Run: `examples/opsd_kl_trainer/run_alfworld_7b_sft_step_skill.sh`

Experiment: `grpo_opsd_kl_qwen2.5_7b_sft_coef0.01_sampled_kl_gatetanh_c1_step_skill_none_length_pairwise`

Training finished 180/180. Checkpoint: `global_step_180`.

Final validation:

| metric | value |
|---|---|
| val/success_rate | 0.96875 |
| val/text/test_score | 8.244 |
| val/response_steps/mean | 8.898 |
| val/pick_and_place_success_rate | 1.000 |
| val/pick_clean_then_place_in_recep_success_rate | 1.000 |
| val/pick_heat_then_place_in_recep_success_rate | 1.000 |
| val/pick_cool_then_place_in_recep_success_rate | 1.000 |
| val/pick_two_obj_and_place_success_rate | 0.920 |
| val/look_at_obj_in_light_success_rate | 0.857 |

Same-step train episode success rate: 0.875.

SwanLab: https://swanlab.cn/@67r/verl_agent_alfworld/runs/xe8d0njy
