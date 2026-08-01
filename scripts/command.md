cd /vepfs_hyh/hyh/FastSD

bash scripts/run_vanilla_profile.sh vanilla vanilla_run --num_drafts 4 --dataset humaneval --max_tokens 256
bash scripts/run_vanilla_profile.sh proactive_only proactive_run --num_drafts 4 --dataset humaneval --max_tokens 256
bash scripts/run_vanilla_profile.sh pipeline_only pipeline_run --num_drafts 4 --dataset humaneval --max_tokens 256
bash scripts/run_vanilla_profile.sh both both_run --num_drafts 4 --dataset humaneval --max_tokens 256

bash scripts/run_fastsd_profile.sh fastsd --num_drafts 4 --dataset humaneval --max_tokens 256

bash scripts/run_fastsd_profile.sh fastsd