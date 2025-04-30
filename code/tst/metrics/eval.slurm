#!/bin/bash
#SBATCH --job-name=n-eval-vi
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=nmes_cpu,cpu


source /scratch_tmp/users/k21157437/aid_env/bin/activate
which python 


LANG="vi"
if [ "$LANG" = "en" ]; then
   #SUBSET="default"
   SUBSET="paras"
elif [ "$LANG" = "pt" ]; then
   SUBSET="default"
elif [ "$LANG" = "vi" ]; then
   SUBSET="mpairs"
fi
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3" "few4" "few5")

python eval.py --lang $LANG --subset $SUBSET --prompt_techs "${PROMPT_TECHNIQUES[@]}"
