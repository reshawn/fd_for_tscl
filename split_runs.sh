#!/bin/bash
echo "starting"
BASE_COMMAND="python run.py --forms o fd"
START_NUM=0
INCREMENT_STEP=3
NUM_PER_RUN=2
END_NUM=90

for i in $(seq $START_NUM $INCREMENT_STEP $END_NUM); do
  echo "$i"
  FULL_COMMAND="$BASE_COMMAND -s $i -e $(($i+$NUM_PER_RUN))"
  echo ""
  echo "-> Running: $FULL_COMMAND (Sequence index: $i)"
  $FULL_COMMAND
  echo "------------------------------------"
done