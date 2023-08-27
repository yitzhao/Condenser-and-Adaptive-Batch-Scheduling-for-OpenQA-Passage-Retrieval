PRETRAIN=Luyu/co-condenser-marco
TRAIN_SIZE=100000
DEV_SIZE=1500
EPOCH=5
EVAL_STEP=500
SAVE_STEP=500

accelerate config # Only required in first time

accelerate launch train_abs.py \
  --mix \
  --pretrain $PRETRAIN \
  --sampler abs-beamsearch \
  --epoch $EPOCH \
  --train_size $TRAIN_SIZE \
  --dev_size $DEV_SIZE \
  --eval_step $EVAL_STEP \
  --save_step $SAVE_STEP \
  --output ABS-${PRETRAIN}-Beamsearch-Mix-${TRAIN_SIZE}-${DEV_SIZE}-epoch-${EPOCH}


accelerate launch train_abs.py \
  --mix \
  --pretrain $PRETRAIN \
  --sampler abs-encode \
  --epoch $EPOCH \
  --train_size $TRAIN_SIZE \
  --dev_size $DEV_SIZE \
  --eval_step $EVAL_STEP \
  --save_step $SAVE_STEP \
  --output ABS-${PRETRAIN}-Encode-Mix-${TRAIN_SIZE}-${DEV_SIZE}-epoch-${EPOCH}


accelerate launch train_abs.py \
  --pretrain $PRETRAIN \
  --sampler abs-bm25 \
  --epoch $EPOCH \
  --train_size $TRAIN_SIZE \
  --dev_size $DEV_SIZE \
  --eval_step $EVAL_STEP \
  --save_step $SAVE_STEP \
  --output ABS-${PRETRAIN}-BM25-${TRAIN_SIZE}-${DEV_SIZE}-epoch-${EPOCH}


accelerate launch train_abs.py \
  --pretrain $PRETRAIN \
  --sampler random \
  --epoch $EPOCH \
  --train_size $TRAIN_SIZE \
  --dev_size $DEV_SIZE \
  --eval_step $EVAL_STEP \
  --save_step $SAVE_STEP \
  --output Non-ABS-${PRETRAIN}-Random-${TRAIN_SIZE}-${DEV_SIZE}-epoch-${EPOCH}

accelerate launch train_abs.py \
  --pretrain $PRETRAIN \
  --sampler sequential \
  --epoch $EPOCH \
  --train_size $TRAIN_SIZE \
  --dev_size $DEV_SIZE \
  --eval_step $EVAL_STEP \
  --save_step $SAVE_STEP \
  --output Non-ABS-${PRETRAIN}-Sequential-${TRAIN_SIZE}-${DEV_SIZE}-epoch-${EPOCH}
