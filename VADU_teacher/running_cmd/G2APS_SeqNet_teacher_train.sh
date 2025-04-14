	CONFIG=configs/cuhk_sysu.yaml
	DATA_ROOT=/home/patrick/wqf/Data/G2APS/G2APS_in_market1501_style/Market-1501-v15.09.15/
	OUTPUT_DIR=/home/patrick/wqf/SeqNet-master/logs/teacher_seqnet/
	RESULT_LOG=/home/patrick/wqf/SeqNet-master/logs/teacher_seqnet/result.txt
	cd ..; CUDA_VISIBLE_DEVICES=1  nohup python train.py --cfg $CONFIG INPUT.BATCH_SIZE_TRAIN 3 INPUT.NUM_WORKERS_TRAIN 3 INPUT.MIN_SIZE [600,800,900] MODEL.LOSS.LUT_SIZE 2078 MODEL.LOSS.CQ_SIZE 2000 \
	SOLVER.MAX_EPOCHS 300 SOLVER.BASE_LR 0.0036 eval_interval 10  SOLVER.LR_DECAY_MILESTONES [80,150] OUTPUT_DIR $OUTPUT_DIR \
	duke_path $DATA_ROOT > $RESULT_LOG 2>&1 &