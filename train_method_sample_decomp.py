import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data
from datasets.build_mix_view import build_train_loader, build_test_loader    #加载混合数据集
from datasets.build_single_view import build_train_loader_aerial, build_train_loader_ground, build_test_loader_aerial, build_test_loader_ground, build_dataset  #加载单一视角数据集
from defaults import get_default_cfg
from engine import train_one_epoch_sample_decomp, evaluate_performance_aerial, evaluate_performance_ground, evaluate_performance_Re_ID
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed

# baseline
#from models.seqnet_baseline import SeqNet
#from models.seqnet_baseline_test_time import SeqNet

# solution1
#from models.seqnet_share_backbone import Seqnet                                 # share backbone, sample decomp, solution1
#from models.seqnet_share_backbone_reid_decomp_2_mencontrast import SeqNet       # share backbone, sample decomp, three re-id loss and two memcontrast loss
#from models.seqnet_share_backbone_two_box_predictor_in_reid_res5 import Seqnet  # share backbone, sample decomp, two box predictor in reid_res5

# solution2, (include our VADU method)
#from models.seqnet_model_decomp import SeqNet                                      # two backbone, two detection module, one re-ID loss
#from models.seqnet_model_decomp_mix_dataset_input import SeqNet                    # two backbone, two detection module, one re-ID loss, mix dataset input
#from models.seqnet_model_decomp_pair_dataset_input import SeqNet                   # two backbone, two detection module, one re-ID loss, pair dataset input
#from models.seqnet_model_decomp_reid_decomp import SeqNet                          # two backbone, two detection module, three re-ID loss
#from models.seqnet_model_decomp_reid_decomp_mix_dataset_input import SeqNet        # two backbone, two detection module, three re-ID loss, mix dataset input
#from models.seqnet_model_decomp_reid_decomp_triplet_loss import SeqNet             # two backbone, two detection module, three re-ID loss and one triplet loss
from models.seqnet_model_decomp_reid_decomp_2_memcontrast import SeqNet            # two backbone, two detection module, three re-ID loss and two mencontrast loss, our VADU method
#from models.seqnet_model_decomp_reid_decomp_2_mencontrast_test_time import Seqnet  # two backbone, two detection module, three re-ID loss and two mencontrast loss, our VADU method
#from models.seqnet_model_decomp_reid_decomp_a2gcontrast import SeqNet              # two backbone, two detection module, three re-ID loss and one a2gmencontrast loss
#from models.seqnet_model_decomp_reid_decomp_g2acontrast import SeqNet              # two backbone, two detection module, three re-ID loss and one g2amencontrast loss

# soution3
#from models.solution3 import SeqNet 

'''
def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)

    print("Loading data")
    train_loader = build_train_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")
'''

def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)

    print("Loading gallery and query data")
    # load地面数据集和空中数据集
    #单一数据集的gallery_loader, query_loader
    gallery_loader_aerial, query_loader_aerial = build_test_loader_aerial(cfg)
    gallery_loader_ground, query_loader_ground = build_test_loader_ground(cfg)
    # load混合数据集用于测试Re-ID性能
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        #循环评估所有epoch的模型性能
        #for epoch_num in range(9, 18):
            # 动态生成模型路径
            #ckpt_path = osp.join(args.ckpt, f"epoch_{epoch_num}.pth")
        
            #加载当前的 epoch 模型
            #resume_from_ckpt(ckpt_path, model)
        evaluate_performance_aerial(       
                model,
                device,
                gallery_loader_aerial,
                query_loader_aerial,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        evaluate_performance_ground(      
                    model,
                    device,
                    gallery_loader_ground,
                    query_loader_ground,
                    use_gt=cfg.EVAL_USE_GT,
                    use_cache=cfg.EVAL_USE_CACHE,
                    use_cbgm=cfg.EVAL_USE_CBGM,
            )

        evaluate_performance_Re_ID(      
                    model,
                    gallery_loader,
                    query_loader,
                    device,
                    use_gt=cfg.EVAL_USE_GT,
                    #use_gt=True,
                    use_cache=cfg.EVAL_USE_CACHE,
                    use_cbgm=cfg.EVAL_USE_CBGM,
                )
        exit(0)
    
    '''
    # optimizer参数分组
    bk_params = []
    head_params = []
    for n,p in model.named_parameters():
        if n.startswith("backbone"):
            bk_params.append(p)
        else:
            head_params.append(p)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        [
            {'params': head_params, 'lr': cfg.SOLVER.BASE_LR},
            {'params': bk_params, 'lr': cfg.SOLVER.BASE_LR*0.5}
        ],
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    '''
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    #初始化最佳性能指标
    best_performance = -1.0
    best_epoch = -1
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        print("Loading training data")
        # load地面训练数据集和空中训练数据集，每个epoch重新调用数据集类和dataloader
        train_loader_aerial = build_train_loader_aerial(cfg)
        train_loader_ground = build_train_loader_ground(cfg)
    
        train_one_epoch_sample_decomp(cfg, model, optimizer, train_loader_aerial, train_loader_ground, device, epoch, tfboard)
        lr_scheduler.step()
        #先保存，再评估
        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )
        
        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            '''
            evaluate_performance_aerial(       
                model,
                device,
                gallery_loader_aerial,
                query_loader_aerial,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

            evaluate_performance_ground(      
                    model,
                    device,
                    gallery_loader_ground,
                    query_loader_ground,
                    use_gt=cfg.EVAL_USE_GT,
                    use_cache=cfg.EVAL_USE_CACHE,
                    use_cbgm=cfg.EVAL_USE_CBGM,
            )
            '''
            #先只评估混合检测和Re-ID
            evaluate_performance_Re_ID(      
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                #use_gt=True,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )
            

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
