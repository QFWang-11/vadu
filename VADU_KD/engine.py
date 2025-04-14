import math
import sys
from copy import deepcopy

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import random

from eval_func import eval_detection, eval_search_cuhk, eval_search_prw
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler


def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


def train_one_epoch_0(cfg, model, optimizer, data_loader, device, epoch, tfboard=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, cfg.DISP_PERIOD, header)
    ):
        images, targets = to_device(images, targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tfboard:
            iter = epoch * len(data_loader) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)

def train_one_epoch(cfg, model, optimizer, data_loader_aerial, data_loader_ground, device, epoch, tfboard=None):
    # Combine the two data loaders using zip
    combined_loader = zip(data_loader_aerial, data_loader_ground)
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader_aerial) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # Calculate the total length of the combined data loaders
    total_length = min(len(data_loader_aerial), len(data_loader_ground))

    # Loop over batches from both data loaders using zip
    for i, ((images_aerial, targets_aerial), (images_ground, targets_ground)) in enumerate(
        metric_logger.log_every(zip(data_loader_aerial, data_loader_ground), cfg.DISP_PERIOD, header, total_length)
    ):
        #print('targets_aerial:', type(targets_aerial))
        #print('images_aerials:', type(images_aerial))
        # Move data to the device
        images_aerial, targets_aerial = to_device(images_aerial, targets_aerial, device)
        images_ground, targets_ground = to_device(images_ground, targets_ground, device)
        # Forward pass through the model with both aerial and ground data
        loss_dict = model(images_aerial, images_ground, targets_aerial, targets_ground)  #这个loss_dict是一个字典，包含了所有的loss
        losses = sum(loss for loss in loss_dict.values())   # Total loss

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        # Check if the loss is finite
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        # Backpropagation
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        # Update the optimizer
        optimizer.step()
        # Step the warmup scheduler if it's the first epoch
        if epoch == 0:
            warmup_scheduler.step()
        # Log the metrics
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # Log the metrics to TensorBoard
        if tfboard:
            iter = epoch * len(data_loader_aerial) + i    # Total iterations across both datasets
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)


def train_one_epoch_method_3(cfg, model, optimizer, train_loader, device, epoch, tfboard=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(train_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # Calculate the total length of the combined data loaders
    total_length = len(train_loader)

    # Loop over batches from both data loaders using zip
    for i, [images, targets] in enumerate(
        metric_logger.log_every(train_loader, cfg.DISP_PERIOD, header, total_length)
    ):
        #print('targets_aerial:', type(targets_aerial))
        #print('images_aerials:', type(images_aerial))
        images_ground = images[::2]
        #print('images_ground', images_ground.shape)
        images_aerial = images[1::2]
        #print('images_aerial', images_aerial.shape)
        targets_ground = targets[::2]
        #print('targets_length', len(targets_ground))
        targets_aerial = targets[1::2]
        #print('targets_aerial', targets_aerial)
        # Move data to the device
        images_aerial, targets_aerial = to_device(images_aerial, targets_aerial, device)
        images_ground, targets_ground = to_device(images_ground, targets_ground, device)
        # Forward pass through the model with both aerial and ground data
        loss_dict = model(images_aerial, images_ground, targets_aerial, targets_ground)  #这个loss_dict是一个字典，包含了所有的loss
        losses = sum(loss for loss in loss_dict.values())   # Total loss

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        # Check if the loss is finite
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        # Backpropagation
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        # Update the optimizer
        optimizer.step()
        # Step the warmup scheduler if it's the first epoch
        if epoch == 0:
            warmup_scheduler.step()
        # Log the metrics
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # Log the metrics to TensorBoard
        if tfboard:
            iter = epoch * len(train_loader) + i    # Total iterations across both datasets
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)

'''
@torch.no_grad()
def evaluate_performance(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images, targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()
            assert (
                gt_box - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images, targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")

    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )
'''
@torch.no_grad()
def evaluate_performance_aerial(
    model, device, gallery_loader_aerial=None, query_loader_aerial=None, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    query_dets, query_feats = None, None    #不需要，参数设置为None
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache_aerial.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader_aerial, ncols=0):
            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images_aerial=images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images_aerial=images, targets_aerial=targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        #query_dets, query_feats = [], []
        #for images, targets in tqdm(query_loader_aerial, ncols=0):
        #    images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
        #    outputs = model(images_aerial=images, targets_aerial=deepcopy(targets), query_img_as_gallery=True)

            # consistency check
        #    gt_box = targets[0]["boxes"].squeeze()
        #    assert (
        #        gt_box - outputs[0]["boxes"][0]
        #    ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

        #    for output in outputs:
        #        box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
        #        query_dets.append(box_w_scores.cpu().numpy())
        #        query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader_aerial, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images_aerial=images, targets_aerial=targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache_aerial.pth")
    print("Aerial view detection results:")
    eval_detection(gallery_loader_aerial.dataset, gallery_dets, det_thresh=0.01)

    '''
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )
    '''

@torch.no_grad()
def evaluate_performance_ground(
    model, device, gallery_loader_ground=None, query_loader_ground=None, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    query_dets, query_feats = None, None    #不需要，参数设置为None
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache_ground.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader_ground, ncols=0):
            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images_ground=images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images_ground=images, targets_ground=targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        #query_dets, query_feats = [], []
        #for images, targets in tqdm(query_loader_ground, ncols=0):
        #    images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
        #    outputs = model(images_ground=images, targets_ground=deepcopy(targets), query_img_as_gallery=True)

            # consistency check
        #    gt_box = targets[0]["boxes"].squeeze()
        #    assert (
        #        gt_box - outputs[0]["boxes"][0]
        #    ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

        #    for output in outputs:
        #        box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
        #        query_dets.append(box_w_scores.cpu().numpy())
        #        query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader_ground, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images_ground=images, targets_ground=targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache_ground.pth")

    print("Ground view detection results:")
    eval_detection(gallery_loader_ground.dataset, gallery_dets, det_thresh=0.01)
    '''
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )
    '''

# 测试的batchsize=1,每次只有一个数据输入，可能是空中图像，也可能是地面图像

@torch.no_grad()
def evaluate_performance_Re_ID(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    query_dets, query_feats = None, None    #不需要，参数设置为None
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):   #这行代码表明测试时，gallery_loader和query_loader的batchsize都是1  
            images, targets = to_device(images, targets, device)
            img_name = targets[0]["img_name"]  #提取图像名称
            if not use_gt:
                #加一个视角的判断
                if img_name.startswith("D"):  #如果图像名称以D开头，说明是空中视角图像
                    outputs = model(images_aerial=images)
                else:                         #否则是地面视角图像
                    outputs = model(images_ground=images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                if img_name.startswith("D"):
                    embeddings = model(images_aerial=images, targets_aerial=targets)
                else:
                    embeddings = model(images_ground=images, targets_ground=targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        # query_dets, query_feats = [], []
        # for images, targets in tqdm(query_loader, ncols=0):
        #    images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            #加一个视角的判断
        #    if img_name.startswith("D"):
        #        outputs = model(images_aerial=images, targets_aerial=deepcopy(targets), query_img_as_gallery=True)
        #    else:
        #        outputs = model(images_ground=images, targets_ground=deepcopy(targets), query_img_as_gallery=True)

            # consistency check
        #    gt_box = targets[0]["boxes"].squeeze()
        #    assert (
        #        gt_box - outputs[0]["boxes"][0]
        #    ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

        #    for output in outputs:
        #        box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
        #        query_dets.append(box_w_scores.cpu().numpy())
        #        query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            #img_name = targets[0]["img_name"]  #提取图像名称
            #加一个视角的判断
            #if img_name.startswith("D"):
            #    embeddings = model(images_aerial=images, targets_aerial=targets)
            #else:
            #    embeddings = model(images_ground=images, targets_ground=targets)
            embeddings = model(images_ground=images, targets_ground=targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")
    #混合数据集测试检测性能
    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    #混合数据集测试Re-ID性能
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )

'''
@torch.no_grad()
def evaluate_performance_Re_ID(
    model, device, gallery_loader_aerial=None, gallery_loader_ground=None, query_loader_ground=None, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for (images_aerial, targets_aerial), (images_ground, targets_ground) in tqdm(zip(gallery_loader_aerial, gallery_loader_ground), ncols=0):
            print(f"Loaded batch - images_aerial: {len(images_aerial)}, images_ground: {len(images_ground)}")
            images_aerial, targets_aerial = to_device(images_aerial, targets_aerial, device)
            images_ground, targets_ground = to_device(images_ground, targets_ground, device)
            if not use_gt:
                outputs = model(images_aerial=images_aerial, images_ground=images_ground)
            else:
                boxes_aerial = targets_aerial[0]["boxes"]
                boxes_ground = targets_ground[0]["boxes"]
                n_boxes_aerial = boxes_aerial.size(0)
                n_boxes_ground = boxes_ground.size(0)
                embeddings_aerial = model(images_aerial=images_aerial, targets_aerial=targets_aerial)
                embeddings_ground = model(images_ground=images_ground, targets_ground=targets_ground)
                outputs = [
                    {
                        "boxes": torch.cat([boxes_aerial, boxes_ground]),
                        "embeddings": torch.cat([embeddings_aerial, embeddings_ground]),
                        "labels": torch.ones(n_boxes_aerial + n_boxes_ground).to(device),
                        "scores": torch.ones(n_boxes_aerial + n_boxes_ground).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for images_ground, targets_ground in tqdm(query_loader_ground, ncols=0):
            images_ground, targets_ground = to_device(images_ground, targets_ground, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images_ground=images_ground, targets_ground=deepcopy(targets_ground), query_img_as_gallery=True)

            # consistency check
            gt_box_ground = targets_ground[0]["boxes"].squeeze()
            assert (
                gt_box_ground - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images_ground, targets_ground in tqdm(query_loader_ground, ncols=0):
            images_ground, targets_ground = to_device(images_ground, targets_ground, device)
            embeddings_ground = model(images_ground=images_ground, targets_ground=targets_ground)
            assert len(embeddings_ground) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings_ground[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")

    eval_detection(gallery_loader_ground.dataset, gallery_dets, det_thresh=0.01)
    eval_search_func = (
        eval_search_cuhk if gallery_loader_ground.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader_ground.dataset,
        query_loader_ground.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )
'''


@torch.no_grad()
def evaluate_performance_Re_ID_0(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    query_dets, query_feats = None, None    #不需要，参数设置为None
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):   #这行代码表明测试时，gallery_loader和query_loader的batchsize都是1  
            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images=images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images=images, targets=targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        # query_dets, query_feats = [], []
        # for images, targets in tqdm(query_loader, ncols=0):
        #    images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            #加一个视角的判断
        #    if img_name.startswith("D"):
        #        outputs = model(images_aerial=images, targets_aerial=deepcopy(targets), query_img_as_gallery=True)
        #    else:
        #        outputs = model(images_ground=images, targets_ground=deepcopy(targets), query_img_as_gallery=True)

            # consistency check
        #    gt_box = targets[0]["boxes"].squeeze()
        #    assert (
        #        gt_box - outputs[0]["boxes"][0]
        #    ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

        #    for output in outputs:
        #        box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
        #        query_dets.append(box_w_scores.cpu().numpy())
        #        query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            #img_name = targets[0]["img_name"]  #提取图像名称
            embeddings = model(images=images, targets=targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")
    #混合数据集测试检测性能
    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    #混合数据集测试Re-ID性能
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )