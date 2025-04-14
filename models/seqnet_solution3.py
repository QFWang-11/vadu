from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from models.oim import OIMLoss
from models.resnet import build_resnet
from collections import OrderedDict
from .image_list import ImageList

'''
class SeqNet(nn.Module):
    def __init__(self, cfg):
        super(SeqNet, self).__init__()

        backbone, box_head = build_resnet(name="resnet34", pretrained=True)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )
        # RegionProposalNetwork：区域建议网络，用于生成候选区域。
        # FastRCNNPredictor：用于分类和回归的预测器。
        # MultiScaleRoIAlign：多尺度 RoI 对齐，用于从特征图中提取 RoI 特征。
        # BBoxRegressor：边界框回归器。
        # SeqRoIHeads：RoI 头部，用于处理 RoI 特征。
        faster_rcnn_predictor = FastRCNNPredictor(512, 2)    # ResNet-34的输出通道数为512，类别数为2
        reid_head = deepcopy(box_head)    #Re-ID的head是检测head的深拷贝
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(512, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )
        # 图像预处理
        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID

    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features)
            embeddings, _ = self.roi_heads.embedding_head(box_features)
            return embeddings.split(1, 0)
        else:
            # gallery
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)  #候选区域proposals通过坐标值表示，loss是一个字典，包括分类损失（classification loss）：用于区分前景（包含目标）和背景（不包含目标）。回归损失（regression loss）：用于调整候选区域的边界框，使其更准确地定位目标
        _, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # roi_heads, Region of interest,用于进一步处理候选区域，输出检测结果和损失，detections是roi_heads在推理阶段生成的检测结果，包括了最终的检测框、类别标签、置信度分数和嵌入特征。即包含多个字典的列表
        # detector_losses是训练阶段roi_heads的损失，包括了分类损失、回归损失，是一个字典，loss_classifier：分类损失，用于区分不同类别；loss_box_reg：回归损失，用于调整检测框的位置和大小
        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        # 先训练检测部分，移除Re-ID损失
        losses["loss_box_reid"] *= self.lw_box_reid
        return losses


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,
        reid_head,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = NormAwareEmbedding()
        self.reid_loss = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(
                proposals, targets
            )

        # ------------------- Faster R-CNN head ------------------ #
        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, _, box_pid_labels, box_reg_targets = self.select_training_samples(boxes, targets)
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, _ = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )

        cws = True
        gt_det = None
        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head(gt_box_features)
            embeddings, _ = self.embedding_head(gt_box_features)
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, 256)
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- Baseline head -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features)
        box_regs = self.box_predictor(box_features["feat_res5"])
        box_embeddings, box_cls_scores = self.embedding_head(box_features)
        if box_cls_scores.dim() == 0:
            box_cls_scores = box_cls_scores.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            box_labels = [y.clamp(0, 1) for y in box_pid_labels]
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                box_cls_scores,
                box_regs,
                box_labels,
                box_reg_targets,
            )
            # 移除Re-ID损失
            loss_box_reid = self.reid_loss(box_embeddings, box_pid_labels)
            losses.update(loss_box_reid=loss_box_reid)
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_cls_scores,
                box_regs,
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]
                    )
                )
        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels
'''

class SeqNet(nn.Module):
    def __init__(self, cfg):
        super(SeqNet, self).__init__()

        backbone, box_head = build_resnet(name="resnet50", pretrained=True)
        #两视角数据集的backbone(res1-4)也不共享
        backbone_ground = backbone
        backbone_aerial = deepcopy(backbone)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        # RPN网络生成候选区域proposals，proposals通过坐标值表示，loss是一个字典，包括分类损失（classification loss）：用于区分前景（包含目标）和背景（不包含目标）。回归损失（regression loss）：用于调整候选区域的边界框，使其更准确地定位目标
        # 创建两个独立的RPN实例
        rpn_air = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )
        rpn_ground = deepcopy(rpn_air)  # 深拷贝 RPN 实例

        # RegionProposalNetwork：区域建议网络，用于生成候选区域。
        # FastRCNNPredictor：用于分类和回归的预测器。
        # MultiScaleRoIAlign：多尺度 RoI 对齐，用于从特征图中提取 RoI 特征。
        # BBoxRegressor：边界框回归器。
        # SeqRoIHeads：RoI 头部，用于处理 RoI 特征。
        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)    # ResNet-34的输出通道数为512，类别数为2, 不同视角的预测器在SeqRoIHead中进行deepcopy

        reid_head = deepcopy(box_head)    #Re-ID的head是检测head的深拷贝

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=(16,8), sampling_ratio=2     #传统方案
        )    #RoI pooling

        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)

        # SeqRoIHeads handles object detection and Re-ID
        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )
        # 图像预处理
        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        #self.backbone = backbone
        self.backbone_aerial = backbone_aerial   #backbone参数不共享
        self.backbone_ground = backbone_ground
        self.rpn_air = rpn_air
        self.rpn_ground = rpn_ground
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID

    def inference(self, images_aerial=None, images_ground=None, targets_aerial=None, targets_ground=None, query_img_as_gallery=False):
        """
        images: 同时包括空中和地面视角的测试图像
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        推理的batchsize=1,只需要判断视角,不需要进行一些特征的拼接
        """

        original_image_sizes = []
        images = []
        targets = []
        
        # 记录空中视角和地面视角的图像数量
        num_images_aerial = 0
        num_images_ground = 0

        # 处理空中视角的图像和目标
        if images_aerial is not None:
            original_image_sizes.extend([img.shape[-2:] for img in images_aerial])
            images.extend(images_aerial)
            num_images_aerial = len(images_aerial)
            if targets_aerial is not None:
                targets.extend(targets_aerial)
            else:
                targets = None
            
            # 将图像和目标转换为 ImageList 对象
            images, targets = self.transform(images, targets)
            features = self.backbone_aerial(images.tensors)

            if query_img_as_gallery:
                assert targets is not None
            
            if targets is not None and not query_img_as_gallery:
                # query 空中视角
                boxes = [t["boxes"] for t in targets]
                box_features = self.roi_heads.box_roi_pool_air(features, boxes, images.image_sizes)
                box_features = self.roi_heads.reid_head_air(box_features)
                embeddings, _ = self.roi_heads.embedding_head_aerial(box_features)
                return embeddings.split(1, 0)
                pass
            else:
                # gallery 空中视角
                features_aerial = OrderedDict([("feat_res4", features["feat_res4"])])    #features_aerial和features是一致的
                proposals_aerial, _ = self.rpn_air(images, features_aerial, targets)

                detections, _ = self.roi_heads(
                    features, images.image_sizes, features_aerial, None, proposals_aerial, None, targets_aerial, None, query_img_as_gallery
                )
            
                detections = self.transform.postprocess(
                    detections, images.image_sizes, original_image_sizes
                )
                return detections



        # 处理地面视角的图像和目标
        if images_ground is not None:
            original_image_sizes.extend([img.shape[-2:] for img in images_ground])
            images.extend(images_ground)
            num_images_ground = len(images_ground)
            if targets_ground is not None:
                targets.extend(targets_ground)
            else:
                targets = None
            
            # 将图像和目标转换为 ImageList 对象
            images, targets = self.transform(images, targets)
            features = self.backbone_ground(images.tensors)
            
            if query_img_as_gallery:
                assert targets is not None

            if targets is not None and not query_img_as_gallery:
                # query 地面视角
                boxes = [t["boxes"] for t in targets]
                box_features = self.roi_heads.box_roi_pool_ground(features, boxes, images.image_sizes)
                box_features = self.roi_heads.reid_head_ground(box_features)
                embeddings, _ = self.roi_heads.embedding_head_ground(box_features)
                return embeddings.split(1, 0)
            else:
                # gallery 地面视角
                features_ground = OrderedDict([("feat_res4", features["feat_res4"])])
                proposals_ground, _ = self.rpn_ground(images, features_ground, targets)

                detections, _ = self.roi_heads(
                    features, images.image_sizes, None, features_ground, None, proposals_ground, None, targets_ground, query_img_as_gallery
                )
                detections = self.transform.postprocess(
                    detections, images.image_sizes, original_image_sizes
                )
                return detections

    def forward(self, images_aerial=None, images_ground=None, targets_aerial=None, targets_ground=None, query_img_as_gallery=False):
        # 如果模型不是处于训练状态，则调用inference方法执行前向传播，并生成检测结果或嵌入特征
        if not self.training:
            if images_aerial is not None and images_ground is None:
                return self.inference(images_aerial, None, targets_aerial, None, query_img_as_gallery)
            elif images_aerial is None and images_ground is not None:
                return self.inference(None, images_ground, None, targets_ground, query_img_as_gallery)
            else:  #两个视角都有
                return self.inference(images_aerial, images_ground, targets_aerial, targets_ground, query_img_as_gallery)

        # 记录每个部分的索引范围
        batch_size_air = len(images_aerial)
        batch_size_ground = len(images_ground)

        # 将两个视角的图像和目标在 batch 维度上拼接
        images = images_aerial + images_ground
        targets = targets_aerial + targets_ground if targets_aerial is not None and targets_ground is not None else None

        # Transform images and targets
        images, targets = self.transform(images, targets)  #ImageList object

        # 打印转换后的图像和目标的维度
        #print("images.tensors shape:", images.tensors.shape)
        #print("targets:", targets)

        # 分别提取出 images_aerial 和 images_ground
        images_aerial_tensors = images.tensors[:batch_size_air]
        images_ground_tensors = images.tensors[batch_size_air:]

        # 重新构造 ImageList 对象
        images_aerial = ImageList(images_aerial_tensors, images.image_sizes[:batch_size_air])
        images_ground = ImageList(images_ground_tensors, images.image_sizes[batch_size_air:])

        # 分别提取出 targets_aerial 和 targets_ground
        if targets is not None:
            targets_aerial = targets[:batch_size_air]
            targets_ground = targets[batch_size_air:]
        
        #features = self.backbone(images.tensors)
        features_aerial = self.backbone_aerial(images_aerial.tensors)  #输出是一个字典，key是“feat_res4”, value的shape是torch.Size([10, 256, 54, 94])
        features_ground = self.backbone_ground(images_ground.tensors)
        
        features = torch.cat((features_aerial["feat_res4"], features_ground["feat_res4"]), dim=0)   #这样features还是一个字典类型，把values的值在batchsize维度拼接
        # 打印通过backbone后的features的维度
        #for key, value in features.items():
        #    print(f"{key} shape: {value.shape}")  #输出feat_res4 shape: torch.Size([10, 256, 54, 94])
        
        # Split features for aerial and ground views
        #features_aerial, features_ground = torch.split(features["feat_res4"], [batch_size_air, features["feat_res4"].size(0) - batch_size_air], dim=0)
        # 将拆分后的特征重新包装成OrderedDict
        #features_aerial = OrderedDict([("feat_res4", features_aerial)])
        #features_ground = OrderedDict([("feat_res4", features_ground)])

        # 打印分割后的特征维度
        #print("features_aerial shape:", features_aerial["feat_res4"].shape)
        #print("features_ground shape:", features_ground["feat_res4"].shape)
        
        #两个特征通过各自的RPN, RoIHeads是进行Re-ID的部分，不需要进行特征拆分，需要将两个视角RPN的输出拼接
        proposals_aerial, proposal_losses_aerial = self.rpn_air(images_aerial, features_aerial, targets_aerial)
        
        proposals_ground, proposal_losses_ground = self.rpn_ground(images_ground, features_ground, targets_ground)
        
        # 获取 image_sizes
        image_sizes = images_aerial.image_sizes + images_ground.image_sizes

        _, detector_losses = self.roi_heads(features, image_sizes, features_aerial, features_ground, proposals_aerial, proposals_ground, targets_aerial, targets_ground)



        # 检测部分的loss
        # rename rpn losses to be consistent with detection losses
        proposal_losses_aerial["loss_rpn_reg"] = proposal_losses_aerial.pop("loss_rpn_box_reg")
        proposal_losses_aerial["loss_rpn_cls"] = proposal_losses_aerial.pop("loss_objectness")
        
        proposal_losses_ground["loss_rpn_reg"] = proposal_losses_ground.pop("loss_rpn_box_reg")
        proposal_losses_ground["loss_rpn_cls"] = proposal_losses_ground.pop("loss_objectness")

        # 对aerial和ground视角对应loss乘以0.5再相加
        combined_rpn_losses = {}
        for key in proposal_losses_aerial.keys():
            combined_rpn_losses[key] = 0.5 * proposal_losses_aerial[key] + 0.5 * proposal_losses_ground[key]
        
        # roi_heads, Region of interest,用于进一步处理候选区域，输出检测结果和损失，detections是roi_heads在推理阶段生成的检测结果，包括了最终的检测框、类别标签、置信度分数和嵌入特征。即包含多个字典的列表
        # detector_losses是训练阶段roi_heads的损失，包括了分类损失、回归损失，是一个字典，loss_classifier：分类损失，用于区分不同类别；loss_box_reg：回归损失，用于调整检测框的位置和大小
        
        # rename rpn losses to be consistent with detection losses
        losses = {}
        losses.update(detector_losses)
        losses.update(combined_rpn_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        # 先训练检测部分，移除Re-ID损失
        losses["loss_box_reid"] *= self.lw_box_reid
        return losses  #模型返回的就是所有loss的字典

# 使用独立的Re-ID head，共享OIM loss
# 训练和推理逻辑：在训练时，分别计算两个视角的检测损失和Re-ID损失，最后合并损失。在推理时，合并两个视角的检测结果和Re-ID特征。
class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        faster_rcnn_predictor,
        reid_head,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        # 使用 deepcopy 创建独立的 embedding_head，确保参数不共享
        self.embedding_head = NormAwareEmbedding()
        self.embedding_head_aerial = deepcopy(self.embedding_head)
        self.embedding_head_ground = deepcopy(self.embedding_head)

        self.reid_loss = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)

        # 使用 deepcopy 创建独立的 faster_rcnn_predictor head，确保参数不共享
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.faster_rcnn_predictor_air = deepcopy(self.faster_rcnn_predictor)
        self.faster_rcnn_predictor_ground = deepcopy(self.faster_rcnn_predictor)

        # 使用 deepcopy 创建独立的 box_roi_pool，确保参数不共享
        self.box_roi_pool_air = deepcopy(self.box_roi_pool)
        self.box_roi_pool_ground = deepcopy(self.box_roi_pool)

        # 使用 deepcopy 创建独立的 detection head，确保参数不共享
        self.box_head_air = deepcopy(self.box_head)
        self.box_head_ground = deepcopy(self.box_head)

        # 使用 deepcopy 创建独立的 Re-ID head，确保参数不共享
        self.reid_head_air = deepcopy(reid_head)
        self.reid_head_ground = deepcopy(reid_head)

        # 使用 deepcopy 创建独立的 box_predictor， 确保参数不共享
        self.box_predictor_air = deepcopy(self.box_predictor)
        self.box_predictor_ground = deepcopy(self.box_predictor)

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections   #传统计算模块

    def forward(self, features, image_shapes, features_air=None, features_ground=None, proposals_air=None, proposals_ground=None, targets_aerial=None, targets_ground=None, query_img_as_gallery=False):
        """
        Arguments:
            features_air (List[Tensor]): Features from the air view backbone.
            features_ground (List[Tensor]): Features from the ground view backbone.
            proposals_air (List[Tensor[N, 4]]): Proposals from the air view RPN.
            proposals_ground (List[Tensor[N, 4]]): Proposals from the ground view RPN.
            image_shapes (List[Tuple[H, W]]): Shapes of the input images.
            targets_aerial (List[Dict]): Ground truth aerial targets.
            targets_ground (List[Dict]): Ground truth ground targets.
        """
        if self.training:
            # Select training samples for air and ground views
            if proposals_air is not None:
                proposals_air, _, proposal_pid_labels_air, proposal_reg_targets_air = self.select_training_samples(
                    proposals_air, targets_aerial
                )
            if proposals_ground is not None:
                proposals_ground, _, proposal_pid_labels_ground, proposal_reg_targets_ground = self.select_training_samples(
                    proposals_ground, targets_ground
                )

        # ------------------- Faster R-CNN head for air view ------------------ #
        if features_air is not None and proposals_air is not None:
            proposal_features_air = self.box_roi_pool_air(features_air, proposals_air, image_shapes)
            proposal_features_air = self.box_head_air(proposal_features_air)
            proposal_cls_scores_air, proposal_regs_air = self.faster_rcnn_predictor_air(
                proposal_features_air["feat_res5"]
            )
        else:
            proposal_cls_scores_air, proposal_regs_air = None, None

        # ------------------- Faster R-CNN head for ground view ------------------ #
        if features_ground is not None and proposals_ground is not None:
            proposal_features_ground = self.box_roi_pool_ground(features_ground, proposals_ground, image_shapes)
            proposal_features_ground = self.box_head_ground(proposal_features_ground)
            proposal_cls_scores_ground, proposal_regs_ground = self.faster_rcnn_predictor_ground(
                proposal_features_ground["feat_res5"]
            )
        else:
            proposal_cls_scores_ground, proposal_regs_ground = None, None

        if self.training:
            # Get boxes for air and ground views
            if proposal_regs_air is not None and proposals_air is not None:
                boxes_air = self.get_boxes(proposal_regs_air, proposals_air, image_shapes)
                boxes_air = [boxes_per_image.detach() for boxes_per_image in boxes_air]
                boxes_air, _, box_pid_labels_air, box_reg_targets_air = self.select_training_samples(boxes_air, targets_aerial)
            else:
                boxes_air, box_pid_labels_air, box_reg_targets_air = [], [], []

            if proposal_regs_ground is not None and proposals_ground is not None:
                boxes_ground = self.get_boxes(proposal_regs_ground, proposals_ground, image_shapes)
                boxes_ground = [boxes_per_image.detach() for boxes_per_image in boxes_ground]
                boxes_ground, _, box_pid_labels_ground, box_reg_targets_ground = self.select_training_samples(boxes_ground, targets_ground)
            else:
                boxes_ground, box_pid_labels_ground, box_reg_targets_ground = [], [], []
        else:
            # Postprocess proposals for air and ground views
            if proposal_cls_scores_air is not None and proposal_regs_air is not None and proposals_air is not None:
                boxes_air, scores_air, _ = self.postprocess_proposals(
                    proposal_cls_scores_air, proposal_regs_air, proposals_air, image_shapes
                )
            else:
                boxes_air, scores_air = None, None

            if proposal_cls_scores_ground is not None and proposal_regs_ground is not None and proposals_ground is not None:
                boxes_ground, scores_ground, _ = self.postprocess_proposals(
                    proposal_cls_scores_ground, proposal_regs_ground, proposals_ground, image_shapes
                )
            else:
                boxes_ground, scores_ground = None, None

        # --------------------- Re-ID head for air view -------------------- #
        if boxes_air and boxes_air[0].shape[0]>0:     #不同虚拟环境版本对图像的处理有一些不同
            box_features_air = self.box_roi_pool_air(features_air, boxes_air, image_shapes)   #其实也可以不用deepcopy，因为就是传统计算方法
            box_features_air = self.reid_head_air(box_features_air)
            box_regs_air = self.box_predictor_air(box_features_air["feat_res5"])
            box_embeddings_air, box_cls_scores_air = self.embedding_head_aerial(box_features_air)
            if box_cls_scores_air.dim() == 0:
                box_cls_scores_air = box_cls_scores_air.unsqueeze(0)
        else:
            box_regs_air, box_embeddings_air, box_cls_scores_air = None, None, None

        # --------------------- Re-ID head for ground view -------------------- #
        if boxes_ground and boxes_ground[0].shape[0]>0:
            box_features_ground = self.box_roi_pool_ground(features_ground, boxes_ground, image_shapes)  
            box_features_ground = self.reid_head_ground(box_features_ground)
            box_regs_ground = self.box_predictor_ground(box_features_ground["feat_res5"])
            box_embeddings_ground, box_cls_scores_ground = self.embedding_head_ground(box_features_ground)
            if box_cls_scores_ground.dim() == 0:
                box_cls_scores_ground = box_cls_scores_ground.unsqueeze(0)
        else:
            box_regs_ground, box_embeddings_ground, box_cls_scores_ground = None, None, None
        

        result, losses = [], {}
        if self.training:
            # Compute detection losses for air and ground views
            if proposal_cls_scores_air is not None and proposal_regs_air is not None:
                proposal_labels_air = [y.clamp(0, 1) for y in proposal_pid_labels_air]
                box_labels_air = [y.clamp(0, 1) for y in box_pid_labels_air]
                losses_air = detection_losses(
                    proposal_cls_scores_air,
                    proposal_regs_air,
                    proposal_labels_air,
                    proposal_reg_targets_air,
                    box_cls_scores_air,
                    box_regs_air,
                    box_labels_air,
                    box_reg_targets_air
                )
            else:
                losses_air = {}

            if proposal_cls_scores_ground is not None and proposal_regs_ground is not None:
                proposal_labels_ground = [y.clamp(0, 1) for y in proposal_pid_labels_ground]
                box_labels_ground = [y.clamp(0, 1) for y in box_pid_labels_ground]
                losses_ground = detection_losses(
                    proposal_cls_scores_ground,
                    proposal_regs_ground,
                    proposal_labels_ground,
                    proposal_reg_targets_ground,
                    box_cls_scores_ground,
                    box_regs_ground,
                    box_labels_ground,
                    box_reg_targets_ground
                )
            else:
                losses_ground = {}

            # 对losses_air和losses_ground的对应元素乘以0.5后相加
            combined_losses = {}
            for key in set(losses_air.keys()).union(losses_ground.keys()):
                combined_losses[key] = 0.5 * losses_air.get(key, 0) + 0.5 * losses_ground.get(key, 0)

            # box_embedidngs类嵌入特征，形状为(N,D)的张量，N为检测框的数量，D为嵌入特征的维度
            # combined_pid_labels是合并后的Person ID标签，用于计算Re-ID损失
            # 合并 box_pid_labels  合并计算Re-ID loss
            #print('box_embedding_air',box_embeddings_air.shape)       #([640,256])
            #print('box_embedding_ground',box_embeddings_ground.shape) #([640,256])
            #print('box_pid_labels_air', box_pid_labels_air)
            #box_embeddings = box_embeddings_air + box_embeddings_ground
            box_embeddings = torch.cat((box_embeddings_air, box_embeddings_ground), dim=0)
            #print('box_embedding',box_embeddings.shape)
            combined_pid_labels = box_pid_labels_air + box_pid_labels_ground
            loss_box_reid = self.reid_loss(box_embeddings, combined_pid_labels)   #共享OIM loss的计算
            combined_losses.update(loss_box_reid=loss_box_reid)

            losses = {**combined_losses}

        else:
            if box_regs_ground is None and box_regs_air is None:
                result.append(
                    dict(
                        boxes=torch.zeros((0,4), dtype=torch.int, device=features["feat_res4"].device), labels=torch.zeros((0,), dtype=torch.int, device=features["feat_res4"].device), scores=torch.zeros((0,), device=features["feat_res4"].device), embeddings=torch.zeros((0,256), device=features["feat_res4"].device)
                    )
                )
            else:
                # Postprocess boxes for inference
                boxes, scores, embeddings, labels = self.postprocess_boxes(
                    class_logits=box_cls_scores_air if box_cls_scores_air is not None else box_cls_scores_ground,
                    box_regression=box_regs_air if box_regs_air is not None else box_regs_ground,
                    embeddings=box_embeddings_air if box_embeddings_air is not None else box_embeddings_ground,
                    proposals=boxes_air if boxes_air is not None else boxes_ground,
                    image_shapes=image_shapes,
                    fcs=scores_air if scores_air is not None else scores_ground,
                    gt_det=None,
                    cws=True,
                    
                )
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        dict(
                            boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]
                        )
                    )
        return result, losses
    
    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        
        # 添加调试代码，检查 proposals 和 class_logits 的数量
        #print(f"proposals shape: {sum(boxes_per_image)}")
        #print(f"class_logits shape: {class_logits.shape[0]}")
        #print(f"fcs shape: {fcs[0].shape if fcs is not None else 'None'}")

        if cws:
            # Confidence Weighted Similarity (CWS)
            # 添加调试代码，检查 embeddings 和 pred_scores 的尺寸
            #print(f"embeddings shape: {embeddings.shape}")
            #print(f"pred_scores shape: {pred_scores.shape}")
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp
                                                         

class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
    box_cls_scores,
    box_regs,
    box_labels,
    box_reg_targets,
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    box_labels = torch.cat(box_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)
    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores, box_labels.float())

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )


'''
#只进行faster-rcnn检测部分的loss计算
def detection_losses_1(
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()


    return dict(
        loss_proposal_cls=loss_proposal_cls,   #RPN网络的分类损失
        loss_proposal_reg=loss_proposal_reg,   #RPN网络的回归损失
    )

#只进行RoIHeads检测部分的loss计算
def detection_losses_2(
    box_cls_scores,
    box_regs,
    box_labels,
    box_reg_targets,
):
    box_labels = torch.cat(box_labels, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores, box_labels.float())

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_box_cls=loss_box_cls,             #RoIHeads的分类损失
        loss_box_reg=loss_box_reg,             #RoIHeads的回归损失
    )
'''