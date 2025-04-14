import os.path as osp
import numpy as np
from scipy.io import loadmat
from .base import BaseDataset

class CUHKSYSU(BaseDataset):
    def __init__(self, root, transforms, split, view='ground'):
        self.name = "CUHK-SYSU"
        self.img_prefix = osp.join(root, "Image", "SSM")
        self.view = view
        super(CUHKSYSU, self).__init__(root, transforms, split)

        #记录数据集大小
        self.images_aerial_train = 10810
        self.images_ground_train = 11152

        #需要补全空中数据集
        if self.view == 'aerial' and self.split == 'train':
            self._pad_aerial_dataset()

    def _pad_aerial_dataset(self):
        """
        随机抽取 342 个数据，将空中视角数据集补全到和地面视角数据集一样的大小。
        """
        # 计算需要补全的数量
        num_to_pad = self.images_ground_train - self.images_aerial_train
        if num_to_pad <= 0:
            return  # 不需要补全

        # 加载原始空中数据集
        original_annotations = self.annotations

        # 随机抽取 342 个数据（不允许重复）

        random_indices = np.random.choice(len(original_annotations), size=num_to_pad, replace=False)
        padded_annotations = [original_annotations[i] for i in random_indices]

        # 将补全的数据添加到原始数据集中
        self.annotations.extend(padded_annotations)
        self.images_aerial_train += num_to_pad  # 更新数据集大小

    def _is_valid_view(self, img_name):
        if self.view == 'ground':
            return img_name.startswith('S')
        elif self.view == 'aerial':
            return img_name.startswith('D')
        return False

    def _load_queries(self):
        protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
        protoc = protoc["TestG50"].squeeze()
        queries = []
        for item in protoc["Query"]:
            img_name = str(item["imname"][0, 0][0])
            if not self._is_valid_view(img_name):
                continue
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([-100]),  # dummy pid
                }
            )
        return queries

    def _load_split_img_names(self):
        assert self.split in ("train", "gallery")
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs if self._is_valid_view(str(a[0]))]
        if self.split == "gallery":
            return gallery_imgs
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs if self._is_valid_view(str(a[0][0]))]
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        unlabeled_pid = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:
            img_name = str(img_name[0])
            if not self._is_valid_view(img_name):
                continue
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
            boxes = boxes[valid_index]
            name_to_boxes[img_name] = boxes.astype(np.int32)
            name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return

        if self.split == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for img_name, box, _ in scenes:
                    img_name = str(img_name[0])
                    if not self._is_valid_view(img_name):
                        continue
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
        else:
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):
                im_name = str(item["Query"][0, 0][0][0])
                if not self._is_valid_view(im_name):
                    continue
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if not self._is_valid_view(im_name):
                        continue
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
            pids = name_to_pids[img_name]
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": boxes,
                    "pids": pids,
                }
            )
        return annotations



# 加载地面视角的训练集
#train_dataset_ground = CUHKSYSU(root='/home/wqf/Data/G2APS/G2APS/' , split='train', view='ground')

# 加载空中视角的训练集
#train_dataset_aerial = CUHKSYSU(root='/home/wqf/Data/G2APS/G2APS/' , split='train', view='aerial')

# 加载地面视角的图库
#gallery_dataset_ground = CUHKSYSU(root='/home/wqf/Data/G2APS/G2APS/' , split='gallery', view='ground')

# 加载空中视角的图库
#gallery_dataset_aerial = CUHKSYSU(root='/home/wqf/Data/G2APS/G2APS/' , split='gallery', view='aerial')

# 加载地面视角的查询集
#query_dataset_ground = CUHKSYSU(root='/home/wqf/Data/G2APS/G2APS/' , split='query', view='ground')

# 加载空中视角的查询集
#query_dataset_aerial = CUHKSYSU(root='/home/wqf/Data/G2APS/G2APS/' , split='query', view='aerial')