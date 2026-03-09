#!/usr/bin/env python3
"""
H2O (cam4) 100DOH extraction on FULL-RES RGB (NOT rgb256).
- Scans:  <subject_root>/**/cam4/rgb/*
- Saves:  <...>/cam4/bounding_box/<frame_stem>.txt

Per-frame TXT format (raw pixel coords; NOT normalized):

Line 1:
  W H

Then 5 lines (fixed order):
  LH, RH, LO, RO, THO

Each of those 5 lines:
  flag cx cy x1 y1 x2 y2 score

Where:
  flag = 1 if exists else 0
  (cx, cy) = bbox center in raw pixels
  (x1,y1,x2,y2) = bbox in raw pixels (same coordinate system as RGB image)
  score = detector confidence

Last line:
  twohand_giou lo_idx ro_idx

Notes:
- LH/RH: highest-score hand detection per side.
- LO/RO: only if corresponding hand is "in contact" (state>0) AND matched to an object.
- THO: exists if LO and RO exist AND gIoU(LO,RO) >= --giou_thresh_twohand (default 0.9)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# -----------------------------
# Geometry
# -----------------------------
def bbox_center_xyxy(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def giou_xyxy(a: List[float], b: List[float]) -> float:
    iou = iou_xyxy(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if c_area <= 0:
        return iou

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    union = area_a + area_b - inter
    if union <= 0:
        return iou

    return iou - (c_area - union) / c_area


# -----------------------------
# 100DOH matching (offset-vector -> nearest object center)
# -----------------------------
def match_object_indices(obj_dets: Optional[np.ndarray], hand_dets: Optional[np.ndarray]) -> List[int]:
    if hand_dets is None or len(hand_dets) == 0:
        return []
    if obj_dets is None or len(obj_dets) == 0:
        return [-1] * int(hand_dets.shape[0])

    obj_centers = np.array(
        [bbox_center_xyxy(obj_dets[j, :4].tolist()) for j in range(obj_dets.shape[0])],
        dtype=np.float32,
    )

    out: List[int] = []
    for i in range(hand_dets.shape[0]):
        state = float(hand_dets[i, 5])
        if state <= 0:
            out.append(-1)
            continue

        hx1, hy1, hx2, hy2 = hand_dets[i, :4].tolist()
        hc_x, hc_y = bbox_center_xyxy([hx1, hy1, hx2, hy2])

        # offset_vector: (mag, dir_x, dir_y)
        mag = float(hand_dets[i, 6])
        dir_x = float(hand_dets[i, 7])
        dir_y = float(hand_dets[i, 8])

        px = hc_x + mag * 10000.0 * dir_x
        py = hc_y + mag * 10000.0 * dir_y

        d2 = np.sum((obj_centers - np.array([px, py], dtype=np.float32)) ** 2, axis=1)
        out.append(int(np.argmin(d2)))

    return out


# -----------------------------
# 100DOH detector wrapper (repo must be built)
# -----------------------------
class DOHDetector:
    def __init__(
        self,
        handobj_repo: Path,
        model_dir: Path,
        checkpoint: int,
        checksession: int = 1,
        checkepoch: int = 8,
        net: str = "res101",
        cfg_file: str = "cfgs/res101.yml",
        class_agnostic: bool = False,
        thresh_hand: float = 0.5,
        thresh_obj: float = 0.5,
        use_cuda: bool = False,
    ):
        sys.path.insert(0, str(handobj_repo))
        import _init_paths  # noqa: F401

        from model.utils.config import cfg, cfg_from_file, cfg_from_list  # type: ignore
        from model.faster_rcnn.resnet import resnet  # type: ignore
        from model.faster_rcnn.vgg16 import vgg16  # type: ignore

        self.cfg = cfg
        cfg_from_file(str(handobj_repo / cfg_file))
        cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
        self.cfg.USE_GPU_NMS = use_cuda
        self.cfg.CUDA = use_cuda

        self.pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
        self.class_agnostic = class_agnostic
        self.thresh_hand = float(thresh_hand)
        self.thresh_obj = float(thresh_obj)
        self.use_cuda = bool(use_cuda)

        if net == "vgg16":
            self.net = vgg16(self.pascal_classes, pretrained=False, class_agnostic=class_agnostic)
        elif net == "res101":
            self.net = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=class_agnostic)
        elif net == "res50":
            self.net = resnet(self.pascal_classes, 50, pretrained=False, class_agnostic=class_agnostic)
        elif net == "res152":
            self.net = resnet(self.pascal_classes, 152, pretrained=False, class_agnostic=class_agnostic)
        else:
            raise ValueError(f"Unsupported net: {net}")

        self.net.create_architecture()

        ckpt_path = model_dir / f"faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if self.use_cuda:
            checkpoint_data = torch.load(str(ckpt_path))
        else:
            checkpoint_data = torch.load(str(ckpt_path), map_location=(lambda storage, loc: storage))

        self.net.load_state_dict(checkpoint_data["model"])
        if "pooling_mode" in checkpoint_data:
            self.cfg.POOLING_MODE = checkpoint_data["pooling_mode"]

        if self.use_cuda:
            self.net.cuda()
        self.net.eval()

        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)
        self.box_info = torch.FloatTensor(1)

        if self.use_cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()
            self.box_info = self.box_info.cuda()

        from model.utils.blob import im_list_to_blob  # type: ignore
        from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv  # type: ignore
        from model.roi_layers import nms  # type: ignore

        self._im_list_to_blob = im_list_to_blob
        self._clip_boxes = clip_boxes
        self._bbox_transform_inv = bbox_transform_inv
        self._nms = nms

    def _get_image_blob(self, im_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        im_orig = im_bgr.astype(np.float32, copy=True)
        im_orig -= self.cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
                im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)

            im_resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            processed_ims.append(im_resized)
            im_scale_factors.append(im_scale)

        blob = self._im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors, dtype=np.float32)

    @torch.no_grad()
    def detect(self, im_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        blobs, im_scales = self._get_image_blob(im_bgr)
        assert len(im_scales) == 1

        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        self.gt_boxes.resize_(1, 1, 5).zero_()
        self.num_boxes.resize_(1).zero_()
        self.box_info.resize_(1, 1, 5).zero_()

        rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list = self.net(
            self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info
        )

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        contact_vector = loss_list[0][0]
        offset_vector = loss_list[1][0].detach()
        lr_vector = loss_list[2][0].detach()

        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
        lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

        if self.cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if self.use_cuda:
                    stds = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                    means = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    stds = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS)
                    means = torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                if self.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * stds + means
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * stds + means
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

            pred_boxes = self._bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = self._clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[2]))

        pred_boxes /= im_scales[0]
        scores = scores.squeeze(0)
        pred_boxes = pred_boxes.squeeze(0)

        obj_dets = None
        hand_dets = None

        for j in range(1, len(self.pascal_classes)):
            cls_name = self.pascal_classes[j]
            thresh = self.thresh_hand if cls_name == "hand" else self.thresh_obj
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() == 0:
                continue

            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)

            if self.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

            cls_dets = torch.cat(
                (
                    cls_boxes,
                    cls_scores.unsqueeze(1),
                    contact_indices[inds],
                    offset_vector.squeeze(0)[inds],
                    lr[inds],
                ),
                1,
            )
            cls_dets = cls_dets[order]
            keep = self._nms(cls_boxes[order, :], cls_scores[order], self.cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]

            if cls_name == "targetobject":
                obj_dets = cls_dets.cpu().numpy()
            elif cls_name == "hand":
                hand_dets = cls_dets.cpu().numpy()

        return hand_dets, obj_dets


def pick_best_hand(hand_dets: Optional[np.ndarray], lr_value: int) -> Optional[Tuple[int, np.ndarray]]:
    if hand_dets is None or len(hand_dets) == 0:
        return None
    lr_col = hand_dets[:, -1].astype(np.int32)
    cand = np.where(lr_col == int(lr_value))[0]
    if len(cand) == 0:
        return None
    best = cand[np.argmax(hand_dets[cand, 4])]
    return int(best), hand_dets[best]


def as_line(flag: int, box: Optional[List[float]], score: float) -> str:
    if flag == 0 or box is None:
        return "0 0 0 0 0 0 0 0"
    cx, cy = bbox_center_xyxy(box)
    x1, y1, x2, y2 = box
    return f"1 {cx:.2f} {cy:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.6f}"


def find_rgb_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("cam4/rgb") if p.is_dir()])


def list_frames(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_root", type=str, required=True,
                    help="Root to scan, e.g. .../h2o/subject1_ego or .../h2o")
    ap.add_argument("--handobj_repo", type=str, required=True, help="Path to hand_object_detector repo root")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Folder containing faster_rcnn_{checksession}_{checkepoch}_{checkpoint}.pth")
    ap.add_argument("--checkpoint", type=int, required=True)

    ap.add_argument("--checksession", type=int, default=1)
    ap.add_argument("--checkepoch", type=int, default=8)
    ap.add_argument("--net", type=str, default="res101", choices=["vgg16", "res50", "res101", "res152"])
    ap.add_argument("--cfg_file", type=str, default="cfgs/res101.yml")

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--thresh_hand", type=float, default=0.5)
    ap.add_argument("--thresh_obj", type=float, default=0.5)
    ap.add_argument("--giou_thresh_twohand", type=float, default=0.9)

    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--max_frames_per_seq", type=int, default=-1)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--out_folder_name", type=str, default="bounding_box")
    args = ap.parse_args()

    root = Path(args.subject_root).resolve()
    handobj_repo = Path(args.handobj_repo).resolve()
    model_dir = Path(args.model_dir).resolve()

    rgb_dirs = find_rgb_dirs(root)
    if not rgb_dirs:
        raise RuntimeError(f"No cam4/rgb folders found under: {root}")

    det = DOHDetector(
        handobj_repo=handobj_repo,
        model_dir=model_dir,
        checkpoint=args.checkpoint,
        checksession=args.checksession,
        checkepoch=args.checkepoch,
        net=args.net,
        cfg_file=args.cfg_file,
        thresh_hand=args.thresh_hand,
        thresh_obj=args.thresh_obj,
        use_cuda=args.cuda,
    )

    print(f"[INFO] Found {len(rgb_dirs)} cam4/rgb folders under {root}")

    for img_dir in rgb_dirs:
        cam4_dir = img_dir.parent  # .../cam4
        out_dir = cam4_dir / args.out_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = list_frames(img_dir)
        if args.frame_stride > 1:
            frames = frames[:: max(1, args.frame_stride)]
        if args.max_frames_per_seq != -1:
            frames = frames[: args.max_frames_per_seq]

        print(f"\n[SEQ] {img_dir}  frames={len(frames)}  ->  {out_dir}")

        for img_path in frames:
            out_path = out_dir / f"{img_path.stem}.txt"
            if args.skip_existing and out_path.exists():
                continue

            im_bgr = cv2.imread(str(img_path))
            if im_bgr is None:
                continue
            H, W = im_bgr.shape[:2]

            hand_dets, obj_dets = det.detect(im_bgr)
            matches = match_object_indices(obj_dets, hand_dets)

            left = pick_best_hand(hand_dets, lr_value=0)   # left hand
            right = pick_best_hand(hand_dets, lr_value=1)  # right hand

            lh_box = rh_box = lo_box = ro_box = tho_box = None
            lh_score = rh_score = lo_score = ro_score = tho_score = 0.0
            lo_idx = -1
            ro_idx = -1
            twohand_g = 0.0

            # LH + LO
            if left is not None:
                li, lrow = left
                lh_box = [float(v) for v in lrow[:4]]
                lh_score = float(lrow[4])
                l_state = int(lrow[5])
                if l_state > 0 and li < len(matches) and matches[li] != -1 and obj_dets is not None:
                    lo_idx = int(matches[li])
                    o = obj_dets[lo_idx]
                    lo_box = [float(v) for v in o[:4]]
                    lo_score = float(o[4])

            # RH + RO
            if right is not None:
                ri, rrow = right
                rh_box = [float(v) for v in rrow[:4]]
                rh_score = float(rrow[4])
                r_state = int(rrow[5])
                if r_state > 0 and ri < len(matches) and matches[ri] != -1 and obj_dets is not None:
                    ro_idx = int(matches[ri])
                    o = obj_dets[ro_idx]
                    ro_box = [float(v) for v in o[:4]]
                    ro_score = float(o[4])

            # THO
            if lo_box is not None and ro_box is not None:
                twohand_g = float(giou_xyxy(lo_box, ro_box))
                if twohand_g >= float(args.giou_thresh_twohand):
                    if lo_score >= ro_score:
                        tho_box, tho_score = lo_box, lo_score
                    else:
                        tho_box, tho_score = ro_box, ro_score

            lines = [
                f"{W} {H}",
                as_line(1 if lh_box is not None else 0, lh_box, lh_score),   # LH
                as_line(1 if rh_box is not None else 0, rh_box, rh_score),   # RH
                as_line(1 if lo_box is not None else 0, lo_box, lo_score),   # LO
                as_line(1 if ro_box is not None else 0, ro_box, ro_score),   # RO
                as_line(1 if tho_box is not None else 0, tho_box, tho_score),# THO
                f"{twohand_g:.6f} {lo_idx} {ro_idx}",
            ]
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[OK] Done. Saved under each cam4/bounding_box/.")


if __name__ == "__main__":
    main()