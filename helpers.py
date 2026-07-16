"""
Self-contained helper utilities for WildFL.

This module bundles everything the federated-learning algorithms in ``algorithms/``
need at runtime: the canonical directory/file names of a prepared dataset, model
evaluation (tiled and full-image inference, image-level counting errors, detection
metrics), and a few small convenience utilities.

The code was extracted from the DC11_Utils codebase and reduced to the
box-detection functionality used in this repository.
"""

import json
import math
import pickle
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.backends import cudnn

from sahi.predict import predict
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou


__all__ = ["DATA_ANN_FORMATS", "CLASS_COLORS", "NAMES_DIRECTORIES_FILES",
           "init_rng", "modify_cfg_yaml_nc", "read_YOLO_label_file",
           "load_img_gt", "match_predictions", "plot_img_predictions",
           "read_output_SAHI", "img_inference",
           "compute_errors_img_lvl", "compute_em_img_lvl",
           "eval_single_model", "get_single_metric_single_client"]


# index/key maps describing how annotations are stored in the dataset annotation files
DATA_ANN_FORMATS = {
    "BX_WH": {"label_key": "bbox", "x_min_idx": 0, "y_min_idx": 1, "width_idx": 2, "height_idx": 3, "category_key": "category_id"}
}

# BGR colors used when visualizing predictions
CLASS_COLORS = {0: (134, 22, 171),      # purple
                1: (38, 154, 247),      # brown
                2: (204, 232, 19),      # turquoise
                3: (0, 97, 242),        # orange
                4: (0, 255, 0),         # green
                5: (255, 0, 0)}         # blue

# canonical names of the files and directories that make up a prepared dataset
NAMES_DIRECTORIES_FILES = {"count_directory":       "image_counts",
                           "train":                 "train",
                           "val":                   "val",
                           "test":                  "test",
                           "val_full_dir":          "val_full_imgs",
                           "inference_output_dir":  "detections",
                           "annotations_json":      "annotations.json",
                           "ultralytics_yaml":      "dataset.yaml",
                           "dataset_file_SAHI":     "dataset_SAHI.json",
                           "total_counts_file":     "counts_total.json",
                           "FL_all_data_dir":       "all_data",
                           "FL_client_stats_json":  "client_stats.json",
                           "FL_state_dict":         "state.pt"
                          }


def init_rng(seed: int) -> None:
    """
    Primes all sorts of random number generator seeds.

    Args:
        seed (int): random seed to prime RNGs with.
    """
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True


def modify_cfg_yaml_nc(yaml_file: str, nc: int) -> None:
    """
    Update only the 'nc' value in a YAML file without changing any other formatting or content.

    Arguments:
        yaml_file (str):    Path to the .yaml file that needs modifying.
        nc (int):           Number of classes to set.
    """
    new_lines = []
    with open(yaml_file, "r") as f:
        for line in f:
            if line.strip().startswith("nc:"):
                # Replace the line with the new value, preserving indentation
                indent = line[:line.find("nc:")]
                new_lines.append(f"{indent}nc: {nc}\n")
            else:
                new_lines.append(line)

    with open(yaml_file, "w") as f:
        f.writelines(new_lines)


def read_YOLO_label_file(file_path: str, convert2abs: bool, bx_format: str = "x1y1x2y2", img_size: int = None) -> torch.Tensor:
    """
    Read a YOLO .txt label file into a tensor. Optionally converts normalized coordinates to
    absolute pixel values and box coordinates to the x1y1x2y2 format.
    Arguments:
        file_path (str):        path to the label file.
        convert2abs (bool):     if True, coordinates are multiplied by img_size.
        bx_format (str):        target box format ("x1y1x2y2" converts from xywh).
        img_size (int):         image size used for the conversion to absolute coordinates.
    Returns:
        torch.Tensor containing one row per annotation.
    """
    with open(file_path, encoding="utf-8") as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        lb = np.array(lb, dtype=np.float32)

    if convert2abs:
       lb[:, 1:] = lb[:, 1:] * img_size

    if bx_format == "x1y1x2y2":
        lb[:, 3] = lb[:, 1] + lb[:, 3]
        lb[:, 4] = lb[:, 2] + lb[:, 4]

    return torch.from_numpy(lb)


def load_img_gt(annotations: list, ann_format: str, device: torch.device, box_dims: dict = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expects the annotations in an image as a list of dictionaries, and from that creates tensors for the ground truth labels.
    Arguments:
        annotations (list):                 list containing all annotations (as dictionaries) within the image under consideration.
        ann_format (string):                string indicating the annotation format.
        device (torch.device):              device on which to store the gt tensors.
        box_dims (dict):                    dimensions of ground-truth bounding boxes. For cases where dimensions that differ
                                            from what is specified in the annotations were used.
    Returns:
        tuple[torch.Tensor, torch.Tensor]:  the ground truth box coordinates (x1y1x2y2) and classes.
    """
    coords_list = []
    cls_list = []
    for ann in annotations:
        label = ann[DATA_ANN_FORMATS[ann_format]["label_key"]]
        cat_id = ann[DATA_ANN_FORMATS[ann_format]["category_key"]]

        x_center = label[DATA_ANN_FORMATS[ann_format]["x_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["width_idx"]] / 2.0)
        y_center = label[DATA_ANN_FORMATS[ann_format]["y_min_idx"]] + (label[DATA_ANN_FORMATS[ann_format]["height_idx"]] / 2.0)

        box_dims_checked = {}
        if box_dims:
            box_dims_checked["width"] = box_dims[cat_id]["width"]
            box_dims_checked["height"] = box_dims[cat_id]["height"]
        else:
            box_dims_checked["width"] = label[DATA_ANN_FORMATS[ann_format]["width_idx"]]
            box_dims_checked["height"] = label[DATA_ANN_FORMATS[ann_format]["height_idx"]]

        xmin = x_center - (box_dims_checked["width"] / 2.0)
        xmax = x_center + (box_dims_checked["width"] / 2.0)
        ymin = y_center - (box_dims_checked["height"] / 2.0)
        ymax = y_center + (box_dims_checked["height"] / 2.0)
        coords_list.append([xmin, ymin, xmax, ymax])

        cls_list.append(ann["category_id"])

    if not coords_list:
        coords_t = torch.empty((0, 4), dtype=torch.float)
        cls_t = torch.empty((0, 1))
    else:
        coords_t = torch.tensor(coords_list, device=device, dtype=torch.float)
        cls_t = torch.tensor(cls_list, device=device)

    return coords_t, cls_t


# copied from the ultralytics repo (validator.py) to avoid circular imports
def match_predictions(pred_classes, true_classes, iou, min_iou=0.5, use_scipy=False):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
        min_iou (float): Minimum IoU threshold to evaluate predictions at.
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    iouv = torch.linspace(min_iou, 0.95, int(round((0.95 - min_iou) / 0.05)) + 1)
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        if use_scipy:
            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
            import scipy  # scope import to avoid importing for all commands

            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def plot_img_predictions(img_fn: str, coords: torch.Tensor, cls: torch.Tensor, output_dir: str, gt_coords: torch.Tensor = None,
                         gt_class: torch.Tensor = None, plot_gt_color: bool = False, pre_nms: bool = False) -> None:
    """
    Plot an image and visualize the corresponding box predictions. If provided, the ground truth is plotted as well.
    Arguments:
        img_fn (str):               path to the image file.
        coords (torch.Tensor):      tensor containing the prediction box coordinates in xyxy format.
        cls (torch.Tensor):         tensor containing the predicted classes.
        output_dir (str):           path to the folder where the plot can be stored.
        gt_coords (torch.Tensor):   tensor containing the ground truth coordinates.
        gt_class (torch.Tensor):    tensor containing the ground truth classes.
        plot_gt_color (bool):       if True, ground truth annotations will be plotted in colors of the corresponding class.
        pre_nms (bool):             if True, the coords tensor is assumed to contain predictions before global NMS was applied.
    """

    img_arr = cv2.imread(img_fn)

    for i in range(coords.shape[0]):
        cv2.rectangle(img=img_arr, pt1=(int(round(coords[i, 0].item())), int(round(coords[i, 1].item()))),
                      pt2=(int(round(coords[i, 2].item())), int(round(coords[i, 3].item()))), color=CLASS_COLORS[int(cls[i].item())],
                      thickness=3)

    if gt_coords is not None:
        for i in range(gt_coords.shape[0]):
            gt_color = (0, 255, 0) if not plot_gt_color else CLASS_COLORS[int(gt_class[i].item())]
            cv2.rectangle(img=img_arr, pt1=(int(round(gt_coords[i, 0].item())), int(round(gt_coords[i, 1].item()))),
                          pt2=(int(round(gt_coords[i, 2].item())), int(round(gt_coords[i, 3].item()))), color=gt_color, thickness=3)

    output_ext = "_pre_nms" if pre_nms else ""

    assert cv2.imwrite(f"{output_dir}/{Path(img_fn).stem}{output_ext}.jpg", img_arr), "Plotting failed!"


def read_output_SAHI(out_json_SAHI: str, dataset_json_SAHI: str, class_id2name: dict, output_dir: str, iou_thresh: float = None, min_iou: float = 0.5, ann_file: str = None,
                     ann_format: str = None, box_dims: dict = None, vis_prob: float = -1.0, vis_density: int = math.inf, plot_gt: bool = False, plot_gt_color: bool = False) -> None:
    """
    Given a SAHI output file, make a detections directory that for each image contains a .json file with counts and (if
    also given ground truth annotations) confusion matrices in the shape of a .npy file.
    Arguments:
        out_json_SAHI (str):            path to the SAHI output file.
        dataset_json_SAHI (str):        path to the SAHI dataset file.
        class_id2name (dict):           dictionary mapping class ids to class names.
        output_dir (str):               path to the output directory.
        iou_thresh (float):             threshold to be used during NMS.
        min_iou (float):                minimum IoU for a prediction to be considered a TP.
        ann_file (str):                 path to the file containing the annotations for the images to be processed. The file is
                                        expected to contain a dictionary that maps image filenames to annotations.
        ann_format (string):            format of the annotations.
        box_dims (dict):                dimensions of the bounding boxes used for model training. For cases where
                                        dimensions that differ from what is specified in the annotations were used.
        vis_prob (float):               plotting probability. For each image, a random number between 0 and 1 is drawn,
                                        and if it's below vis_prob, the image in question & the corresponding predictions
                                        will be visualized.
        vis_density (int):              number of objects/animals that will trigger plotting. If the number of predictions
                                        for an image is equal to or exceeds vis_density, the image & predictions will be plotted
                                        regardless of vis_prob.
        plot_gt (bool):                 if True, ground truth annotations will be plotted as well.
        plot_gt_color (bool):           if True, ground truth annotations will be plotted in colors corresponding to the predicted class.
    """

    dets_dir = f"{output_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}"
    Path(dets_dir).mkdir(exist_ok=True)

    if vis_prob > 0 or vis_density < math.inf:
        vis_dir = f"{output_dir}/vis"
        Path(vis_dir).mkdir(exist_ok=True)

    with open(out_json_SAHI, "r") as out, open(dataset_json_SAHI, "r") as data:
        out_SAHI = json.load(out)
        dataset_SAHI = json.load(data)

    if ann_file is not None:
        with open(ann_file, "r") as anns:
            ann_dict = json.load(anns)

    counts_total = {cid: 0 for cid in class_id2name.keys()}

    # for each image in the dataset
    for img in dataset_SAHI["images"]:
        coords = []
        conf = []
        cls = []
        counts = defaultdict(int)
        img_id = img["id"]
        img_path = img["file_name"]
        img_fn = Path(img_path).stem

        # extract and group predictions
        for det in out_SAHI:
            if det["image_id"] == img_id:
                xmin = det["bbox"][0]
                ymin = det["bbox"][1]
                xmax = xmin + det["bbox"][2]
                ymax = ymin + det["bbox"][3]

                cscore = det["score"]
                cat = det["category_id"]
                counts[int(cat)] += 1

                coords.append([xmin, ymin, xmax, ymax])
                conf.append(cscore)
                cls.append(cat)

        coords_t = torch.tensor(coords, dtype=torch.float, device=torch.device("cuda"))
        conf_t = torch.tensor(conf, dtype=torch.float, device=torch.device("cuda"))
        cls_t = torch.tensor(cls, dtype=torch.float, device=torch.device("cuda"))

        with open(f"{dets_dir}/{img_fn}.json", "w") as f:
            json.dump(counts, f, indent=1)

        for cid in counts.keys():
            counts_total[cid] += counts[cid]

        # if annotations are available, make CMs and extract EM
        if ann_file is not None:
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[img_fn], ann_format=ann_format,
                                            device=torch.device("cuda"), box_dims=box_dims)

            cfm_img = ConfusionMatrix(names=class_id2name)
            detections = {"bboxes": coords_t, "conf": conf_t, "cls": cls_t}
            batch = {"bboxes": gt_coords, "cls": gt_cls}

            if iou_thresh is not None:
                cfm_img.process_batch(detections=detections, batch=batch, iou_thres=iou_thresh)
            else:
                cfm_img.process_batch(detections=detections, batch=batch)

            # write confusion matrix to file
            with open(f"{dets_dir}/{img_fn}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)

            # make stats dict
            npr = coords_t.size(dim=0)
            stat = dict(
                conf=torch.zeros(0, device=gt_coords.device),
                pred_cls=torch.zeros(0, device=gt_coords.device),
                tp=torch.zeros(npr, int(round((0.95 - min_iou) / 0.05)) + 1, dtype=torch.bool, device=gt_coords.device),
            )

            nl = gt_cls.size(dim=0)
            stat["target_cls"] = gt_cls

            if npr != 0:
                stat["conf"] = conf_t
                stat["pred_cls"] = cls_t
                # Evaluate
                if nl:
                    iou = box_iou(gt_coords, coords_t)
                    stat["tp"] = match_predictions(cls_t, gt_cls, iou, min_iou=min_iou)
            else:
                if not nl:
                    raise ValueError("No predictions and no labels case is skipped in the original code, but I need a matching file, so I'm adding zero-stats.\n"
                                     "This is going to wrongly reduce the performance metrics. Normally this shouldn't happen as there shouldn't be empty images.\n"
                                     "This error being raised, however, means that there are!!")

            # write to pickle file:
            with open(f"{dets_dir}/{img_fn}_stats.pickle", "wb") as f:
                pickle.dump(stat, f)
        else:
            gt_coords = None
            gt_cls = None

        post_nms = coords_t.shape[0]
        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (post_nms >= vis_density)

        if visualize and post_nms > 0:
            assert not (plot_gt and ann_file is None), "Plotting ground truth requested but no annotation file provided!"
            plot_img_predictions(img_fn=img_path, coords=coords_t, cls=cls_t, output_dir=vis_dir, gt_coords=gt_coords,
                                 gt_class=gt_cls, plot_gt_color=plot_gt_color)

    with open(f"{output_dir}/counts_toal.json", "w") as f:
        json.dump(counts_total, f)


def img_inference(model: str | YOLO, class_id2name: dict, imgs_dir: str, img_files_ext: str, output_dir: str, iou_thresh: float = None,
                  min_iou: float = 0.5, ann_file: str = None, ann_format: str = None, box_dims: dict = None,
                  device: str = "cuda:0", vis_prob: float = -1.0, vis_density: int = math.inf, plot_gt: bool = False,
                  plot_gt_color: bool = False) -> None:
    """
    Perform inference on a directory of images. If ground truth annotations are available, produces confusion matrices and other output
    to compute evaluation metrics.
    Arguments:
        model (str | YOLO):         path to the model file (.pt) to be used for inference, or already constructed model object.
        class_id2name (dict):       dictionary mapping class ids to class names.
        imgs_dir (str):             path to the folder containing the images inference needs to be performed on.
        img_files_ext (str):        Extension of image files. E.g., 'JPG'.
        output_dir (str):           path to the directory the output will be stored in.
        iou_thresh (float):         IoU threshold to apply during NMS. Defaults to None.
        min_iou (float):            minimum IoU for a prediction to be considered a TP. Only relevant if ann_file is provided. Defaults to 0.5.
        ann_file (string):          path to the file containing the annotations for the images to be processed. The file is
                                    expected to contain a dictionary that maps image filenames to annotations.
        ann_format (string):        format of the annotations.
        box_dims (dict):            dimensions of the bounding boxes used for model training. For cases where
                                    dimensions that differ from what is specified in the annotations were used.
        device (str):               device onto which to load the model. Defaults to "cuda:0".
        vis_prob (float):           plotting probability. For each image, a random number between 0 and 1 is drawn,
                                    and if it's below vis_prob, the image in question & the corresponding predictions
                                    will be visualized.
        vis_density (int):          number of objects/animals that will trigger plotting. If the number of predictions
                                    for an image is equal to or exceeds vis_density, the image & predictions will be plotted
                                    regardless of vis_prob.
        plot_gt (bool):             if True, ground truth annotations will be plotted as well.
        plot_gt_color (bool):       if True, ground truth annotations will be plotted in colors corresponding to the predicted class.
    Returns:
        None
    """

    # make output folders
    det_dir = f"{output_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}"
    Path(det_dir).mkdir(exist_ok=True)

    if vis_prob > 0 or vis_density < math.inf:
        vis_dir = f"{output_dir}/vis"
        Path(vis_dir).mkdir(exist_ok=True)

    mdl = YOLO(model).to(device) if isinstance(model, str) else model.to(device)
    assert mdl.task == "detect", "Only box-detection models (task='detect') are supported."

    # read annotations file if available
    if ann_file is not None:
        with open(ann_file, "r") as f:
            ann_dict = json.load(f)

    img_fns = list(Path(imgs_dir).glob(f"*.{img_files_ext}"))

    # for accumulating total counts across images
    counts_sum = {cls_id: 0 for cls_id in class_id2name.keys()}

    # run detection on images
    print("*** Performing inference..")
    predictions = mdl(img_fns, verbose=False)

    print("*** Collecting results...")
    for img_preds in tqdm(predictions):
        img_p = img_preds.path
        data = img_preds.boxes

        coords = data.xyxy
        conf = data.conf.unsqueeze(1)
        cls = data.cls.unsqueeze(1)

        # combine coordinates, confidence and class into one tensor
        preds_t = torch.hstack((coords, conf, cls))

        # If annotations are available, collect evaluation metrics at the image level
        if ann_file is not None:
            gt_coords, gt_cls = load_img_gt(annotations=ann_dict[Path(img_p).stem], ann_format=ann_format,
                                            device=coords.device, box_dims=box_dims)

            # Make confusion matrix
            cfm_img = ConfusionMatrix(names=class_id2name)
            detections = {"bboxes": preds_t[:, :4], "conf": preds_t[:, 4], "cls": preds_t[:, -1]}
            batch = {"bboxes": gt_coords, "cls": gt_cls}
            if iou_thresh is not None:
                cfm_img.process_batch(detections=detections, batch=batch, iou_thres=iou_thresh)
            else:
                cfm_img.process_batch(detections=detections, batch=batch)

            # write confusion matrix to file
            with open(f"{det_dir}/{Path(img_p).stem}_cfm.npy", "wb") as f:
                np.save(f, cfm_img.matrix)

            npr = preds_t.size(dim=0)
            stat = dict(
                conf=torch.zeros(0, device=preds_t.device),
                pred_cls=torch.zeros(0, device=preds_t.device),
                tp=torch.zeros(npr, 10, dtype=torch.bool, device=preds_t.device),
                no_labels=False
            )
            nl = gt_cls.size(dim=0)
            stat["target_cls"] = gt_cls

            if npr != 0:
                stat["conf"] = preds_t[:, 4]
                stat["pred_cls"] = preds_t[:, 5]
                # Evaluate
                if nl:
                    iou = box_iou(box1=gt_coords, box2=preds_t[:, :4])
                    stat["tp"] = match_predictions(pred_classes=preds_t[:, 5], true_classes=gt_cls, iou=iou, min_iou=min_iou)
            else:
                if nl == 0:
                    stat["no_labels"] = True

            # write to pickle file:
            with open(f"{det_dir}/{Path(img_p).stem}_stats.pickle", "wb") as f:
                pickle.dump(stat, f)
        else:
            gt_coords = None
            gt_cls = None

        # get counts post nms
        post_nms = preds_t.shape[0]
        cls_idx_post, counts_post = torch.unique(preds_t[:, -1], return_counts=True)
        counts_post_dict = {}
        for j in range(cls_idx_post.shape[0]):
            counts_post_dict[int(cls_idx_post[j].item())] = int(counts_post[j].item())

        with open(f"{det_dir}/{Path(img_p).stem}.json", "w") as f:
            json.dump(counts_post_dict, f, indent=1)

        # add to count sum
        for class_idx, n in counts_post_dict.items():
            counts_sum[class_idx] += n

        visualize = (random.randint(0, 1000) / 1000 <= vis_prob) or (post_nms >= vis_density)

        if visualize:
            assert not (plot_gt and ann_file is None), "Plotting ground truth requested but no annotation file provided!"
            plot_img_predictions(img_fn=img_p, coords=coords, cls=cls, pre_nms=False, output_dir=vis_dir,
                                 gt_coords=gt_coords, gt_class=gt_cls, plot_gt_color=plot_gt_color)

    # save counts
    with open(f"{Path(det_dir).parent}/counts_total.json", "w") as f:
        json.dump(counts_sum, f, indent=1)


def compute_errors_img_lvl(gt_counts_dir: str, pred_counts_dir: str, class_ids: list, output_dir: str) -> dict:
    """
    Given a directory containing .json files of ground truth counts of an image set, and predicted counts,
    compute the MAE and MSE. Also outputs an excel sheet containing the count differences per image.
    Arguments:
        gt_counts_dir (str):        path to the directory containing the gt counts.
        pred_counts_dir (str):      path to the directory containing the predicted counts.
        class_ids (list):           list of integer class ids.
        output_dir (str):           path to a folder where the output can be stored.
    Returns:
        dictionary containing the MAE and MSE per class.
    """

    # get paths to prediction files
    gt_files = [p for p in Path(gt_counts_dir).glob("*.json")]
    pred_files = [p for p in Path(pred_counts_dir).glob("*.json")]

    # make empty df to collect count differences
    df_cols = ["fn"]
    df_cols.extend([str(cls_id) for cls_id in class_ids])
    df_cols.append("binary")
    df_cols.extend([f"{str(cls_id)}_rel_diff" for cls_id in class_ids])
    df_cols.extend([f"{str(cls_id)}_rel_det" for cls_id in class_ids])
    df_cols.append("binary_rel_diff")
    df_cols.append("binary_rel_det")
    count_diffs_df = pd.DataFrame(columns=df_cols)

    # collect gt and predicted counts to plot later
    gts = {cls_id: [] for cls_id in class_ids}
    gts["binary"] = []
    preds = {cls_id: [] for cls_id in class_ids}
    preds["binary"] = []

    for fn_gt in gt_files:

        # get matching prediction file for a gt file
        fn_pred = [pf for pf in pred_files if fn_gt.stem == pf.stem]

        assert len(fn_pred) == 1, f"Found {len(fn_pred)} matching prediction files for file {fn_gt}. Expected 1."

        fn_pred = fn_pred[0]

        with open(fn_gt, "r") as gt:
            gt_dict = json.load(gt)

        with open(fn_pred, "r") as pred:
            pred_dict = json.load(pred)

        # if no predictions for a class were made, the count number is zero
        gt_dict = {int(k): v for k, v in gt_dict.items()}
        pred_dict = {int(k): v for k, v in pred_dict.items()}
        pred_dict_updated = {cls_id: 0 if cls_id not in pred_dict.keys() else pred_dict[cls_id] for cls_id in class_ids}

        # store predictions in list
        for cls_id in class_ids:
            gts[cls_id].append(gt_dict[cls_id])
            gts["binary"].append(sum(gt_dict.values()))
            preds[cls_id].append(pred_dict_updated[cls_id])
            preds["binary"].append(sum(pred_dict_updated.values()))

        # get abs difference
        diffs = {str(cls_id): [gt_dict[cls_id] - pred_dict_updated[cls_id]] for cls_id in class_ids}
        diffs["binary"] = [sum(gt_dict.values()) - sum(pred_dict_updated.values())]
        diffs["fn"] = [fn_gt.stem]
        # get rel difference
        diffs_rel_diff = {f"{k}_rel_diff": [diffs[str(k)][0] / gt_dict[k]] if gt_dict[k] != 0 else diffs[str(k)][0] for k in gt_dict.keys()}
        diffs_rel_det = {f"{k}_rel_det": [pred_dict_updated[k] / gt_dict[k]] if gt_dict[k] != 0 else diffs[str(k)][0] for k in gt_dict.keys()}
        diffs_rel_diff["binary_rel_diff"] = [diffs["binary"][0] / sum(gt_dict.values())] if sum(gt_dict.values()) != 0 else diffs["binary"][0]
        diffs_rel_det["binary_rel_det"] = [sum(pred_dict_updated.values()) / sum(gt_dict.values())] if sum(gt_dict.values()) != 0 else diffs["binary"][0]

        # merge dicts
        diffs.update(diffs_rel_diff)
        diffs.update(diffs_rel_det)
        # put into df
        row_df = pd.DataFrame(diffs)
        count_diffs_df = pd.concat([count_diffs_df, row_df], ignore_index=True)

    # get stats per class
    summary_dict = {cls_id: {"MAE": count_diffs_df[f"{cls_id}"].abs().sum() / count_diffs_df.shape[0],
                             "ME":  count_diffs_df[f"{cls_id}"].sum() / count_diffs_df.shape[0],
                             "MSE": (count_diffs_df[f"{cls_id}"] ** 2).sum() / count_diffs_df.shape[0],
                             "MARE": count_diffs_df[f"{cls_id}_rel_diff"].abs().sum() / count_diffs_df.shape[0],
                             "MRE": count_diffs_df[f"{cls_id}_rel_det"].abs().sum() / count_diffs_df.shape[0]} for cls_id in class_ids}

    # get binary stats
    summary_dict["binary"] = {"MAE": count_diffs_df["binary"].abs().sum() / count_diffs_df.shape[0],
                              "ME":  count_diffs_df["binary"].sum() / count_diffs_df.shape[0],
                              "MSE": (count_diffs_df["binary"] ** 2).sum() / count_diffs_df.shape[0],
                              "MARE": count_diffs_df["binary_rel_diff"].abs().sum() / count_diffs_df.shape[0],
                              "MRE": count_diffs_df["binary_rel_det"].abs().sum() / count_diffs_df.shape[0]}

    # plot counts
    for k in gts.keys():
        plt.scatter(gts[k], preds[k], c="purple", s=5)
        plt.xlabel("True Count")
        plt.ylabel("Predicted Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/counts_gt_pred_{k}.png")
        plt.close()

    # output
    with open(f"{output_dir}/errors_img_lvl.json", "w") as f:
        json.dump(summary_dict, f, indent=1)

    count_diffs_df.to_excel(f"{output_dir}/count_diffs_img_lvl.xlsx")

    return summary_dict


def compute_em_img_lvl(preds_dir: str, class_id2name: dict, task: str, output_dir: str) -> dict:
    """
    Given a directory containing .npy files representing confusion matrices for an image set, compute
    a set of evaluation metrics.
    Arguments:
        preds_dir (str):            path to the directory containing the .npy files.
        class_id2name (dict):       dict mapping class ids to class names.
        task (str):                 task string. Only "detect" is supported in this release.
        output_dir (str):           path to a folder where the output can be stored.
    Returns:
        dictionary containing evaluation metrics.
    """
    em_dict = {cat: {} for cat in class_id2name.keys()}

    # get paths to cm files
    cm_files = [p for p in Path(preds_dir).glob("*.npy")]
    cm_final = np.load(cm_files[0])
    cm_files = cm_files[1:]

    # aggregate single file cms
    for f in cm_files:
        cm_final += np.load(f)

    # get paths to pickle files
    stat_files = [p for p in Path(preds_dir).glob("*.pickle")]
    stats_glob = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}

    # aggregate stat dicts
    for sf in stat_files:
        with open(sf, "rb") as f:
            stat_dict = pickle.load(f)

        if stat_dict.get("no_labels"):
            """
            in 'img_inference' the 'no_labels' attribute is set only if there are no labels AND no predictions.
            The logic is copied from the ultralytics validator code. Note that this is done only for the
            evaluation metrics and not for the confusion matrix.
            """
            continue
        else:
            for k in stats_glob.keys():
                if k == "target_img":
                    t = torch.unique(stat_dict["target_cls"]).cpu()
                else:
                    t = stat_dict[k].cpu()

                if len(t.shape) == 1 and k == "tp":
                    t.unsqueeze_(0)

                if k == "target_cls" and t.shape[0] == 0:
                    """I do this to prevent errors when there are no targets but predictions. In
                     my understanding the target_cls vector is not used for metrics calculations
                     (cf. ap_per_class method in ultralytics' metrics.py). So this
                     shouldn't affect the results."""
                    t = torch.zeros((1,), dtype=torch.float32).cpu()

                stats_glob[k].append(t)

    found_stuff = any(t.any().item() for t in stats_glob["tp"])
    """
    initialize Metrics. I need to account for the possibility that different versions of the ultralytics
    repo may be used, which construct DetMetrics differently. Hence the try/except block.
    """
    try:
        metrics = DetMetrics(names=class_id2name, save_dir=Path(output_dir), plot=True)
        if found_stuff:
            # concatenate stats
            stats_final = {k: torch.cat(v, 0).numpy() for k, v in stats_glob.items()}  # to numpy
            metrics.process(tp=stats_final["tp"],
                            conf=stats_final["conf"],
                            pred_cls=stats_final["pred_cls"],
                            target_cls=stats_final["target_cls"])
    except:
        metrics = DetMetrics(names=class_id2name)
        metrics.stats = stats_glob
        if found_stuff:
            metrics.process(save_dir=Path(output_dir), plot=True)

    for cls_id in range(cm_final.shape[0] - 1):
        # confusion for each class = 1 - (TP / FP (without background))
        confusion = (1 - (cm_final[cls_id, cls_id] / (sum(cm_final[cls_id, :-1]) + 1e-6))) * 100
        precision = (cm_final[cls_id, cls_id] / (sum(cm_final[cls_id]) + 1e-6)) * 100
        recall = (cm_final[cls_id, cls_id] / (sum(cm_final[:, cls_id]) + 1e-6)) * 100

        em_dict[cls_id]["confusion"] = confusion
        em_dict[cls_id]["precision"] = precision
        em_dict[cls_id]["recall"] = recall
        em_dict[cls_id]["f1"] = ((2 * precision * recall) / (precision + recall + 1e-6))

        # collect YOLO metrics
        for i, k in enumerate(metrics.keys):
            if found_stuff:
                em_dict[cls_id][k] = metrics.class_result(cls_id)[i]
            else:
                em_dict[cls_id][k] = 0.0

    binary = {}
    precision_binary = (np.trace(cm_final) / (np.trace(cm_final) + sum(cm_final[:, -1]) + 1e-6)) * 100
    recall_binary = (np.trace(cm_final) / (np.trace(cm_final) + sum(cm_final[-1]) + 1e-6)) * 100

    binary["confusion"] = sum([em_dict[cls_id]["confusion"] for cls_id in em_dict.keys()]) / len(em_dict.keys())
    binary["precision"] = precision_binary
    binary["recall"] = recall_binary
    binary["f1"] = ((2 * precision_binary * recall_binary) / (precision_binary + recall_binary + 1e-6))
    # Add YOLO metrics
    for i, k in enumerate(metrics.keys):
        binary[k] = metrics.mean_results()[i]

    em_dict["binary"] = binary

    with open(f"{output_dir}/em.json", "w") as f:
        json.dump(em_dict, f, indent=1)

    return em_dict


def eval_single_model(mdl: str, task: str, data_dir: str, img_format: str, datasets2test: list, eval_split: str, use_tiling: bool,
                      class_id2name: dict, radii: dict | None, dor: float | None, ann_format: str, box_dims: dict | None,
                      patch_dims: dict | None, patch_overlap: float | None, output_dir: str, min_iou: float = 0.5, inf_params: dict = {}) -> dict:
    """
    Evaluate the performance of a model across all client datasets. This includes the baseline, trained on all client
    data.
    The structure of data_dir must meet the following requirements:
    --data_dir
        --c1
            --eval_split
        --c2
            --eval_split
        ...
    with one subdirectory for each client whose datasets is part of the evaluation.
    Each eval_split subdirectory is further expected to contain an annotation json file and ground truth files for counting.
    Arguments:
        mdl (str):                      path to the model to be evaluated.
        task (str):                     task of the model being evaluated. Only "detect" is supported in this release.
        data_dir (str):                 directory containing all client splits.
        img_format (str):               file extension format of images (e.g., 'png', 'jpg').
        datasets2test (list):           list of local datasets the client under investigation is to be tested on.
        eval_split (str):               dataset split to evaluate ('val' or 'test').
        use_tiling (bool):              if True, tiled inference is performed.
        class_id2name (dict):           mapping of class IDs to class names.
        radii (dict | None):            unused; retained for signature compatibility.
        dor (float | None):             unused; retained for signature compatibility.
        ann_format (str):               annotation format (e.g., "BX_WH").
        box_dims (dict | None):         bounding box dimensions, if working with pseudo-boxes.
        patch_dims (dict | None):       dimensions of image patches for inference.
        patch_overlap (float | None):   overlap ratio between adjacent patches.
        output_dir (str):               directory where temporary outputs can be stored.
        min_iou (float):                minimum IoU for a prediction to be considered a TP.
        inf_params (dict):              dictionary containing inference parameters for SAHI predictions.
    Returns:
        dict containing the performance across client datasets of the client under investigation.
        Structure:{c1: {"count_err": output of compute_errors_img_lvl,
                        "em": output of compute_em_img_lvl}
                   c2: ...
                   c3: ...
                   ...}
    """
    assert type(mdl) == str, "Need path to the .pt file for SAHI inference! For now, passing a loaded model is not supported."
    assert task == "detect", "This release supports box detection only (task='detect')."

    results_dict = {c: {} for c in datasets2test}

    for td in datasets2test:
        eval_dir = f"{data_dir}/{td}/{eval_split}"
        ann_file = f"{eval_dir}/{NAMES_DIRECTORIES_FILES['annotations_json']}"
        temp_dir = f"{output_dir}/inf_temp"

        if use_tiling:
            predict(
                model_path=mdl,
                model_type="ultralytics",
                model_device="cuda:0",
                source=eval_dir,
                slice_height=patch_dims["height"],
                slice_width=patch_dims["width"],
                overlap_height_ratio=patch_overlap,
                overlap_width_ratio=patch_overlap,
                dataset_json_path=f"{eval_dir}/{NAMES_DIRECTORIES_FILES['dataset_file_SAHI']}",
                project=output_dir,
                name=temp_dir,
                novisual=True,
                verbose=0,
                FL_args=inf_params
            )

            read_output_SAHI(out_json_SAHI=f"{temp_dir}/result.json", dataset_json_SAHI=f"{eval_dir}/{NAMES_DIRECTORIES_FILES['dataset_file_SAHI']}",
                             class_id2name=class_id2name, ann_file=ann_file, ann_format=ann_format, output_dir=temp_dir, min_iou=min_iou, box_dims=box_dims)
        else:
            Path(temp_dir).mkdir(exist_ok=True)
            img_inference(model=mdl,
                          class_id2name=class_id2name,
                          imgs_dir=eval_dir,
                          img_files_ext=img_format,
                          ann_file=ann_file,
                          ann_format=ann_format,
                          output_dir=temp_dir,
                          min_iou=min_iou,
                          box_dims=box_dims)

        count_err = compute_errors_img_lvl(gt_counts_dir=f"{eval_dir}/{NAMES_DIRECTORIES_FILES['count_directory']}",
                                           pred_counts_dir=f"{temp_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}",
                                           class_ids=list(class_id2name.keys()),
                                           output_dir=temp_dir)
        em = compute_em_img_lvl(preds_dir=f"{temp_dir}/{NAMES_DIRECTORIES_FILES['inference_output_dir']}",
                                class_id2name=class_id2name, task=task,
                                output_dir=temp_dir)

        results_dict[td]["count_err"] = count_err
        results_dict[td]["em"] = em

        shutil.rmtree(temp_dir)

    return results_dict


def get_single_metric_single_client(results: dict, metric: str, cats: list, client: str) -> dict:
    """
    Retrieve values of a single evaluation metric for a single client across all rounds.
    Arguments:
        results (dict):     dictionary containing the results of the FL training.
        metric (str):       metric to extract.
        cats (list):        list of categories to extract the metric for.
        client (str):       client to extract the metric for.
    Returns:
        dict:              dictionary containing the metric for each category.
    """
    rounds = sorted(results.keys(), key=lambda x: int(x.replace("round", "")))

    out = {cat: {"global": []} for cat in cats}

    for c in cats:
        for rnd in rounds:
            if metric in results[rnd][client]["count_err"][c].keys():
                out[c]["global"].append(results[rnd][client]["count_err"][c][metric])
            else:
                out[c]["global"].append(results[rnd][client]["em"][c][metric])

    return out
