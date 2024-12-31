import os, sys
import shutil
import socket
import getpass
import copy
import numpy as np
from modules.box_utils import nms
import datetime
import logging
import torch
import pdb
import torchvision


# from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
class BufferList(torch.nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def setup_logger(args):
    """
    Sets up the logging.
    """
    log_file_name = "{:s}/{:s}-{date:%m-%d-%Hx}.log".format(
        args.SAVE_ROOT, args.MODE, date=datetime.datetime.now()
    )
    args.log_dir = "logs/" + args.exp_name + "/"
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    added_log_file = "{}{}-{date:%m-%d-%Hx}.log".format(
        args.log_dir, args.MODE, date=datetime.datetime.now()
    )

    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=_FORMAT, stream=sys.stdout)
    logging.getLogger().addHandler(logging.FileHandler(log_file_name, mode="a"))
    # logging.getLogger().addHandler(logging.FileHandler(added_log_file, mode='a'))


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def copy_source(source_dir):
    if not os.path.isdir(source_dir):
        os.system("mkdir -p " + source_dir)

    for dirpath, dirs, files in os.walk("./", topdown=True):
        for file in files:
            if file.endswith(".py"):  # fnmatch.filter(files, filepattern):
                shutil.copy2(os.path.join(dirpath, file), source_dir)


def set_args(args):
    args.MAX_SIZE = int(args.MIN_SIZE * 1.35)
    args.MILESTONES = [int(val) for val in args.MILESTONES.split(",")]
    # args.GAMMAS = [float(val) for val in args.GAMMAS.split(',')]
    args.EVAL_EPOCHS = [int(val) for val in args.EVAL_EPOCHS.split(",")]

    args.TRAIN_SUBSETS = [val for val in args.TRAIN_SUBSETS.split(",") if len(val) > 1]
    args.VAL_SUBSETS = [val for val in args.VAL_SUBSETS.split(",") if len(val) > 1]
    args.TEST_SUBSETS = [val for val in args.TEST_SUBSETS.split(",") if len(val) > 1]
    args.TUBES_EVAL_THRESHS = [
        float(val) for val in args.TUBES_EVAL_THRESHS.split(",") if len(val) > 0.0001
    ]
    args.model_subtype = args.MODEL_TYPE.split("-")[0]
    ## check if subsets are okay
    possible_subets = ["test", "train", "val"]
    for idx in range(1, 4):
        possible_subets.append("train_" + str(idx))
        possible_subets.append("val_" + str(idx))

    if len(args.VAL_SUBSETS) < 1 and args.DATASET == "road":
        args.VAL_SUBSETS = [ss.replace("train", "val") for ss in args.TRAIN_SUBSETS]
    if len(args.TEST_SUBSETS) < 1:
        # args.TEST_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
        args.TEST_SUBSETS = args.VAL_SUBSETS

    for subsets in [args.TRAIN_SUBSETS, args.VAL_SUBSETS, args.TEST_SUBSETS]:
        for subset in subsets:
            assert (
                subset in possible_subets
            ), "subest should from one of these " + "".join(possible_subets)

    args.DATASET = args.DATASET.lower()
    args.ARCH = args.ARCH.lower()

    args.MEANS = [0.485, 0.456, 0.406]
    args.STDS = [0.229, 0.224, 0.225]

    username = getpass.getuser()
    hostname = socket.gethostname()
    args.hostname = hostname
    args.user = username

    args.model_init = "kinetics"

    args.MODEL_PATH = (
        args.MODEL_PATH[:-1] if args.MODEL_PATH.endswith("/") else args.MODEL_PATH
    )

    assert args.MODEL_PATH.endswith("kinetics-pt") or args.MODEL_PATH.endswith(
        "imagenet-pt"
    )
    args.model_init = (
        "imagenet" if args.MODEL_PATH.endswith("imagenet-pt") else "kinetics"
    )

    if args.MODEL_PATH == "imagenet":
        args.MODEL_PATH = os.path.join(args.MODEL_PATH, args.ARCH + ".pth")
    else:
        args.MODEL_PATH = os.path.join(
            args.MODEL_PATH, args.ARCH + args.MODEL_TYPE + ".pth"
        )

    print(
        "Your working directories are::\nLOAD::> ",
        args.DATA_ROOT,
        "\nSAVE::> ",
        args.SAVE_ROOT,
    )
    print("Your model will be initialized using", args.MODEL_PATH)

    return args


def create_exp_name(args):
    """Create name of experiment using training parameters"""
    splits = "".join([split[0] + split[-1] for split in args.TRAIN_SUBSETS])
    args.exp_name = (
        "{:s}{:s}{:d}-P{:s}-b{:0d}s{:d}x{:d}x{:d}-{:s}{:s}-h{:d}x{:d}x{:d}".format(
            args.ARCH,
            args.MODEL_TYPE,
            args.MIN_SIZE,
            args.model_init,
            args.BATCH_SIZE,
            args.SEQ_LEN,
            args.MIN_SEQ_STEP,
            args.MAX_SEQ_STEP,
            args.DATASET,
            splits,
            args.HEAD_LAYERS,
            args.CLS_HEAD_TIME_SIZE,
            args.REG_HEAD_TIME_SIZE,
        )
    )

    args.SAVE_ROOT += args.DATASET + "/"
    args.SAVE_ROOT = args.SAVE_ROOT + "cache/" + args.exp_name + "/"
    if not os.path.isdir(args.SAVE_ROOT):
        print("Create: ", args.SAVE_ROOT)
        os.makedirs(args.SAVE_ROOT)

    return args


# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") > -1:
        m.eval()
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def get_individual_labels(gt_boxes, tgt_labels):
    # print(gt_boxes.shape, tgt_labels.shape)
    new_gts = np.zeros((gt_boxes.shape[0] * 20, 5))
    ccc = 0
    for n in range(tgt_labels.shape[0]):
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n, t] > 0:
                new_gts[ccc, :4] = gt_boxes[n, :]
                new_gts[ccc, 4] = t
                ccc += 1
    return new_gts[:ccc, :]


def get_individual_location_labels(gt_boxes, tgt_labels):
    return [gt_boxes, tgt_labels]


def filter_detections(args, scores, decoded_boxes_batch):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.asarray([])

    boxes = decoded_boxes_batch[c_mask, :].view(-1, 4)
    ids, counts = nms(
        boxes, scores, args.NMS_THRESH, args.TOPK * 5
    )  # idsn - ids after nms
    scores = scores[ids[: min(args.TOPK, counts)]].cpu().numpy()
    boxes = boxes[ids[: min(args.TOPK, counts)]].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

    return cls_dets


def filter_detections_for_tubing(args, scores, decoded_boxes_batch, confidences):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.zeros((0, 200))

    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences = confidences[c_mask, :].clone().view(-1, numc)

    max_k = min(args.TOPK * 60, scores.shape[0])
    ids, counts = nms(boxes, scores, args.NMS_THRESH, max_k)  # idsn - ids after nms
    scores = scores[ids[: min(args.TOPK, counts)]].cpu().numpy()
    boxes = boxes[ids[: min(args.TOPK, counts)], :].cpu().numpy()
    confidences = confidences[ids[: min(args.TOPK, counts)], :].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences[:, 1:])).astype(np.float32)
    # print(save_data.shape)
    return save_data


def filter_detections_for_dumping(args, scores, decoded_boxes_batch, confidences):
    """
    Filters detections for dumping.

    Parameters:
    args (object): Contains various parameters for the function.
    scores (torch.Tensor): Tensor containing scores of detections.
    decoded_boxes_batch (torch.Tensor): Tensor containing bounding boxes of detections.
    confidences (torch.Tensor): Tensor containing confidences of detections.

    Returns:
    tuple: A tuple containing filtered detections and their corresponding confidences.
    """
    c_mask = scores.gt(args.GEN_CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.zeros((0, 5)), np.zeros((0, 200))

    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences = confidences[c_mask, :].clone().view(-1, numc)

    max_k = min(args.GEN_TOPK * 500, scores.shape[0])
    ids, counts = nms(boxes, scores, args.GEN_NMS, max_k)  # idsn - ids after nms
    scores = scores[ids[: min(args.GEN_TOPK, counts)]].cpu().detach().numpy()
    boxes = boxes[ids[: min(args.GEN_TOPK, counts)], :].cpu().detach().numpy()
    confidences = (
        confidences[ids[: min(args.GEN_TOPK, counts)], :].cpu().detach().numpy()
    )
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences[:, 1:])).astype(np.float32)
    return cls_dets, save_data


def make_joint_probs_from_marginals(frame_dets, childs, num_classes_list, start_id=4):

    # pdb.set_trace()

    add_list = copy.deepcopy(num_classes_list[:3])
    add_list[0] = start_id + 1
    add_list[1] = add_list[0] + add_list[1]
    add_list[2] = add_list[1] + add_list[2]
    # for ind in range(frame_dets.shape[0]):
    for nlt, ltype in enumerate(["duplex", "triplet"]):
        lchilds = childs[ltype + "_childs"]
        lstart = start_id
        for num in num_classes_list[: 4 + nlt]:
            lstart += num

        for c in range(num_classes_list[4 + nlt]):
            tmp_scores = []
            for chid, ch in enumerate(lchilds[c]):
                if len(tmp_scores) < 1:
                    tmp_scores = copy.deepcopy(frame_dets[:, add_list[chid] + ch])
                else:
                    tmp_scores *= frame_dets[:, add_list[chid] + ch]
            frame_dets[:, lstart + c] = tmp_scores

    return frame_dets


def eval_strings():
    return [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ",
    ]


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args(parser):
    parser.add_argument("--DATA_ROOT", type=str, default="/workspace/")
    parser.add_argument(
        "--SAVE_ROOT",
        type=str,
        default="/workspace/road/cache/resnet50I3D512-Pkinetics-b4s8x1x1-roadt3-h3x3x3/",
    )
    parser.add_argument("--MODEL_PATH", type=str, default="/workspace/kinetics-pt/")

    parser.add_argument("--ANNO_ROOT", type=str, default="")
    parser.add_argument(
        "--MODE",
        default="gen_dets",
        help="MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes",
    )
    # Name of backbone network, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported
    parser.add_argument("--ARCH", default="resnet50", type=str, help=" base arch")
    parser.add_argument("--MODEL_TYPE", default="I3D", type=str, help=" base model")
    parser.add_argument("--model_subtype", default="I3D", type=str, help=" sub model")
    parser.add_argument(
        "--ANCHOR_TYPE",
        default="RETINA",
        type=str,
        help="type of anchors to be used in model",
    )

    parser.add_argument("--SEQ_LEN", default=8, type=int, help="NUmber of input frames")
    parser.add_argument(
        "--TEST_SEQ_LEN", default=8, type=int, help="NUmber of input frames"
    )
    parser.add_argument(
        "--MIN_SEQ_STEP",
        default=1,
        type=int,
        help="DIFFERENCE of gap between the frames of sequence",
    )
    parser.add_argument(
        "--MAX_SEQ_STEP",
        default=1,
        type=int,
        help="DIFFERENCE of gap between the frames of sequence",
    )
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    # parser.add_argument('--MULIT_SCALE', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument(
        "--HEAD_LAYERS",
        default=3,
        type=int,
        help="0 mean no shareding more than 0 means shareing",
    )
    parser.add_argument(
        "--NUM_FEATURE_MAPS",
        default=5,
        type=int,
        help="0 mean no shareding more than 0 means shareing",
    )
    parser.add_argument(
        "--CLS_HEAD_TIME_SIZE",
        default=3,
        type=int,
        help="Temporal kernel size of classification head",
    )
    parser.add_argument(
        "--REG_HEAD_TIME_SIZE",
        default=3,
        type=int,
        help="Temporal kernel size of regression head",
    )

    #  Name of the dataset only voc or coco are supported
    parser.add_argument(
        "--DATASET", default="road", type=str, help="dataset being used"
    )
    parser.add_argument(
        "--TRAIN_SUBSETS",
        default="train_3,",
        type=str,
        help="Training SUBSETS seprated by ,",
    )
    parser.add_argument(
        "--VAL_SUBSETS", default="", type=str, help="Validation SUBSETS seprated by ,"
    )
    parser.add_argument(
        "--TEST_SUBSETS", default="", type=str, help="Testing SUBSETS seprated by ,"
    )
    # Input size of image only 600 is supprted at the moment
    parser.add_argument("--MIN_SIZE", default=512, type=int, help="Input Size for FPN")

    #  data loading argumnets
    parser.add_argument(
        "-b", "--BATCH_SIZE", default=4, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--TEST_BATCH_SIZE", default=1, type=int, help="Batch size for testing"
    )
    # Number of worker to load data in parllel
    parser.add_argument(
        "--NUM_WORKERS",
        "-j",
        default=8,
        type=int,
        help="Number of workers used in dataloading",
    )
    # optimiser hyperparameters
    parser.add_argument("--OPTIM", default="SGD", type=str, help="Optimiser type")
    parser.add_argument("--RESUME", default=0, type=int, help="Resume from given epoch")
    parser.add_argument(
        "--MAX_EPOCHS", default=30, type=int, help="Number of training epoc"
    )
    parser.add_argument(
        "-l",
        "--LR",
        "--learning-rate",
        default=0.004225,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument("--MOMENTUM", default=0.9, type=float, help="momentum")
    parser.add_argument(
        "--MILESTONES", default="20,25", type=str, help="Chnage the lr @"
    )
    parser.add_argument("--GAMMA", default=0.1, type=float, help="Gamma update for SGD")
    parser.add_argument(
        "--WEIGHT_DECAY", default=1e-4, type=float, help="Weight decay for SGD"
    )

    # Freeze layers or not
    parser.add_argument(
        "--FBN",
        "--FREEZE_BN",
        default=True,
        type=str2bool,
        help="freeze bn layers if true or else keep updating bn layers",
    )
    parser.add_argument(
        "--FREEZE_UPTO",
        default=1,
        type=int,
        help="layer group number in ResNet up to which needs to be frozen",
    )

    # Loss function matching threshold
    parser.add_argument(
        "--POSTIVE_THRESHOLD",
        default=0.5,
        type=float,
        help="Min threshold for Jaccard index for matching",
    )
    parser.add_argument(
        "--NEGTIVE_THRESHOLD",
        default=0.4,
        type=float,
        help="Max threshold Jaccard index for matching",
    )
    # Evaluation hyperparameters
    parser.add_argument(
        "--EVAL_EPOCHS",
        default="30",
        type=str,
        help="eval epochs to test network on these epoch checkpoints usually the last epoch is used",
    )
    parser.add_argument(
        "--VAL_STEP",
        default=1,
        type=int,
        help="Number of training epoch before evaluation",
    )
    parser.add_argument(
        "--IOU_THRESH",
        default=0.5,
        type=float,
        help="Evaluation threshold for validation and for frame-wise mAP",
    )
    parser.add_argument(
        "--CONF_THRESH",
        default=0.025,
        type=float,
        help="Confidence threshold for to remove detection below given number",
    )
    parser.add_argument(
        "--NMS_THRESH",
        default=0.5,
        type=float,
        help="NMS threshold to apply nms at the time of validation",
    )
    parser.add_argument(
        "--TOPK", default=10, type=int, help="topk detection to keep for evaluation"
    )
    parser.add_argument(
        "--GEN_CONF_THRESH",
        default=0.05,
        type=float,
        help="Confidence threshold at the time of generation and dumping",
    )
    parser.add_argument(
        "--GEN_TOPK", default=100, type=int, help="topk at the time of generation"
    )
    parser.add_argument(
        "--GEN_NMS", default=0.5, type=float, help="NMS at the time of generation"
    )
    parser.add_argument(
        "--CLASSWISE_NMS",
        default=False,
        type=str2bool,
        help="apply classwise NMS/no tested properly",
    )
    parser.add_argument(
        "--JOINT_4M_MARGINALS",
        default=False,
        type=str2bool,
        help="generate score of joints i.e. duplexes or triplet by marginals like agents and actions scores",
    )

    ## paths hyper parameters
    parser.add_argument(
        "--COMPUTE_PATHS",
        default=False,
        type=str2bool,
        help=" COMPUTE_PATHS if set true then it overwrite existing ones",
    )
    parser.add_argument(
        "--PATHS_IOUTH",
        default=0.5,
        type=float,
        help="Iou threshold for building paths to limit neighborhood search",
    )
    parser.add_argument(
        "--PATHS_COST_TYPE",
        default="score",
        type=str,
        help="cost function type to use for matching, other options are scoreiou, iou",
    )
    parser.add_argument(
        "--PATHS_JUMP_GAP",
        default=4,
        type=int,
        help="GAP allowed for a tube to be kept alive after no matching detection found",
    )
    parser.add_argument(
        "--PATHS_MIN_LEN", default=6, type=int, help="minimum length of generated path"
    )
    parser.add_argument(
        "--PATHS_MINSCORE",
        default=0.1,
        type=float,
        help="minimum score a path should have over its length",
    )

    ## paths hyper parameters
    parser.add_argument(
        "--COMPUTE_TUBES",
        default=False,
        type=str2bool,
        help="if set true then it overwrite existing tubes",
    )
    parser.add_argument(
        "--TUBES_ALPHA",
        default=0,
        type=float,
        help="alpha cost for changeing the label",
    )
    parser.add_argument(
        "--TRIM_METHOD",
        default="none",
        type=str,
        help="other one is indiv which works for UCF24",
    )
    parser.add_argument(
        "--TUBES_TOPK",
        default=10,
        type=int,
        help="Number of labels to assign for a tube",
    )
    parser.add_argument(
        "--TUBES_MINLEN", default=5, type=int, help="minimum length of a tube"
    )
    parser.add_argument(
        "--TUBES_EVAL_THRESHS",
        default="0.2,0.5",
        type=str,
        help="evaluation threshold for checking tube overlap at evaluation time, one can provide as many as one wants",
    )

    ###
    parser.add_argument(
        "--LOG_START",
        default=10,
        type=int,
        help="start loging after k steps for text/tensorboard",
    )
    parser.add_argument(
        "--LOG_STEP",
        default=10,
        type=int,
        help="Log every k steps for text/tensorboard",
    )
    parser.add_argument(
        "--TENSORBOARD",
        default=1,
        type=str2bool,
        help="Use tensorboard for loss/evalaution visualization",
    )

    # Program arguments
    parser.add_argument(
        "--MAN_SEED", default=123, type=int, help="manualseed for reproduction"
    )
    parser.add_argument(
        "--MULTI_GPUS",
        default=True,
        type=str2bool,
        help="If  more than 0 then use all visible GPUs by default only one GPU used ",
    )

    args, unknown = parser.parse_known_args()

    if args.MODE == "train":
        args.TEST_SEQ_LEN = args.SEQ_LEN
    else:
        args.SEQ_LEN = args.TEST_SEQ_LEN

    args.SEQ_LEN = args.TEST_SEQ_LEN
    args.MAX_SEQ_STEP = 1
    args.SUBSETS = args.TEST_SUBSETS
    args.skip_beggning = 0
    args.skip_ending = 0
    if args.MODEL_TYPE == "I3D" or "SlowFast":
        args.skip_beggning = 2
        args.skip_ending = 2
    elif args.MODEL_TYPE != "C2D":
        args.skip_beggning = 2

    return args, unknown
