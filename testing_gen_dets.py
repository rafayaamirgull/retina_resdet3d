import torch
import numpy as np
import argparse
from models.retinanet import build_retinanet
from modules import utils
from torchvision import transforms
import data.transforms as vtf
from data import VideoDataset
import cv2
import os
from PIL import Image
import time

parser = argparse.ArgumentParser(
    description="Training single stage FPN with OHEM, resnet as backbone"
)

# Use CUDA_VISIBLE_DEVICES=0,1,4,6 to select GPUs to use

## Parse arguments
args, _ = utils.get_args(parser)
args = utils.set_args(args)  # set directories and SUBSETS fo datasets
args.MULTI_GPUS = False if args.BATCH_SIZE == 1 else args.MULTI_GPUS

full_test = True  # args.MODE != 'train'
skip_step = args.SEQ_LEN - args.skip_beggning

val_transform = transforms.Compose(
    [
        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
        vtf.ToTensorStack(),
        vtf.Normalize(mean=args.MEANS, std=args.STDS),
    ]
)

val_dataset = VideoDataset(
    args,
    train=False,
    transform=val_transform,
    skip_step=skip_step,
    full_test=full_test,
)

args.num_classes = val_dataset.num_classes
# one for objectness
args.label_types = val_dataset.label_types
args.num_label_types = val_dataset.num_label_types
args.all_classes = val_dataset.all_classes
args.num_classes_list = val_dataset.num_classes_list
args.num_ego_classes = val_dataset.num_ego_classes
args.ego_classes = val_dataset.ego_classes
args.head_size = 256

net = build_retinanet(args).cuda()
if args.MULTI_GPUS:
    # logger.info("\nLets do dataparallel\n")
    net = torch.nn.DataParallel(net)

for epoch in args.EVAL_EPOCHS:
    args.MODEL_PATH = args.SAVE_ROOT + "model_{:06d}.pth".format(epoch)
    net.eval()
    net.load_state_dict(torch.load(args.MODEL_PATH))
    net.eval()  # switch net to evaluation mode


base_path = "/workspace/road/rgb-images"
event = "le_creusot_recorded_video_testing"
im_list = [
    "00710.jpg",
    "00711.jpg",
    "00712.jpg",
    "00713.jpg",
    "00714.jpg",
    "00715.jpg",
    "00716.jpg",
    "00717.jpg",
]

imgs_paths = [os.path.join(base_path, event, im) for im in im_list]
print(imgs_paths)

images = [Image.open(img_path).convert("RGB") for img_path in imgs_paths]
inp_x, inp_y = images[0].size
images = val_transform(images).cuda(0, non_blocking=True)[None, :]

torch.cuda.synchronize()
activation = torch.nn.Sigmoid().cuda()
decoded_boxes, confidences, ego_preds = net(images)
confidence = activation(confidences)
seq_len = ego_preds.shape[1]

ratio_factor_x = inp_x / args.MAX_SIZE
ratio_factor_y = inp_y / args.MIN_SIZE

for frame_number in range(len(imgs_paths)):
    decoded_boxes_batch = decoded_boxes[0, frame_number]
    confidence_batch = confidence[0, frame_number]
    scores = confidence_batch[:, 0].squeeze().clone()

    _, save_data = utils.filter_detections_for_dumping(
        args, scores, decoded_boxes_batch, confidence_batch
    )
    del scores, confidence_batch, decoded_boxes_batch
    frame = cv2.imread(imgs_paths[frame_number])
    for det in save_data:
        bbox = det[:4]  # x1, y1, x2, y2
        score = det[4]
        if score >= 0.5:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1, x2, y2 = (
                int(x1 * ratio_factor_x),
                int(y1 * ratio_factor_x),
                int(x2 * ratio_factor_x),
                int(y2 * ratio_factor_x),
            )
            COLORS = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color=COLORS, thickness=2
            )  # random unique list(COLORS)

    del save_data
    frame_name = "/workspace/testme_" + str(frame_number) + ".jpg"
    cv2.imwrite(frame_name, frame)
    print("Saved detections to", frame_name)


del net, images, decoded_boxes, confidences, ego_preds, confidence, activation

print("EXITING...peacefully")

torch.cuda.empty_cache()
time.sleep(10)
