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

import os

base_path = "/workspace/road/rgb-images"
event = "le_creusot_recorded_video_testing"
im_list = sorted(os.listdir(os.path.join(base_path, event)))
batch_size = 8
batches = [im_list[i : i + batch_size] for i in range(0, len(im_list), batch_size)]

inp_x, inp_y = 1920, 1080
ratio_factor_x = inp_x / args.MAX_SIZE
ratio_factor_y = inp_y / args.MIN_SIZE

for batch in batches:
    imgs_paths = [os.path.join(base_path, event, im) for im in batch]
    print(imgs_paths)
    images = val_transform(
        [Image.open(img_path).convert("RGB") for img_path in imgs_paths]
    ).cuda(0, non_blocking=True)[None, :]
    torch.cuda.synchronize()
    activation = torch.nn.Sigmoid().cuda()

    decoded_boxes, confidences, ego_preds = net(images)
    confidence = activation(confidences)

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

        frame_name = "/workspace/testme_" + str(frame_number) + ".jpg"
        cv2.imshow(event, frame)
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(30)

        # closing all open windows
        # cv2.destroyAllWindows()
    del images, decoded_boxes, confidences, ego_preds, confidence, imgs_paths
    torch.cuda.empty_cache()


del net

print("EXITING...peacefully")

torch.cuda.empty_cache()
