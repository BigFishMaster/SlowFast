import numpy as np
import time
import json
import os
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
import slowfast.datasets.utils as utils

from slowfast.models.build import MODEL_REGISTRY
from slowfast.datasets.extractor import Extractor
from torch import multiprocessing

logger = logging.get_logger(__name__)


def get_video(dataloader, index_queue, result_queue):
    while True:
        index = index_queue.get()
        video_data = dataloader[index]
        result_queue.put(video_data)


@torch.no_grad()
def run(cfg, model, video_data, num_frames, batch_size, fout):
    frames, labels, video_idx, meta = video_data
    slow, fast = frames
    C, T_show, H, W = slow.shape
    _, T_fast, _, _ = fast.shape

    num_frames_fast = num_frames
    num_batch = T_fast // (num_frames_fast * batch_size)
    batch_fast = num_frames_fast * batch_size * num_batch
    remain_fast = T_fast - batch_fast

    num_frames_slow = num_frames//cfg.SLOWFAST.ALPHA
    batch_slow = num_frames_slow * batch_size * num_batch
    remain_slow = T_show - batch_slow
    # slow: C x T x H x W
    # fast: C x aT x H x W
    features = []
    if batch_fast > 0:
        fast1 = fast[:, :batch_fast].reshape(C, num_batch, batch_size, num_frames_fast, H, W)
        fast1 = fast1.permute(1, 2, 0, 3, 4, 5)
        slow1 = slow[:, :batch_slow].reshape(C, num_batch, batch_size, num_frames_slow, H, W)
        slow1 = slow1.permute(1, 2, 0, 3, 4, 5)
        for k in range(num_batch):
            s = slow1[k].contiguous()
            f = fast1[k].contiguous()
            if torch.cuda.is_available():
                s = s.cuda(non_blocking=True)
                f = f.cuda(non_blocking=True)
            feat = model([s, f], ftype="video")
            features.append(feat.detach().cpu())

    if remain_fast > 0:
        slow2 = slow[:, -remain_slow:]
        fast2 = fast[:, -remain_fast:]
        for k in range(0, remain_slow, num_frames_slow):
            start_s = k
            end_s = min(start_s + num_frames_slow, remain_slow)
            start_f = start_s * cfg.SLOWFAST.ALPHA
            end_f = min(start_f + num_frames_fast, remain_fast)

            # 1 x C x T x H x W
            s = slow2[:, start_s:end_s].unsqueeze(0).contiguous()
            f = fast2[:, start_f:end_f].unsqueeze(0).contiguous()
            if torch.cuda.is_available():
                s = s.cuda(non_blocking=True)
                f = f.cuda(non_blocking=True)
            feat = model([s, f], ftype="video")
            features.append(feat.detach().cpu())

    features = torch.cat(features, dim=0).numpy()
    feat_name = os.path.join(cfg.OUTPUT_DIR, meta["video_name"] + ".feat.npy")
    np.save(feat_name, features)

    meta["feature_shape"] = features.shape
    meta["feature_frame"] = T_fast
    meta["video_feature"] = feat_name
    meta["step_frames"] = num_frames_fast
    json_str = json.dumps(meta)
    fout.write(json_str + "\n")
    fout.flush()


def video_extract(cfg):
    ctx = multiprocessing.get_context("spawn")
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Extract with config:")
    logger.info(cfg)

    # initialize model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # initialize data loader
    dataloader = Extractor(cfg)
    logger.info("Testing model for {} videos".format(len(dataloader)))

    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            path_to_checkpoint=cfg.TEST.CHECKPOINT_FILE_PATH,
            model=model,
            data_parallel=True,
            optimizer=None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        logger.info("Testing with random initialization. Only for debugging.")

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=get_video, args=(dataloader, index_queue, result_queue))
               for i in range(cfg.TEST.WORKERS)]

    for w in workers:
        w.daemon = True
        w.start()

    num_video = len(dataloader)

    for i in range(num_video):
        index_queue.put(i)

    # NUM_FRAMES must be divided by ALPHA.
    num_frames = cfg.DATA.NUM_FRAMES
    batch_size = cfg.TEST.BATCH_SIZE
    fout = open(cfg.TEST.OUTPUT_FEATURE_FILE, "w")
    start_time = time.time()
    for i in range(num_video):
        video_data = result_queue.get()
        run(cfg, model, video_data, num_frames, batch_size, fout)
        period = time.time() - start_time
        logger.info("video index: %d, period: %.2f sec, speed: %.2f sec/video."
                    %(i, period, period/(i+1)))
    fout.close()
