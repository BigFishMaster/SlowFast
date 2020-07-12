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
def run(cfg, model, video_data, num_frames, step_frames, fout):
    frames, labels, video_idx, meta = video_data
    length = (frames.shape[1] // cfg.SLOWFAST.ALPHA) * cfg.SLOWFAST.ALPHA
    features = []
    for k in range(0, length, step_frames):
        start = k
        end = min(k + num_frames, length)
        inputs = frames[:, start:end]
        slow, fast = utils.pack_pathway_output(cfg, inputs)
        slow = slow.unsqueeze(0).contiguous()
        fast = fast.unsqueeze(0).contiguous()
        if torch.cuda.is_available():
            slow = slow.cuda(non_blocking=True)
            fast = fast.cuda(non_blocking=True)
        feat = model([slow, fast], ftype="video")
        features.append(feat.detach().cpu())

    features = torch.cat(features, dim=0).numpy()
    feat_name = os.path.join(cfg.OUTPUT_DIR, meta["video_name"] + ".feat.npy")
    np.save(feat_name, features)

    meta["feature_shape"] = features.shape
    meta["feature_frame"] = length
    meta["video_feature"] = feat_name
    meta["step_frames"] = step_frames
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
    step_frames = num_frames
    fout = open(cfg.TEST.OUTPUT_FEATURE_FILE, "w")
    start_time = time.time()
    for i in range(num_video):
        video_data = result_queue.get()
        run(cfg, model, video_data, num_frames, step_frames, fout)
        period = time.time() - start_time
        logger.info("video index: %d, period: %.2f sec, speed: %.2f sec/video."
                    %(i, period, period/(i+1)))
    fout.close()
