import numpy as np
import json
import os
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
import slowfast.datasets.utils as utils

from slowfast.models.build import MODEL_REGISTRY
from slowfast.datasets.extractor import Extractor

logger = logging.get_logger(__name__)


@torch.no_grad()
def run(loader, model, cfg):
    model.eval()
    num_frames = cfg.DATA.NUM_FRAMES
    step_frames = int(num_frames / 2)
    fout = open(cfg.TEST.OUTPUT_FEATURE_FILE, "w")
    batch_size = cfg.TEST.BATCH_SIZE
    for v_ind, (frames, labels, video_idx, meta) in enumerate(loader):
        # Transfer the data to the current GPU device.
        if v_ind % 10 == 0:
            print("process video index:", v_ind, "total:", len(loader))
        length = frames.shape[1]
        features = []
        classifiers = []
        batch = []
        for k in range(0, length, step_frames):
            start = k
            end = min(k + num_frames, length)
            if len(batch) == batch_size or ((end - start < num_frames) and len(batch) > 0):
                # forward
                # batch_size x 3 num_slow_frames x 224 x 224
                b0 = torch.as_tensor(np.stack([b[0] for b in batch])).contiguous()
                # batch_size x 3 num_fast_frames x 224 x 224
                b1 = torch.as_tensor(np.stack([b[1] for b in batch])).contiguous()
                if torch.cuda.is_available():
                    b0 = b0.cuda(non_blocking=True)
                    b1 = b1.cuda(non_blocking=True)
                batch = [b0, b1]
                feat, cls = model(batch)
                features.append(feat.detach().cpu())
                classifiers.append(cls.detach().cpu())
                batch = []
            if end - start < num_frames:
                break
            inputs = frames[:, start:end]
            inputs = utils.pack_pathway_output(cfg, inputs)
            batch.append(inputs)

        # length of features: ceil((length - num_frames + 1)/step_frames)
        features = torch.cat(features, dim=0).numpy()
        classifiers = torch.cat(classifiers, dim=0).numpy()
        feat_name = os.path.join(cfg.OUTPUT_DIR, meta["video_name"] + ".feat.npy")
        np.save(feat_name, features)
        cls_name = os.path.join(cfg.OUTPUT_DIR, meta["video_name"] + ".cls.npy")
        np.save(cls_name, classifiers)

        meta["feature_shape"] = features.shape
        meta["cls_shape"] = classifiers.shape
        meta["feature_frame"] = (len(features)-1) * step_frames + num_frames
        meta["video_feature"] = feat_name
        meta["video_classifier"] = cls_name
        meta["step_frames"] = step_frames
        json_str = json.dumps(meta)
        fout.write(json_str + "\n")
        fout.flush()
    fout.close()


def extract(cfg):
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Extract with config:")
    logger.info(cfg)

    # initialize model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

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

    run(dataloader, model, cfg)
