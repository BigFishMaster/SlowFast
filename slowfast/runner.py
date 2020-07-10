from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from slowfast.tools.test_net import test
from slowfast.tools.train_net import train
from slowfast.tools.extract import extract


def main():
    args = parse_args()
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform feature extraction.
    if cfg.MODEL.EXTRACTOR:
        extract(cfg)


if __name__ == "__main__":
    main()
