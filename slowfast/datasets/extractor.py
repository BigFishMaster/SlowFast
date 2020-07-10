import os
import numpy as np
import torch.utils.data
from fvcore.common.file_io import PathManager
import slowfast.utils.logging as logging

from . import utils as utils
from . import video_container as container
import torchvision.transforms.functional as F

logger = logging.get_logger(__name__)


class Extractor:
    def __init__(self, cfg):
        self.cfg = cfg
        path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "test.txt")
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        self._path_to_videos = []
        self._labels = []
        self._duration_secs = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                path, start, end, label = path_label.split()
                self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                self._labels.append(int(label))
                self._duration_secs.append((float(start), float(end)))
        assert len(self._path_to_videos) > 0, "video paths are ZERO."
        logger.info(
            "Constructing Extractor dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )
        self._path_to_videos = np.array(self._path_to_videos, dtype=np.string_)

    def __getitem__(self, index):
        min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        target_fps = self.cfg.DATA.TARGET_FPS
        # initialize video container.
        video_container = container.get_video_container(
            self._path_to_videos[index].decode(),
            self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
            self.cfg.DATA.DECODING_BACKEND,
        )
        # video info.
        fps = float(video_container.streams.video[0].average_rate)
        video_frames = video_container.streams.video[0].frames
        duration = video_container.streams.video[0].duration
        time_base = float(video_container.streams.video[0].time_base)
        video_seconds = duration * time_base
        target_sampling_rate = int(sampling_rate * fps / target_fps)
        # decode all the frames.
        start, end = self._duration_secs[index]
        pts_per_frame = int(duration / video_frames)
        start_pts = int(start/video_seconds*video_frames) * pts_per_frame if start > 0 else 0
        end_pts = int(end/video_seconds*video_frames) * pts_per_frame if end > 0 else duration

        margin = 1024
        seek_offset = max(start_pts - margin, 0)
        # seek to nearest key frame, and decode from it to get subsequent frames in [start_pts, end_pts].
        video_container.seek(seek_offset, any_frame=False, backward=True, stream=video_container.streams.video[0])
        frames = []
        for i, frame in enumerate(video_container.decode(video=0)):
            if frame.pts < start_pts:
                continue
            if frame.pts > end_pts:
                break
            image = frame.to_image()
            scaled_image = F.resize(image, size=[min_scale, max_scale])
            frames.append(scaled_image)
        video_container.close()
        # sampling
        frames = frames[0:len(frames):target_sampling_rate]
        frames = torch.as_tensor(np.stack(frames))
        # make the min length of frames be NUM_FRAMES.
        if len(frames) < self.cfg.DATA.NUM_FRAMES:
            indices = torch.linspace(0, len(frames), self.cfg.DATA.NUM_FRAMES)
            indices = torch.clamp(indices, 0, frames.shape[0] - 1).long()
            frames = torch.index_select(frames, 0, indices)

        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W
        frames = frames.permute(3, 0, 1, 2)

        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=1,  # 0: left crop, 1: center crop, 2: right crop
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=False,
            inverse_uniform_sampling=False
        )

        label = self._labels[index]
        meta = {
            "video_frames": video_frames,
            "video_fps": fps,
            "target_sampling_rate": target_sampling_rate,
            "sampled_frames": frames.shape[1],
            "video_seconds": video_seconds,
            "video_name": self._path_to_videos[index].decode().split("/")[-1],
            "start_seconds": start,
            "end_seconds": end,
            "video_label": label,
        }
        return frames, label, index, meta

    def __len__(self):
        return len(self._path_to_videos)
