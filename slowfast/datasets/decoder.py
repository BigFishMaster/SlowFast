import math
import numpy as np
import random
import torch
import slowfast.utils.logging as logging
logger = logging.get_logger(__name__)


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def get_start_end_pts(video_size, clip_size, clip_idx, num_clips,
                      duration, time_base, start_sec, end_sec):
    all_sec = duration * time_base
    video_start_idx = int(start_sec / all_sec * video_size)
    video_end_idx = int(end_sec / all_sec * video_size)
    delta = max(video_end_idx - clip_size, 0)

    if clip_idx == -1:
        # Random temporal sampling.
        if video_start_idx < delta:
            start_idx = random.uniform(video_start_idx, delta)
        else:
            logger.warn("In train, video_start_idx {} should be smaller than delta {}.".format(
                video_start_idx, delta))
            # TODO: rethink this strategy?
            new_start_idx = max(delta-clip_size*0.1, 0)
            start_idx = random.uniform(new_start_idx, delta)
    else:
        # Uniformly sample the clip with the given index.
        if video_start_idx < delta:
            step = (delta - video_start_idx) / num_clips
            start_idx = video_start_idx + step * clip_idx
        else:
            logger.warn("In test, video_start_idx {} should be smaller than delta {}.".format(
                video_start_idx, delta))
            new_start_idx = max(delta-clip_size*0.1, 0)
            step = (delta - new_start_idx) / num_clips
            start_idx = video_start_idx + step * clip_idx

    end_idx = start_idx + clip_size - 1

    pts_per_frame = duration / video_size
    video_start_pts = int(start_idx * pts_per_frame)
    video_end_pts = int(end_idx * pts_per_frame)

    return video_start_idx, video_end_pts


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def pyav_decode(
    container, sampling_rate, num_frames, clip_idx, num_clips=10, target_fps=30,
    start_sec=None, end_sec=None
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
        start_sec (float): starting second of the video, with a specified label.
        end_sec (float): ending second of the video, with a specified label.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        if start_sec is not None and end_sec is not None:
            tb = float(container.streams.video[0].time_base)
            video_start_pts = int(start_sec/tb)
            video_end_pts = int(end_sec/tb)
        else:
            video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        if start_sec is not None and end_sec is not None:
            tb = float(container.streams.video[0].time_base)
            video_start_pts, video_end_pts = get_start_end_pts(frames_length,
                int(num_frames * sampling_rate * fps / target_fps),
                clip_idx, num_clips, duration, tb, start_sec, end_sec)
        else:
            start_idx, end_idx = get_start_end_idx(
                frames_length,
                int(num_frames * sampling_rate * fps / target_fps),
                clip_idx,
                num_clips,
            )
            # container.streams.video[0]: Fraction(1, 15360)
            # timebase: 1024
            # pts: in timebase units.
            timebase = duration / frames_length
            video_start_pts = int(start_idx * timebase)
            video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    target_fps=30,
    start_sec=None,
    end_sec=None,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
        start_sec (float): starting second of the video, with a specified label.
        end_sec (float): ending second of the video, with a specified label.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    assert clip_idx >= -1, "Not valid clip_idx {}".format(clip_idx)
    try:
        frames, fps, decode_all_video = pyav_decode(
            container,
            sampling_rate,
            num_frames,
            clip_idx,
            num_clips,
            target_fps,
            start_sec,
            end_sec,
        )
    except Exception as e:
        logger.exception("Failed to decode by pyav with exception: {}".format(e))
        return None

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return None

    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        num_frames * sampling_rate * fps / target_fps,
        clip_idx if decode_all_video else 0,
        num_clips if decode_all_video else 1,
    )
    # Perform temporal sampling from the decoded video.
    frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    return frames
