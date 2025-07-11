# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/
"""

import os
import ffmpeg
import numpy as np
import subprocess
import torch
import tqdm

import videoseal
from videoseal.models import Videoseal
from videoseal.evals.metrics import bit_accuracy


def embed_video_clip(
    model: Videoseal, clip: np.ndarray, msgs: torch.Tensor
) -> np.ndarray:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    outputs = model.embed(
        clip_tensor, msgs=msgs, is_video=True, lowres_attenuation=True
    )
    processed_clip = outputs["imgs_w"]
    processed_clip = (processed_clip * 255.0).byte().permute(0, 2, 3, 1).numpy()
    return processed_clip


def embed_video(
    model: Videoseal, input_path: str, output_path: str, chunk_size: int, crf: int = 23
) -> None:
    # Read video dimensions
    probe = ffmpeg.probe(input_path)
    video_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])
    fps = float(video_info["r_frame_rate"].split("/")[0]) / float(
        video_info["r_frame_rate"].split("/")[1]
    )
    codec = video_info["codec_name"]
    num_frames = int(probe["streams"][0]["nb_frames"])

    # Open the input video
    process1 = (
        ffmpeg.input(input_path)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=fps,
        )
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )
    # Open the output video with optimal thread usage.
    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=fps,
        )
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=subprocess.PIPE)
    )

    # Create a random message
    msgs = model.get_random_msg()
    with open(output_path.replace(".mp4", ".txt"), "w") as f:
        f.write("".join([str(msg.item()) for msg in msgs[0]]))

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frames_in_chunk = 0

    for in_bytes in tqdm.tqdm(
        iter(lambda: process1.stdout.read(frame_size), b""),
        total=num_frames,
        desc="Watermark embedding",
    ):
        # Convert bytes to frame and add to chunk
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frames_in_chunk] = frame
        frames_in_chunk += 1

        # Process chunk when full
        if frames_in_chunk == chunk_size:
            # print(f"embedding at frame: {frame_idx}")
            processed_frames = embed_video_clip(model, chunk, msgs)
            process2.stdin.write(processed_frames.tobytes())
            frames_in_chunk = 0

    # Process final partial chunk if any
    if frames_in_chunk > 0:
        processed_frames = embed_video_clip(model, chunk[:frames_in_chunk], msgs)
        process2.stdin.write(processed_frames.tobytes())

    process1.stdout.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()

    return msgs


def detect_video_clip(model: Videoseal, clip: np.ndarray) -> torch.Tensor:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    outputs = model.detect(clip_tensor, is_video=True)
    output_bits = outputs["preds"][
        :, 1:
    ]  # exclude the first which may be used for detection
    return output_bits


def detect_video(model: Videoseal, input_path: str, chunk_size: int) -> None:
    # Read video dimensions
    probe = ffmpeg.probe(input_path)
    video_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])
    num_frames = int(probe["streams"][0]["nb_frames"])

    # Open the input video
    process1 = (
        ffmpeg.input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frame_count = 0
    soft_msgs = []
    pbar = tqdm.tqdm(total=num_frames, desc="Watermark extraction")
    while True:
        in_bytes = process1.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frame_count % chunk_size] = frame
        frame_count += 1
        pbar.update(1)
        if frame_count % chunk_size == 0:
            soft_msgs.append(detect_video_clip(model, chunk))
    process1.stdout.close()
    process1.wait()

    soft_msgs = torch.cat(soft_msgs, dim=0)
    soft_msgs = soft_msgs.mean(dim=0)  # Average the predictions across all frames
    return soft_msgs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_model = videoseal.load("videoseal")
    video_model.eval()
    video_model.to(device)
    video_model.compile()

    # Create the output directory and path
    os.makedirs(args.output_dir, exist_ok=True)
    args.output = os.path.join(args.output_dir, os.path.basename(args.input))

    # Embed the video
    msgs_ori = embed_video(video_model, args.input, args.output, 16)
    print(f"Saved watermarked video to {args.output}")

    # Detect the watermark in the video
    soft_msgs = detect_video(video_model, args.output, 16)
    bit_acc = bit_accuracy(soft_msgs, msgs_ori).item() * 100
    print(f"Binary message extracted with {bit_acc:.1f}% bit accuracy")

    if args.do_audio:
        # Placeholder to do audio watermarking as well
        pass
    else:
        # Copy just the audio from the original video
        temp_output = args.output + ".tmp"
        os.rename(args.output, temp_output)

        audiostream = ffmpeg.input(args.input)
        videostream = ffmpeg.input(temp_output)
        process3 = (
            ffmpeg.output(
                videostream.video,
                audiostream.audio,
                args.output,
                vcodec="copy",
                acodec="copy",
            )
            .overwrite_output()
            .run_async(pipe_stderr=subprocess.PIPE)
        )
        process3.wait()
        os.remove(temp_output)
        print("Copied audio from the original video")


if __name__ == "__main__":

    import argparse
    import videoseal.utils as utils

    parser = argparse.ArgumentParser(description="Process a video with Video Seal")
    parser.add_argument("--input", type=str, help="Input video path")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="outputs"
    )
    parser.add_argument("--do_audio", type=utils.bool_inst, default=False)
    args = parser.parse_args()

    main(args)
