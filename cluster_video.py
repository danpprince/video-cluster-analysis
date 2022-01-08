import argparse
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import tqdm


def get_clusters_for_video(video: Path) -> Any:
    # TODO: Allow skipping frames

    process_frame_rate = 300

    features = []
    subframes = []
    with imageio.get_reader(video, format="FFMPEG") as reader:
        for index, frame in tqdm(enumerate(reader), total=reader.count_frames()):

            if index % process_frame_rate == 0:
                current_subframes = extract_subframes_from_frame(frame, index)
                subframes.extend(current_subframes)
                current_features = extract_features_from_subframes(current_subframes)
                features.append(current_features)

    pass
    # TODO:
    # clusters = cluster_features(features)

    # analyze_clusters(clusters)


def extract_subframes_from_frame(
    frame: np.ndarray, frame_index: int, write_subframes: bool = True
) -> List[np.ndarray]:
    subframe_size_px = 80

    num_rows = frame.shape[0] // subframe_size_px
    num_cols = frame.shape[1] // subframe_size_px

    if frame.dtype != np.uint8:
        raise ValueError(f"Image frame is of unexpected type {frame.dtype}")

    subframes = []
    for row_index in range(num_rows):
        for col_index in range(num_cols):

            start_row = row_index * subframe_size_px
            end_row = (row_index + 1) * subframe_size_px
            start_col = col_index * subframe_size_px
            end_col = (col_index + 1) * subframe_size_px

            subframe_img = frame[
                start_row:end_row,
                start_col:end_col,
                :,
            ]
            subframes.append(subframe_img)

            # SubFrame(
            #     frame_index=frame_index,
            #     start_row=start_row,
            #     end_row=end_row,
            #     start_col=start_col,
            #     end_col=end_col,
            # )

            if write_subframes:
                output_path = Path("output") / (
                    f"frame{frame_index :06d}"
                    f"_row{row_index :02d}_col{col_index :02d}.png"
                )
                imageio.imwrite(output_path, subframe_img)

    return subframes



def extract_features_from_subframes(
    subframes: List[np.ndarray]
) -> List[np.ndarray]:
    subframe_features = [extract_color_features(f) for f in subframes]
    return subframe_features


def extract_color_features(img: np.ndarray, bins_per_channel: int = 4) -> np.ndarray:
    hist_min_value = np.iinfo(img.dtype).min
    hist_max_value = np.iinfo(img.dtype).max
    bin_edges = np.linspace(hist_min_value, hist_max_value, bins_per_channel + 1)

    num_pixels_in_img = np.prod(img.shape[:2])

    features = []
    for channel_index in range(img.shape[2]):
        img_channel = img[:, :, channel_index]
        channel_hist, _ = np.histogram(img_channel, bin_edges)

        # Normalize the feature histogram to [0.0, 1.0]
        channel_hist = channel_hist / num_pixels_in_img
        features.append(channel_hist)

    return np.concatenate(features)


def extract_edge_features():
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=Path,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    clusters = get_clusters_for_video(args.video)
    pass
