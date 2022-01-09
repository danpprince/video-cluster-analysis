import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.color import rgb2gray
from skimage.filters import sobel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


OUTPUT_DIR = Path("output")
NUM_CLUSTERS = 40


@dataclass
class SubFrame:
    frame_index: int
    timestamp: float
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    img: np.ndarray
    features: Optional[np.ndarray] = None
    standardized_features: Optional[np.ndarray] = None
    cluster_index: Optional[int] = None


def get_clusters_for_video(video: Path) -> Any:
    process_frame_rate = 40

    subframes = []
    with imageio.get_reader(video, format="FFMPEG") as reader:
        video_frame_rate = reader.get_meta_data()["fps"]
        total_frames = reader.count_frames()
        for index in tqdm(
            range(0, total_frames, process_frame_rate), desc="Extracting subframes"
        ):
            frame = reader.get_data(index)

            if index % process_frame_rate == 0:
                current_subframes = extract_subframes_from_frame(
                    frame, index, video_frame_rate
                )
                subframes.extend(current_subframes)

    clusters, standardized_features = cluster_subframes(subframes)
    visualize_clusters(clusters, subframes, standardized_features)


def extract_subframes_from_frame(
    frame: np.ndarray,
    frame_index: int,
    frame_rate: float,
    write_subframes: bool = False,
) -> List[SubFrame]:
    subframe_size_px = 80

    num_rows = frame.shape[0] // subframe_size_px
    num_cols = frame.shape[1] // subframe_size_px

    if frame.dtype != np.uint8:
        raise ValueError(f"Image frame is of unexpected type {frame.dtype}")

    subframes = []
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            timestamp = frame_index / frame_rate

            start_row = row_index * subframe_size_px
            end_row = (row_index + 1) * subframe_size_px
            start_col = col_index * subframe_size_px
            end_col = (col_index + 1) * subframe_size_px

            subframe_img = frame[
                start_row:end_row,
                start_col:end_col,
                :,
            ]

            sf = SubFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                start_row=start_row,
                end_row=end_row,
                start_col=start_col,
                end_col=end_col,
                img=subframe_img,
            )
            sf.features = extract_features_from_image(sf.img)
            subframes.append(sf)

    return subframes


def extract_features_from_image(img: np.ndarray) -> np.ndarray:
    subframe_features = np.concatenate(
        (extract_color_features(img), extract_edge_features(img))
    )
    return subframe_features


def extract_color_features(img: np.ndarray, bins_per_channel: int = 8) -> np.ndarray:
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


def extract_edge_features(img: np.ndarray, num_bins: int = 8) -> np.ndarray:
    hist_min_value = 0.0
    hist_max_value = 0.75
    bin_edges = np.linspace(hist_min_value, hist_max_value, num_bins + 1)
    num_pixels_in_img = np.prod(img.shape[:2])

    img_gray = rgb2gray(img)

    edge_magnitude = sobel(img_gray)
    max_magnitude = np.max(edge_magnitude)
    if max_magnitude > hist_max_value:
        print(f"Edge magnitude of {max_magnitude} found, greater than hist max")

    edge_hist, _ = np.histogram(edge_magnitude, bin_edges)

    # Normalize the feature histogram to [0.0, 1.0]
    edge_hist = edge_hist / num_pixels_in_img

    return edge_hist


def cluster_subframes(subframes: List[SubFrame]) -> Tuple[KMeans, np.ndarray]:
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)

    features = np.stack([sf.features for sf in subframes if sf.features is not None])

    standardized_features = scaler.fit_transform(features)
    kmeans.fit(standardized_features)

    for index, sf in enumerate(subframes):
        sf.standardized_features = standardized_features[index, :]
        sf.cluster_index = kmeans.labels_[index]

    return kmeans, standardized_features


def visualize_clusters(
    clusters: KMeans, subframes: List[SubFrame], features: np.ndarray
) -> None:

    now_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    now_timestamp_dir = OUTPUT_DIR / now_timestamp

    for cluster_index in range(clusters.n_clusters):
        plt.figure(figsize=(10, 8))
        cluster_dir = now_timestamp_dir / f"cluster-{cluster_index :03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        timestamps = [
            sf.timestamp for sf in subframes if sf.cluster_index == cluster_index
        ]
        plt.hist(timestamps, bins=20)

        ticks = np.arange(0, timestamps[-1], 30)
        tick_labels = [f"{t // 60 :0.0f}:{t % 60 :02.0f}" for t in ticks]
        plt.xticks(ticks, tick_labels)

        plt.xlabel("Timestamp (minute:second)")
        plt.ylabel("Number of subframes")

        plt.title(f"Time distribution of cluster {cluster_index}")
        plt.tight_layout()
        plt.savefig(cluster_dir / "time-distribution.png")
        plt.close()

    centroid_imgs = []
    for cluster_index in range(clusters.n_clusters):
        cluster_dir = now_timestamp_dir / f"cluster-{cluster_index :03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Find representative subframe
        feature_centroid = clusters.cluster_centers_[cluster_index]

        distances = np.mean((feature_centroid - features) ** 2, axis=1)
        centroid_subframe_index = np.argmin(distances)
        centroid_img = subframes[centroid_subframe_index].img
        centroid_imgs.append(centroid_img)

        centroid_subframe_path = cluster_dir / "centroid_subframe.png"
        imageio.imwrite(centroid_subframe_path, centroid_img)

    plt.figure(figsize=(10, 8))
    num_rows = 5
    num_cols = NUM_CLUSTERS // num_rows
    for cluster_index, centroid_img in enumerate(centroid_imgs):
        num_subframes_in_cluster = np.sum(clusters.labels_ == cluster_index)

        plt.subplot(num_rows, num_cols, cluster_index + 1)
        plt.imshow(centroid_img)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"Cluster {cluster_index}\n{num_subframes_in_cluster} subframes",
            fontsize="small",
        )
    plt.tight_layout()
    plt.savefig(now_timestamp_dir / "clusters.png")

    for subframe_index, cluster_index in enumerate(clusters.labels_):
        subframe_img = subframes[subframe_index].img

        cluster_dir = OUTPUT_DIR / now_timestamp / f"cluster-{cluster_index :03d}"
        imageio.imwrite(
            cluster_dir / f"subframe-{subframe_index :06d}.png", subframe_img
        )


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
