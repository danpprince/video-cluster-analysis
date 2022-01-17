import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


OUTPUT_DIR = Path("output")
NUM_CLUSTERS = 400


@dataclass
class SubFrame:
    """Contains metadata, features, and pixels for a subsection of a video frame"""

    timestamp: float
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    img: np.ndarray
    features: Optional[np.ndarray] = None
    standardized_features: Optional[np.ndarray] = None
    cluster_index: Optional[int] = None


def analyze_clusters_for_video(video: Path, process_frame_rate: int = 30) -> None:
    """Clusters subframe features for a video file on disk

    Args:
        video: Path to a video file on disk.
        process_frame_rate: The number of frames to skip when extracting subframes.
            A lower number may give better results but take longer to compute.
    """

    subframes = []
    with imageio.get_reader(video, format="FFMPEG") as reader:
        video_frame_rate = reader.get_meta_data()["fps"]
        total_frames = reader.count_frames()
        for index in tqdm(
            range(0, total_frames, process_frame_rate), desc="Extracting subframes"
        ):
            frame = reader.get_data(index)

            if index % process_frame_rate == 0:
                timestamp = index / video_frame_rate
                current_subframes = extract_subframes_from_frame(frame, timestamp)
                subframes.extend(current_subframes)

    clusters, standardized_features = cluster_subframes(subframes)
    visualize_clusters(clusters, subframes, standardized_features)


def extract_subframes_from_frame(
    frame: np.ndarray,
    timestamp: float,
    subframe_size_px: int = 40,
) -> List[SubFrame]:
    """Identify subframes and extract features for one image frame of a video

    Args:
        frame: A color image with the first dimension being number of rows, the second
            dimension the number of columns, and the third dimension the number of
            color channels. A three-channel RGB image is typically expected.
        timestamp: The time in seconds when the frame occurs in the source video.
        subframe_size_px: The height and width of subframes in pixels.

    Returns:
        All SubFrame objects including their features, metadata, and images from the
        frame argument.

    Raises:
        ValueError: The frame is not of the expected unsigned 8-bit integer type.
    """

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

            sf = SubFrame(
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
    """Extract low-dimensional, visually representative features for an image

    Args:
        frame: A color image with the first dimension being number of rows, the second
            dimension the number of columns, and the third dimension the number of
            color channels. A three-channel RGB image is typically expected.

    Returns:
        Extracted color histogram features.
    """
    # TODO: Optionally extract edge or HOG features
    subframe_features = extract_color_features(img)
    return subframe_features


def extract_color_features(img: np.ndarray, bins_per_channel: int = 4) -> np.ndarray:
    """Extract color distribution features for an image

    Args:
        img: A color image with the first dimension being number of rows, the second
            dimension the number of columns, and the third dimension the number of
            color channels. A three-channel RGB image is typically expected.
        bins_per_channel: The number of histogram bins to assign to each image channel.

    Returns:
        Extracted color histogram features, with a number of dimensions equal to the
        number of channels in `img` times `bins_per_channel`. Note that each channel's
        bins are normalized based on the number of pixels in `img`.
    """
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


def extract_edge_features(
    img: np.ndarray, num_bins: int = 4, hist_min_value: float = 0.0, hist_max_value=0.75
) -> np.ndarray:
    """Extract edge distribution features for an image

    Args:
        img: A color image with the first dimension being number of rows, the second
            dimension the number of columns, and the third dimension the number of
            color channels. A three-channel RGB image is typically expected.
        num_bins: The number of histogram bins to assign to each image channel.
        hist_min_value: The minimum value to use in the edge distribution binning.
        hist_max_value: The maximum value to use in the edge distribution binning.

    Returns:
        Extracted edge distribution histogram features from a grayscale version of
        `img`, with a number of dimensions equal to `num_bins`. Note that the bins are
        normalized based on the number of pixels in `img`.
    """
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


def cluster_subframes(
    subframes: List[SubFrame], is_sweeping_num_clusters: bool = False
) -> Tuple[KMeans, np.ndarray]:
    """Standardize and cluster the features from given subframes

    Args:
        subframes: SubFrames to assign clusters to based on their extracted features
        is_sweeping_num_clusters: If True, this function will spend extra time to
            cluster `subframes` multiple times and save a plot to the output directory
            for use in evaluating several cluster numbers via the "elbow" method.

    Returns:
        The KMeans object containing the cluster label for each object in `subframes`,
        and the zero-mean and unit variance standardized version of their features.
    """
    scaler = StandardScaler()
    features = np.stack([sf.features for sf in subframes if sf.features is not None])
    standardized_features = scaler.fit_transform(features)

    if is_sweeping_num_clusters:
        num_cluster_trials = list(range(50, NUM_CLUSTERS, 50))
        inertias = []
        for num_clusters in tqdm(
            num_cluster_trials, desc="Performing clustering trials"
        ):
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            kmeans.fit(standardized_features)
            inertias.append(kmeans.inertia_)

        plt.figure()
        plt.plot(num_cluster_trials, inertias, "-o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.savefig(NOW_TIMESTAMP_DIR / "cluster-trials.png")
        plt.close()

    else:
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
        kmeans.fit(standardized_features)

    for index, sf in enumerate(subframes):
        sf.standardized_features = standardized_features[index, :]
        sf.cluster_index = kmeans.labels_[index]

    return kmeans, standardized_features


def visualize_clusters(
    clusters: KMeans, subframes: List[SubFrame], features: np.ndarray
) -> None:
    """Visualize the clustering results for given subframes and their features

    Args:
        clusters: An object containing the cluster labels assigned to each SubFrame.
        subframes: SubFrame Objects including their features, metadata, and images.
        features: Standardized (zero-mean and unit-variance) features for each object
            in `subframes`.
    """

    max_timestamp = subframes[-1].timestamp

    for cluster_index in range(clusters.n_clusters):
        plt.figure(figsize=(12, 8))
        cluster_dir = NOW_TIMESTAMP_DIR / f"cluster-{cluster_index :03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        timestamps = [
            sf.timestamp for sf in subframes if sf.cluster_index == cluster_index
        ]
        plt.hist(timestamps, bins=np.arange(0, max_timestamp, 30))

        ticks = np.arange(0, timestamps[-1], 30)
        tick_labels = [f"{t // 60 :0.0f}:{t % 60 :02.0f}" for t in ticks]
        plt.xticks(ticks, tick_labels, rotation=45)

        plt.xlabel("Timestamp (minute:second)")
        plt.ylabel("Number of subframes")

        plt.title(f"Time distribution of cluster {cluster_index}")
        plt.tight_layout()
        plt.savefig(cluster_dir / "time-distribution.png")
        plt.close()

    centroid_imgs = []
    for cluster_index in range(clusters.n_clusters):
        cluster_dir = NOW_TIMESTAMP_DIR / f"cluster-{cluster_index :03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Find representative subframe
        feature_centroid = clusters.cluster_centers_[cluster_index]
        distances = np.mean((feature_centroid - features) ** 2, axis=1)
        centroid_subframe_index = np.argmin(distances)
        centroid_img = subframes[centroid_subframe_index].img
        centroid_imgs.append(centroid_img)

        centroid_subframe_path = cluster_dir / "centroid_subframe.png"
        imageio.imwrite(centroid_subframe_path, centroid_img)

    clusters_per_figure = 100
    num_figures = int(np.ceil(NUM_CLUSTERS / clusters_per_figure))
    for figure_index in range(num_figures):
        plt.figure(figsize=(12, 8))
        num_rows = 6
        num_cols = int(np.ceil(clusters_per_figure / num_rows))
        start_cluster = figure_index * clusters_per_figure
        end_cluster = (figure_index + 1) * clusters_per_figure

        for cluster_index in range(start_cluster, end_cluster):
            centroid_img = centroid_imgs[cluster_index]
            num_subframes_in_cluster = np.sum(clusters.labels_ == cluster_index)

            plt.subplot(num_rows, num_cols, cluster_index + 1 - start_cluster)
            plt.imshow(centroid_img)
            plt.xticks([])
            plt.yticks([])
            plt.title(
                f"Clstr. {cluster_index}\n{num_subframes_in_cluster} sfs",
                fontsize="x-small",
            )
        plt.tight_layout()
        plt.savefig(NOW_TIMESTAMP_DIR / f"{figure_index :02d}-clusters.png")
        plt.close()

    for subframe_index, cluster_index in enumerate(
        tqdm(clusters.labels_, desc="Saving subframes")
    ):
        subframe_img = subframes[subframe_index].img

        cluster_dir = OUTPUT_DIR / now_timestamp / f"cluster-{cluster_index :03d}"
        imageio.imwrite(
            cluster_dir / f"subframe-{subframe_index :06d}.png", subframe_img
        )


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments

    Returns:
        An object containing the parsed command line argument for the path to a
        source video on disk.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=Path,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    now_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    NOW_TIMESTAMP_DIR = OUTPUT_DIR / now_timestamp
    NOW_TIMESTAMP_DIR.mkdir(parents=True, exist_ok=True)

    analyze_clusters_for_video(args.video)
