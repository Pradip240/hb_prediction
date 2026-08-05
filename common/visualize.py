"""
Utility functions for visualizing rPPG signals and facial regions.

Provides helpers for plotting extracted signals and generating overlay videos.
"""

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import config


def plot_signals(signals: np.ndarray, timestamps: np.ndarray, clip_name: str, output_path: str) -> None:
    """
    Plot the extracted per-region RGB signals.

    Args:
        signals: Per-region RGB signals with shape (T, R, 3).
        timestamps: Per-frame timestamps in seconds.
        clip_name: Name of the processed clip.
        output_path: Output PNG file.
    """
    # Extract the number of frames and RGB channels.
    n_frames, _regions, n_channels = signals.shape

    # Create the x-axis (time) relative to the first frame and compute clip duration.
    time_axis = timestamps - timestamps[0]
    duration = time_axis[-1] if len(time_axis) else 0.0

    # Create one subplot for each RGB channel.
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 9), sharex=True) # type: ignore
    channels = ["Red", "Green", "Blue"]
    for channel in range(len(channels)):
        ax = axes[channel] # type: ignore

        # Plot the mean RGB signal for every facial region.
        for region, region_name in enumerate(config.REGION_ORDER):
            bgr_color = np.array(config.REGION_COLORS[region_name], dtype=np.float32)
            color = (bgr_color[::-1] / 255).tolist()
            ax.plot( # type: ignore
                time_axis,
                signals[:, region, channel],
                color=color,
                linewidth=1.0,
                label=region_name
            )

        # Label the current channel with legend and show a light grid.
        ax.set_ylabel(f"Mean {channels[channel]}") # type: ignore
        ax.grid(alpha=0.3) # type: ignore
        if channel == 0:
            ax.legend(loc="upper right", fontsize=8) # type: ignore

    axes[-1].set_xlabel("Time (s)") # type: ignore
    # Add an overall title with clip metadata.
    fig.suptitle( # type: ignore
        f"Per-region skin RGB — {clip_name} ({duration:.1f}s, {n_frames} frames)", fontweight="bold"
    )
    # Adjust layout, save the figure, and release resources.
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # type: ignore
    plt.savefig(output_path, dpi=130) # type: ignore
    plt.close(fig)


def make_overlay_frame(
    frame: MatLike,
    skin_mask: NDArray[np.uint8],
    region_masks: NDArray[np.uint8],
    landmarks_xy: NDArray[np.float64],
) -> MatLike:
    """
    Create an overlay visualization for a single video frame.

    The output shows the skin mask together with the facial regions used
    during signal extraction.

    Args:
        frame: Video frame in BGR order with shape (H, W, 3).
        skin_mask: Smoothed binary skin mask with shape (H, W).
        region_masks: Binary region masks with shape (R, H, W).
        landmarks_xy: Smoothed landmark coordinates with shape (478, 2).

    Returns:
        Overlay frame in BGR format.
    """
    vis = frame.copy()

    # Tint the detected skin region.
    tint = np.zeros_like(vis)
    tint[skin_mask > 0] = (0, 180, 0)

    vis = cv2.addWeighted(vis, 1.0, tint,config.OVERLAY_SKIN_TINT_ALPHA, 0)

    # Scale factor for sub-pixel polygon rendering.
    factor = 1 << config.SUBPIX_SHIFT

    # Draw every facial region.
    for region_idx, region_name in enumerate(config.REGION_ORDER):
        color = config.REGION_COLORS[region_name]

        # Fill the extracted region.
        fill = np.zeros_like(vis)
        fill[region_masks[region_idx] > 0] = color
        vis = cv2.addWeighted(vis, 1.0, fill, config.OVERLAY_REGION_FILL_ALPHA, 0)

        # Draw the landmark polygon boundary.
        polygon = landmarks_xy[config.REGIONS[region_name]]
        hull = cv2.convexHull(polygon.astype(np.float32))
        hull = np.round(hull * factor).astype(np.int32)

        cv2.polylines(vis, [hull], True, color, 2, lineType=cv2.LINE_AA, shift=config.SUBPIX_SHIFT)
    return vis