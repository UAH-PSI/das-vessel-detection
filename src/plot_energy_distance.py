#!/usr/bin/env python3
import argparse
import h5py
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import subprocess
from tqdm import tqdm

# Make sure matplotlib does not try to open any window
import matplotlib
matplotlib.use("Agg")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Import the same classes used in draw_dt_diagrams.py
from data_splitter import TripletRegressionReducer, DataRegressionSplitter

# --- VideoGenerator is unchanged from draw_dt_diagrams.py ---
class VideoGenerator:
    def __init__(self, output_path, fps=1):
        """
        Initialize the video generator.
        
        Parameters:
            output_path (str): Path where the video will be saved.
            fps (int): Frames per second (default is 1).
        """
        self.output_path = output_path
        self.fps = fps

    def generate_video(self, static_image, duration, inner_bbox, bar_width=2, bar_color=(0, 0, 255)):
        """
        Generates a temporary video with a scrolling vertical bar overlaying a subregion of the static_image,
        then re-encodes it to H264 using ffmpeg.
        
        Parameters:
            static_image (np.ndarray): The base image (RGB) used as background.
            duration (int): Duration of the video in seconds (i.e. number of frames if fps=1).
            inner_bbox (tuple): (x, y, width, height) in pixel coordinates corresponding to the inner axes.
            bar_width (int): Width of the scrolling vertical bar in pixels.
            bar_color (tuple): Color of the bar in BGR format (default is red).
        """
        
        height, width, _ = static_image.shape
        # Convert static image from RGB (Matplotlib) to BGR (OpenCV)
        static_bgr = cv2.cvtColor(static_image, cv2.COLOR_RGB2BGR)

        # Write a temporary video file using a codec that is supported (e.g., mp4v)
        temp_video = self.output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (width, height))
        total_frames = duration * self.fps  # e.g. if fps=10, duration=5s → 50 frames
        
        inner_x, inner_y, inner_width, inner_height = inner_bbox

        for frame_idx in range(total_frames):
            frame = static_bgr.copy()
            # Compute relative x; ensure if only one frame, rel_x=0
            rel_x = int((frame_idx / (total_frames - 1)) * inner_width) if total_frames > 1 else 0
            abs_x = inner_x + rel_x
            cv2.rectangle(
                frame,
                (abs_x, inner_y),
                (min(abs_x + bar_width, width), inner_y + inner_height),
                bar_color,
                -1
            )
            video_writer.write(frame)
        video_writer.release()

        # Re-encode via ffmpeg to H264 for broad compatibility
        cmd = [
            'ffmpeg', '-y', '-i', temp_video,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            self.output_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_video)


def load_hdf5_data(h5_path):
    """
    Load datasets X, y, datetimes from the HDF5 file.

    Returns:
        X: np.ndarray with shape (N, num_sensors, feature_dim)
        y: np.ndarray with shape (N,)
        dt: list of Python datetime objects (length N)
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found at: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # Required datasets
        if "X" not in f or "y" not in f or "datetimes" not in f:
            raise KeyError("HDF5 file must contain 'X', 'y', and 'datetimes' datasets.")

        X = f["X"][()]  # e.g. shape: (N, num_sensors, feature_dim)
        y = f["y"][()]  # shape: (N,)
        raw_datetimes = f["datetimes"][()]  # array of bytes or unicode

        # Convert bytes to unicode if necessary
        if raw_datetimes.dtype.kind == "S":
            dt_strings = raw_datetimes.astype("U")
        else:
            dt_strings = raw_datetimes.astype(str)

    # Parse each ISO string into a datetime object (with timezone if present)
    dt_list = []
    for s in dt_strings:
        try:
            # datetime.fromisoformat handles offsets like "+00:00"
            dt_list.append(datetime.fromisoformat(s))
        except Exception:
            # fallback to pandas if isoformat is slightly different
            dt_list.append(pd.to_datetime(s, utc=True).to_pydatetime())

    return X, y, dt_list


def main():
    parser = argparse.ArgumentParser(
        description="Draw distance‐time diagrams from an HDF5 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--h5_path", type=str, required=True,
        help="Path to the HDF5 file (e.g., dataset_sensor_range_1440_1690.h5)."
    )
    parser.add_argument(
        "--target_method", type=str, choices=["minimum", "average", "median"],
        default="minimum",
        help="(Ignored when HDF5 already contains y, but required for TripletRegressionReducer signature.)"
    )
    parser.add_argument(
        "--saturation_threshold", type=float, default=None,
        help="Threshold for saturation values (passed to the reducer)."
    )
    parser.add_argument(
        "--regression_threshold", type=float, default=None,
        help="Threshold for regression values (passed to the reducer)."
    )
    parser.add_argument(
        "--n_seconds", type=int, default=10,
        help="Length of the time window in seconds (default: 10)."
    )
    parser.add_argument(
        "--average_signals", action="store_true",
        help="Average signals within the time window (default: False)."
    )
    parser.add_argument(
        "--apply_log", action="store_true",
        help="Apply log transformation to features (default: False)."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-23,
        help="Epsilon value for log transformation (default: 1e-23)."
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random state for data splitting (default: 42)."
    )
    parser.add_argument(
        "--time_offset_seconds", type=int, default=0,
        help="Time offset in seconds (passed to the reducer)."
    )
    parser.add_argument(
        "--eliminate_within_range", type=int, default=None,
        help="Eliminate data within a specific range (passed to the reducer)."
    )
    parser.add_argument(
        "--remove_channels", type=int, nargs="+", default=[],
        help="List of channels to remove from X before plotting (default: none)."
    )
    parser.add_argument(
        "--first_day", type=int, default=16,
        help="First day of June 2023 to process (e.g., 16 → 2023‐06‐16)."
    )
    parser.add_argument(
        "--last_day", type=int, default=26,
        help="Last day of June 2023 to process (exclusive)."
    )
    parser.add_argument(
        "--num_hours_list", type=int, nargs="+", default=[2],
        help="List of window sizes (in hours) to slice each day (default: [2])."
    )
    parser.add_argument(
        "--max_dist", type=int, default=2000,
        help="Maximum distance to visualize in distance plot (default: 2000 m)."
    )
    parser.add_argument(
        "--save_dir", type=str, default="DT",
        help="Directory to save output images/videos (default: DT)."
    )
    parser.add_argument(
        "--hour_range", type=int, nargs=2, default=None,
        help="Two integer hours [start end] to slice each test‐day (e.g. 9 17)."
    )
    parser.add_argument(
        "--generate_video", action="store_true",
        help="Generate video output instead of static images."
    )
    parser.add_argument(
        "--time_interval", type=str, nargs=2, default=None,
        help="Two ISO timestamps (e.g. '2023-06-24T01:00:00+00:00' '2023-06-24T01:30:00+00:00') to select a specific interval."
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for video generation (default: 10)."
    )

    args = parser.parse_args()

    # Print out arguments for logging/debugging
    print("Command‐line arguments:")
    for arg, val in vars(args).items():
        print(f" - {arg}: {val}")
    print()

    # Load X, y, dt from HDF5
    try:
        X, y, dt = load_hdf5_data(args.h5_path)
    except Exception as e:
        print(f"Error loading HDF5 data: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert X, y, dt into numpy and Python types
    X = np.array(X)  # shape: (N, num_sensors, feature_dim)
    y = np.array(y)  # shape: (N,)
    # dt is already a Python list of datetime objects

    print(f"Loaded data from HDF5: X.shape = {X.shape}, y.shape = {y.shape}, len(datetimes) = {len(dt)}")
    num_timestamps, num_sensors, feat_dim = X.shape
    print(f" → Number of timestamps: {num_timestamps}")
    print(f" → Number of sensors:    {num_sensors}")
    print(f" → Feature dimension:    {feat_dim}")
    print()

    # Feed into the reducer (y is already present)
    reducer = TripletRegressionReducer(
        X, y, dt,
        n_seconds=args.n_seconds,
        time_offset_seconds=args.time_offset_seconds,
        apply_log=args.apply_log,
        average_signals=args.average_signals,
        threshold=args.regression_threshold,
        epsilon=args.epsilon,
        target_method=args.target_method,              # kept for signature; not used to recompute y
        eliminate_within_range=args.eliminate_within_range
    )
    X_reduced, y_reduced, dt_reduced = reducer.reduce_triplets()
    splitter = DataRegressionSplitter(X_reduced, y_reduced, dt_reduced)

    # Ensure save directory exists
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Handle the --time_interval branch first ----------
    if args.time_interval is not None:
        try:
            start_time = datetime.fromisoformat(args.time_interval[0])
            end_time = datetime.fromisoformat(args.time_interval[1])
        except Exception as e:
            print("Error parsing --time_interval:", e, file=sys.stderr)
            sys.exit(1)

        # Find indices in dt_reduced closest to those times
        start_idx = min(range(len(dt_reduced)), key=lambda i: abs((dt_reduced[i] - start_time).total_seconds()))
        end_idx = min(range(len(dt_reduced)), key=lambda i: abs((dt_reduced[i] - end_time).total_seconds()))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        indices = list(range(start_idx, end_idx + 1))
        if not indices:
            print("No data found for the given time_interval", file=sys.stderr)
            sys.exit(1)

        X_interval = X_reduced[indices]
        y_interval = np.array(y_reduced)[indices]
        dt_interval = [dt_reduced[i] for i in indices]

        # Remove any unwanted channels
        if args.remove_channels:
            X_interval = np.delete(X_interval, args.remove_channels, axis=1)

        useful_len = len(X_interval)
        num_sensors_now = X_interval.shape[1]

        # Build M (feature heatmap) and M2 (distance heatmap)
        M = np.zeros((useful_len, num_sensors_now))
        M2 = np.zeros((useful_len, num_sensors_now))
        if args.apply_log:
            M[:, :] = np.mean(X_interval, axis=2)
        else:
            M = np.log10(np.sum(np.square(X_interval), axis=2))
        for i in range(num_sensors_now):
            M2[:, i] = y_interval

        img1 = M.T
        img2 = M2.T

        fig, axs = plt.subplots(2, 1, figsize=(10, 5),
                                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        axs[0].imshow(img1, cmap='viridis', aspect='auto')
        axs[0].xaxis.tick_top()
        axs[0].set_ylabel('avg log strain in FB')
        x_ticks = np.linspace(0, useful_len - 1, 6, dtype=int)
        x_labels = [dt_interval[idx].strftime('%H:%M:%S') for idx in x_ticks]
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_labels)

        im2 = axs[1].imshow(img2, cmap='viridis_r', aspect='auto', vmin=0, vmax=args.max_dist)
        axs[1].set_ylabel('dist_min')
        axs[1].set_xticks(x_ticks)
        axs[1].set_xticklabels(x_labels)
        axs[1].set_yticklabels([])
        axs[1].set_yticks([])

        cbar2 = fig.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.25)
        cbar2.set_label('Distance (m)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Render to a static image buffer
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_w, fig_h = fig.canvas.get_width_height()
        static_image = buf.reshape(fig_h, fig_w, 3)

        # Compute inner_bbox for the top subplot (for the scrolling bar)
        renderer = fig.canvas.get_renderer()
        bbox = axs[0].get_window_extent(renderer=renderer)
        inner_x = int(bbox.x0)
        inner_width = int(bbox.x1 - bbox.x0)
        inner_y = fig_h - int(bbox.y1)
        inner_height = int(bbox.y1 - bbox.y0)
        inner_bbox = (inner_x, inner_y, inner_width, inner_height)

        base_filename = f"{save_dir}/combined_plot_interval_" + \
                        f"{start_time.strftime('%Y%m%dT%H%M%S')}_" + \
                        f"{end_time.strftime('%Y%m%dT%H%M%S')}"

        if args.generate_video:
            video_filename = base_filename + f"_{args.max_dist}.mp4"
            video_gen = VideoGenerator(output_path=video_filename, fps=args.fps)
            video_gen.generate_video(static_image, duration=useful_len, inner_bbox=inner_bbox,
                                     bar_width=2, bar_color=(0, 0, 255))
            print(f"Video saved to {video_filename}")
        else:
            png_filename = base_filename + ".png"
            plt.savefig(png_filename)
            print(f"Image saved to {png_filename}")

        plt.close()
        return

    # ---------- End of --time_interval branch ----------

    # Now iterate over days from first_day to last_day (June 2023)
    for day in tqdm(range(args.first_day, args.last_day), desc="Days"):
        test_day = datetime(2023, 6, day).date()
        X_train, X_test, y_train, y_test, dt_train, dt_test = splitter.split_by_day(test_day)

        ys = np.array(y_test)
        Xs = X_test.copy()
        # Remove channels if requested
        if args.remove_channels:
            Xs = np.delete(Xs, args.remove_channels, axis=1)

        # If --hour_range is specified, slice that hour window
        if args.hour_range is not None:
            start_hour, end_hour = args.hour_range
            indices_in_range = [
                i for i, dt_ in enumerate(dt_test)
                if (start_hour <= dt_.hour < end_hour)
            ]
            Xs_filt = Xs[indices_in_range]
            ys_filt = ys[indices_in_range]
            dt_filt = [dt_test[i] for i in indices_in_range]

            if len(Xs_filt) == 0:
                print(f"No data found for {test_day} in hour range {args.hour_range}")
                continue

            useful_len = len(Xs_filt)
            num_sensors_now = Xs_filt.shape[1]

            M = np.zeros((useful_len, num_sensors_now))
            M2 = np.zeros((useful_len, num_sensors_now))
            if args.apply_log:
                M[:, :] = np.mean(Xs_filt, axis=2)
            else:
                M = np.log10(np.sum(np.square(Xs_filt), axis=2))
            for i in range(num_sensors_now):
                M2[:, i] = ys_filt

            img1 = M.T
            img2 = M2.T

            fig, axs = plt.subplots(2, 1, figsize=(10, 5),
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
            axs[0].imshow(img1, cmap='viridis', aspect='auto')
            axs[0].xaxis.tick_top()
            axs[0].set_ylabel('avg log strain in FB')
            x_ticks = np.linspace(0, useful_len - 1, 6, dtype=int)
            x_labels = [dt_filt[idx].strftime('%H:%M') for idx in x_ticks]
            axs[0].set_xticks(x_ticks)
            axs[0].set_xticklabels(x_labels)

            im2 = axs[1].imshow(img2, cmap='viridis', aspect='auto', vmin=0, vmax=args.max_dist)
            axs[1].set_ylabel('dist_min')
            axs[1].set_xticks(x_ticks)
            axs[1].set_xticklabels(x_labels)
            axs[1].set_yticklabels([])
            axs[1].set_yticks([])

            cbar2 = fig.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.25)
            cbar2.set_label('Distance (m)')
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            fig_w, fig_h = fig.canvas.get_width_height()
            static_image = buf.reshape(fig_h, fig_w, 3)

            renderer = fig.canvas.get_renderer()
            bbox = axs[0].get_window_extent(renderer=renderer)
            inner_x = int(bbox.x0)
            inner_width = int(bbox.x1 - bbox.x0)
            inner_y = fig_h - int(bbox.y1)
            inner_height = int(bbox.y1 - bbox.y0)
            inner_bbox = (inner_x, inner_y, inner_width, inner_height)

            base_filename = f"{save_dir}/combined_plot_day_{test_day}_hr_{start_hour}-{end_hour}"
            if args.generate_video:
                video_filename = base_filename + f"_{args.max_dist}.mp4"
                video_gen = VideoGenerator(output_path=video_filename, fps=1)
                video_gen.generate_video(static_image, duration=useful_len, inner_bbox=inner_bbox,
                                         bar_width=2, bar_color=(0, 0, 255))
                print(f"Video saved to {video_filename}")
            else:
                png_filename = base_filename + ".png"
                plt.savefig(png_filename)
                print(f"Image saved to {png_filename}")

            plt.close()

        else:
            # No hour_range: iterate over each window size in num_hours_list
            for num_hours in tqdm(args.num_hours_list, desc="Hours", leave=False):
                # LT = number of samples per window = num_hours * samples_per_hour
                # In original: LT = num_hours * 360 (since 360 samples/hour at 10s interval)
                LT = num_hours * 360  
                nb = Xs.shape[0]
                num_sensors_now = Xs.shape[1]
                k = 0
                while k < nb:
                    useful_len = min(LT, len(Xs[k:]))
                    dt_slice = dt_test[k : k + useful_len]

                    M = np.zeros((useful_len, num_sensors_now))
                    M2 = np.zeros((useful_len, num_sensors_now))

                    if args.apply_log:
                        M[:, :] = np.mean(Xs[k : k + useful_len], axis=2)
                    else:
                        # Occasionally guard against extremely small sums
                        for t in range(k, k + useful_len):
                            for s in range(num_sensors_now):
                                if np.sum(np.square(Xs[t, s])) < args.epsilon / 1e6:
                                    Xs[t, s] = Xs[t, s]
                        M = np.log10(np.sum(np.square(Xs[k : k + useful_len]), axis=2))

                    for i in range(num_sensors_now):
                        M2[:, i] = ys[k : k + useful_len]

                    img1 = M.T
                    img2 = M2.T

                    fig, axs = plt.subplots(2, 1, figsize=(10, 5),
                                            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
                    axs[0].imshow(img1, cmap='viridis', aspect='auto')
                    axs[0].xaxis.tick_top()
                    axs[0].set_ylabel('avg log strain in FB')
                    x_ticks = np.linspace(0, useful_len - 1, 6, dtype=int)
                    x_labels = [dt_slice[idx].strftime('%H:%M') for idx in x_ticks]
                    axs[0].set_xticks(x_ticks)
                    axs[0].set_xticklabels(x_labels)

                    im2 = axs[1].imshow(img2, cmap='viridis', aspect='auto', vmin=0, vmax=args.max_dist)
                    axs[1].set_ylabel('dist_min')
                    axs[1].set_xticks(x_ticks)
                    axs[1].set_xticklabels(x_labels)
                    axs[1].set_yticklabels([])
                    axs[1].set_yticks([])

                    cbar2 = fig.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.25)
                    cbar2.set_label('Distance (m)')
                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    fig_w, fig_h = fig.canvas.get_width_height()
                    static_image = buf.reshape(fig_h, fig_w, 3)

                    renderer = fig.canvas.get_renderer()
                    bbox = axs[0].get_window_extent(renderer=renderer)
                    inner_x = int(bbox.x0)
                    inner_width = int(bbox.x1 - bbox.x0)
                    inner_y = fig_h - int(bbox.y1)
                    inner_height = int(bbox.y1 - bbox.y0)
                    inner_bbox = (inner_x, inner_y, inner_width, inner_height)

                    # Construct a timestamp string based on k
                    time_str = (datetime.min + timedelta(seconds=k * 10)).time().strftime('%H:%M:%S')
                    hours_label = f"{LT * 10 / 3600:02.0f}"  # e.g. "02" for 2 hours
                    base_filename = (
                        f"{save_dir}/combined_plot_d_{test_day.day}-"
                        f"{k:05d}-{time_str}-H{hours_label}"
                    )

                    if args.generate_video:
                        video_filename = base_filename + f"_{args.max_dist}.mp4"
                        video_gen = VideoGenerator(output_path=video_filename, fps=1)
                        video_gen.generate_video(static_image, duration=useful_len, inner_bbox=inner_bbox,
                                                 bar_width=2, bar_color=(0, 0, 255))
                        print(f"Video saved to {video_filename}")
                    else:
                        png_filename = base_filename + ".png"
                        plt.savefig(png_filename)
                        print(f"Image saved to {png_filename}")

                    plt.close()
                    k += LT


if __name__ == "__main__":
    main()
