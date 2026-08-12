#!/usr/bin/env python3
"""
Build an mp4 video from a tracking_data session folder (frames + centroids.csv),
with each frame held on screen for its actual measured capture duration
(via ffmpeg's concat demuxer) rather than a fixed fps.

Example:
    conda activate turtle
    python3 generate_tracking_video.py \
        /Users/esolo/august_10th_aquarium_data/tracking_data/20260624_180754 \
        --source detection
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import tempfile

import cv2

FALLBACK_DURATION = 0.15  # seconds, used when a frame's timestamp can't be determined
MAX_DURATION = 2.0  # clamp so gaps around corrupt/missing rows don't freeze the video for ages


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def frame_number(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def load_centroids(csv_path):
    """Return (rows, {frame_num: row_dict}), skipping any truncated/corrupt trailing rows."""
    with open(csv_path, newline='') as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    by_frame = {}
    skipped = 0
    for row in raw_rows:
        try:
            fnum = int(row['frame'])
            float(row['timestamp'])
        except (TypeError, ValueError):
            skipped += 1
            continue
        rows.append(row)
        by_frame[fnum] = row
    if skipped:
        print(f"Warning: skipped {skipped} malformed row(s) in {csv_path}")
    return rows, by_frame


def draw_overlay(img, row, elapsed, show_centroid, show_confidence, show_timestamp,
                  marker_radius, font_scale):
    has_detection = bool(row) and row.get('tracking_status') == 'tracking' and row.get('cx')

    if show_centroid and has_detection:
        cx, cy = float(row['cx']), float(row['cy'])
        cv2.drawMarker(img, (int(cx), int(cy)), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=marker_radius * 2,
                        thickness=2)
        cv2.circle(img, (int(cx), int(cy)), marker_radius, (0, 255, 0), 2)

    lines = []
    if show_timestamp:
        lines.append(f"t = {elapsed:.1f}s")
    if show_confidence:
        if has_detection and row.get('confidence'):
            lines.append(f"conf = {float(row['confidence']):.2f}")
        else:
            lines.append("conf = --")

    y = 22
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += int(24 * font_scale / 0.6)

    return img


def write_concat_file(concat_path, entries):
    """entries: list of (annotated_frame_path, duration_seconds), in playback order."""
    with open(concat_path, 'w') as f:
        for path, dur in entries:
            f.write(f"file '{path}'\n")
            f.write(f"duration {dur:.6f}\n")
        # ffmpeg's concat demuxer ignores the final duration, so repeat the last frame once more.
        f.write(f"file '{entries[-1][0]}'\n")


def run_ffmpeg_concat(concat_path, output_path):
    if shutil.which('ffmpeg') is None:
        raise SystemExit("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry.")
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_path,
        '-fps_mode', 'vfr',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # libx264 requires even width/height
        '-pix_fmt', 'yuv420p',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', help='Session folder containing centroids.csv, detection/, left/, right/')
    parser.add_argument('--source', choices=['detection', 'left', 'right'], default='detection',
                         help='Which image folder to build the video from (default: detection)')
    parser.add_argument('--output', default=None, help='Output mp4 path (default: <data_dir>/<source>_tracking.mp4)')
    parser.add_argument('--no-centroid', dest='show_centroid', action='store_false',
                         help='Do not draw a centroid marker')
    parser.add_argument('--no-confidence', dest='show_confidence', action='store_false',
                         help='Do not overlay the confidence value')
    parser.add_argument('--no-timestamp', dest='show_timestamp', action='store_false',
                         help='Do not overlay the elapsed time since the start of the experiment')
    parser.add_argument('--marker-radius', type=int, default=10)
    parser.add_argument('--font-scale', type=float, default=0.6)
    parser.set_defaults(show_centroid=True, show_confidence=True, show_timestamp=True)
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, 'centroids.csv')
    source_dir = os.path.join(args.data_dir, args.source)
    output_path = args.output or os.path.join(args.data_dir, f'{args.source}_tracking.mp4')

    rows, by_frame = load_centroids(csv_path)
    first_ts = float(rows[0]['timestamp'])
    dts = [float(rows[i + 1]['timestamp']) - float(rows[i]['timestamp']) for i in range(len(rows) - 1)]
    avg_dt = (sum(dts) / len(dts)) if dts else FALLBACK_DURATION

    frame_paths = sorted(glob.glob(os.path.join(source_dir, 'frame*.jpg')), key=natural_sort_key)
    if not frame_paths:
        raise SystemExit(f"No frames found in {source_dir}")

    tmp_dir = tempfile.mkdtemp(prefix='tracking_video_')
    try:
        entries = []
        timestamps = []  # parallel list of each written frame's own timestamp (or None)
        skipped_images = 0

        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                skipped_images += 1
                continue

            fnum = frame_number(path)
            row = by_frame.get(fnum)
            ts = float(row['timestamp']) if row else None
            elapsed = (ts - first_ts) if ts is not None else (fnum - 1) * avg_dt

            img = draw_overlay(img, row, elapsed, args.show_centroid, args.show_confidence,
                                args.show_timestamp, args.marker_radius, args.font_scale)

            out_path = os.path.join(tmp_dir, f'frame_{len(entries):06d}.jpg')
            cv2.imwrite(out_path, img)
            entries.append(out_path)
            timestamps.append(ts)

        if not entries:
            raise SystemExit("No readable frames to write.")

        # Duration of each frame = time until the next frame's timestamp (fallback to avg_dt
        # across missing/corrupt gaps), clamped so a bad gap can't freeze the video for ages.
        durations = []
        for i in range(len(entries) - 1):
            if timestamps[i] is not None and timestamps[i + 1] is not None:
                dur = timestamps[i + 1] - timestamps[i]
            else:
                dur = avg_dt
            dur = max(1.0 / 60, min(dur, MAX_DURATION))
            durations.append(dur)
        durations.append(avg_dt)  # last frame: no "next" timestamp to diff against

        concat_path = os.path.join(tmp_dir, 'concat_list.txt')
        write_concat_file(concat_path, list(zip(entries, durations)))
        run_ffmpeg_concat(concat_path, output_path)

        if skipped_images:
            print(f"Warning: skipped {skipped_images} unreadable/corrupt image(s) in {source_dir}")
        total_duration = sum(durations)
        print(f"Wrote {len(entries)} frames -> {output_path}")
        print(f"variable-duration playback, total~{total_duration:.1f}s")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
