#!/usr/bin/env python3
"""B-spline smoothing and constant-arclength resampling for planned paths.

The planner emits six future waypoints spaced by time (0.5 s apart), not by
distance, and every point carries the model's own lateral noise. Feeding those
raw points to the MPC makes the vehicle weave. This module fits one smooth
parametric curve through them and re-samples it at a constant spacing, so the
controller always sees the same look-ahead geometry regardless of speed.

Kept free of rospy so it can be exercised offline.
"""

import numpy as np
from scipy.interpolate import splev, splprep


# Consecutive points closer than this are the same point as far as the spline
# parameterisation is concerned. At standstill the whole trajectory collapses
# into the origin and splprep raises on the zero-length knot span.
_DUPLICATE_EPS_M = 1e-3


class TrajectorySmoothingError(RuntimeError):
    """The trajectory is too degenerate to smooth; the caller should fall back."""


def _drop_duplicates(points_xy):
    """Return points with near-coincident consecutive entries removed.

    The first point is always kept, so an anchored start survives.
    """
    keep = [0]
    for index in range(1, len(points_xy)):
        if np.hypot(*(points_xy[index] - points_xy[keep[-1]])) >= _DUPLICATE_EPS_M:
            keep.append(index)
    return np.asarray(keep, dtype=int)


def _chord_length_parameter(points_xy):
    """Normalised cumulative chord length, the usual spline parameterisation."""
    steps = np.hypot(*np.diff(points_xy, axis=0).T)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    return cumulative / cumulative[-1]


def _arclength(points_xy):
    steps = np.hypot(*np.diff(points_xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _tangent_yaw(points_xy):
    """Heading of each point from the forward difference; the last copies its predecessor."""
    deltas = np.diff(points_xy, axis=0)
    yaw = np.arctan2(deltas[:, 1], deltas[:, 0])
    return np.concatenate([yaw, yaw[-1:]])


def smooth_and_resample(points_xy, point_times=None, spacing_m=0.7, sigma_m=0.2,
                        min_points=7, dense_n=200):
    """Smooth a planned path and re-sample it every ``spacing_m`` metres.

    ``points_xy`` is the full path including its start point, (M, 2). The
    coordinate convention is irrelevant here as long as it is metric and
    consistent — the planner passes (x_left, y_front).

    ``point_times`` are the seconds-from-now of each input point, used to give
    the re-sampled points a matching timestamp. Defaults to a unit ramp.

    Returns ``(points, times)`` where ``points`` is (N, 3) of (x, y, yaw) with
    yaw taken from the curve tangent, and ``times`` is (N,) seconds. The first
    point is exactly (0, 0): the smoothed curve is rigidly translated so the
    path always starts at the vehicle's current position.

    Raises TrajectorySmoothingError when the input is too degenerate to fit.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise TrajectorySmoothingError(f"expected (M, 2) points, got {points_xy.shape}")
    if point_times is None:
        point_times = np.arange(len(points_xy), dtype=np.float64)
    point_times = np.asarray(point_times, dtype=np.float64)
    if len(point_times) != len(points_xy):
        raise TrajectorySmoothingError("point_times must match points_xy in length")
    if spacing_m <= 0.0:
        raise TrajectorySmoothingError(f"spacing_m must be positive, got {spacing_m}")

    keep = _drop_duplicates(points_xy)
    unique_xy = points_xy[keep]
    unique_times = point_times[keep]
    if len(unique_xy) < 2:
        # Standstill: every waypoint sits on the origin, so there is no
        # direction to extrapolate along either.
        raise TrajectorySmoothingError("trajectory has no measurable extent")

    # scipy's `s` is a sum of squared residuals; expressing it as m * sigma^2
    # turns the knob into "metres of RMS deviation I will tolerate".
    count = len(unique_xy)
    degree = min(3, count - 1)
    tck, _ = splprep(
        [unique_xy[:, 0], unique_xy[:, 1]],
        u=_chord_length_parameter(unique_xy),
        s=count * float(sigma_m) ** 2,
        k=degree,
    )
    dense_u = np.linspace(0.0, 1.0, dense_n)
    dense_x, dense_y = splev(dense_u, tck)
    dense_xy = np.column_stack([dense_x, dense_y])
    # Rigid translation only: anchoring the start cannot distort the fitted shape.
    dense_xy -= dense_xy[0]
    dense_s = _arclength(dense_xy)
    total_length = dense_s[-1]
    if total_length < _DUPLICATE_EPS_M:
        raise TrajectorySmoothingError("smoothed trajectory has no measurable extent")

    # Strictly uniform spacing: a short tail remainder is dropped rather than
    # appended as an odd-length final segment.
    targets = np.arange(0.0, total_length + 1e-9, spacing_m)
    sampled = np.column_stack([
        np.interp(targets, dense_s, dense_xy[:, 0]),
        np.interp(targets, dense_s, dense_xy[:, 1]),
    ])

    # Where each input point landed on the smoothed curve, in metres. Sampling
    # times through that mapping keeps a re-sampled point at the moment the
    # model predicted for that place, even though spacing is now uniform.
    input_s = np.interp(_chord_length_parameter(unique_xy), dense_u, dense_s)
    times = np.interp(targets, input_s, unique_times)

    # Seconds per metre over the whole prediction, for timing extrapolated points.
    duration = unique_times[-1] - unique_times[0]
    time_rate = duration / total_length if duration > 0.0 else 0.0
    end_direction = (dense_xy[-1] - dense_xy[-2]) / max(dense_s[-1] - dense_s[-2], 1e-12)

    sampled, times = _extend_to_min_points(
        sampled, times, spacing_m, max(min_points, 2), end_direction, time_rate)
    return np.column_stack([sampled, _tangent_yaw(sampled)]), times


def _extend_to_min_points(points_xy, times, spacing_m, min_points,
                          end_direction, time_rate):
    """Extrapolate straight along the final heading until min_points is reached.

    Downstream local_path.cpp stops the vehicle when fewer than four waypoints
    lie ahead of it, so a short prediction must not shrink the published path
    below that. ``end_direction`` covers the case where the path was too short
    to yield even two samples and so has no heading of its own.
    """
    missing = min_points - len(points_xy)
    if missing <= 0:
        return points_xy, times

    if len(points_xy) >= 2:
        direction = points_xy[-1] - points_xy[-2]
        direction = direction / np.hypot(*direction)
    else:
        direction = end_direction
    steps = np.arange(1, missing + 1, dtype=np.float64)
    extra_xy = points_xy[-1] + np.outer(steps * spacing_m, direction)
    extra_times = times[-1] + steps * spacing_m * time_rate
    return np.vstack([points_xy, extra_xy]), np.concatenate([times, extra_times])
