#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live visualisation for the realtime planner.

Opens a matplotlib window (needs a display / $DISPLAY) that shows, in the global
odom frame:
  * the robot's current position and heading           (from /odom)
  * the trajectory it has actually travelled            (accumulated /odom)
  * the ST-P3 predicted future trajectory               (/senpai/path)
  * the current command                                 (/senpai/command)
and prints the robot's x / y / yaw in a corner text box.

Run in its own terminal alongside the planner:
    source /opt/ros/noetic/setup.bash
    python3 realtime/visualize.py
"""

import math
import threading
from collections import deque

import numpy as np

import rospy
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def yaw_from_quaternion(z: float, w: float, x: float = 0.0, y: float = 0.0) -> float:
    """Yaw (rad) about +z from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def base_link_to_global(px, py, rx, ry, ryaw):
    """
    Transform base_link points (x forward, y left) into the global frame given
    the robot pose (rx, ry, ryaw). Accepts scalars or numpy arrays for px, py.
    """
    c, s = math.cos(ryaw), math.sin(ryaw)
    return rx + c * px - s * py, ry + s * px + c * py


class VisualizerNode:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.path_topic = rospy.get_param("~path_topic", "/senpai/path")
        self.command_topic = rospy.get_param("~command_topic", "/senpai/command")
        # Append a travelled-path point only after moving this far (m).
        self.min_step = float(rospy.get_param("~min_step", 0.05))
        self.history_len = int(rospy.get_param("~history_len", 4000))
        self.span = float(rospy.get_param("~view_span", 15.0))  # half-window (m)

        self.lock = threading.Lock()
        self.pose = None                       # (x, y, yaw)
        self.travelled = deque(maxlen=self.history_len)
        self.pred_base = None                  # (N,2) forward/left in base_link
        self.command = "—"

        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=5)
        rospy.Subscriber(self.path_topic, Path, self.cb_path, queue_size=1)
        rospy.Subscriber(self.command_topic, String, self.cb_command, queue_size=1)

        rospy.loginfo(f"[visualize] odom={self.odom_topic} path={self.path_topic} "
                      f"command={self.command_topic}")

    # ---------- callbacks (background threads) ----------

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(o.z, o.w, o.x, o.y)
        with self.lock:
            self.pose = (p.x, p.y, yaw)
            if not self.travelled or math.hypot(p.x - self.travelled[-1][0],
                                                p.y - self.travelled[-1][1]) >= self.min_step:
                self.travelled.append((p.x, p.y))

    def cb_path(self, msg: Path):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        with self.lock:
            self.pred_base = np.asarray(pts, dtype=float) if pts else None

    def cb_command(self, msg: String):
        with self.lock:
            self.command = msg.data

    # ---------- drawing (main thread) ----------

    def snapshot(self):
        with self.lock:
            pose = self.pose
            travelled = np.asarray(self.travelled, dtype=float) if self.travelled else None
            pred_base = None if self.pred_base is None else self.pred_base.copy()
            command = self.command
        return pose, travelled, pred_base, command

    def draw(self, ax):
        pose, travelled, pred_base, command = self.snapshot()
        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)

        if pose is None:
            ax.set_title("waiting for /odom …")
            return

        rx, ry, ryaw = pose

        if travelled is not None and len(travelled) >= 2:
            ax.plot(travelled[:, 0], travelled[:, 1], "-", color="#1f77b4",
                    lw=2, label="travelled")

        if pred_base is not None and len(pred_base) >= 1:
            gx, gy = base_link_to_global(pred_base[:, 0], pred_base[:, 1], rx, ry, ryaw)
            ax.plot(gx, gy, "-o", color="#ff7f0e", ms=3, lw=2, label="predicted")

        # robot position + heading arrow
        ax.plot(rx, ry, "o", color="#d62728", ms=9, label="robot")
        arrow = 0.15 * self.span
        ax.arrow(rx, ry, arrow * math.cos(ryaw), arrow * math.sin(ryaw),
                 head_width=0.08 * self.span, head_length=0.08 * self.span,
                 fc="#d62728", ec="#d62728", length_includes_head=True)

        ax.set_xlim(rx - self.span, rx + self.span)
        ax.set_ylim(ry - self.span, ry + self.span)
        ax.set_title("realtime planner — odom frame")
        ax.legend(loc="upper right", fontsize=8)
        ax.text(0.02, 0.98,
                f"x   = {rx:+.2f} m\ny   = {ry:+.2f} m\nyaw = {math.degrees(ryaw):+.1f}°\n"
                f"cmd = {command}",
                transform=ax.transAxes, va="top", ha="left", family="monospace",
                fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    def spin(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        timer = fig.canvas.new_timer(interval=100)  # ~10 Hz
        timer.add_callback(lambda: (self.draw(ax), fig.canvas.draw_idle()))
        timer.start()
        # Close the window or Ctrl-C to exit.
        fig.canvas.mpl_connect("close_event", lambda _evt: rospy.signal_shutdown("window closed"))
        plt.show()


def main():
    rospy.init_node("visualize", anonymous=False)
    VisualizerNode().spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
