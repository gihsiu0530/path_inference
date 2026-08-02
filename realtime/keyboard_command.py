#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keyboard command publisher for the realtime planner.

Reads the keyboard in raw (cbreak) mode and publishes LEFT / FORWARD / RIGHT to
/senpai/command (std_msgs/String, latched) so realtime_planner_node.py steers
live.

Keys (latching — a key stays in effect until you press another):
    Left  arrow / a   -> LEFT
    Up    arrow / w   -> FORWARD
    Right arrow / d   -> RIGHT
    space             -> FORWARD
    q / Ctrl-C        -> quit

Run in its own terminal (needs a real TTY):
    source /opt/ros/noetic/setup.bash
    python3 realtime/keyboard_command.py
"""

import os
import sys
import select
import termios
import tty

import rospy
from std_msgs.msg import String


KEY_TO_COMMAND = {
    "\x1b[D": "LEFT",     "a": "LEFT",
    "\x1b[A": "FORWARD",  "w": "FORWARD",  " ": "FORWARD",
    "\x1b[C": "RIGHT",    "d": "RIGHT",
}
QUIT_KEYS = {"q", "\x03"}  # q or Ctrl-C

HELP = (
    "\n[keyboard_command] steering the realtime planner\n"
    "  <- / a : LEFT     ^ / w : FORWARD     -> / d : RIGHT     space : FORWARD\n"
    "  q      : quit\n"
    "  (latching: a command stays until you press another)\n"
)


def read_key(fd: int, timeout: float = 0.1):
    """
    Return the next keypress as a string, or None on timeout. Arrow keys arrive
    as a 3-byte escape sequence (ESC [ A/B/C/D), read together here.

    Reads straight from the fd with os.read (not sys.stdin.read) so no bytes get
    stranded in Python's text buffer, which would split an escape sequence.
    """
    if not select.select([fd], [], [], timeout)[0]:
        return None
    ch = os.read(fd, 1).decode("utf-8", "ignore")
    if ch == "\x1b":
        # An arrow sends all 3 bytes at once; a 10 ms window reliably grabs the
        # "[A".."[D" continuation without adding felt latency to a lone ESC.
        if select.select([fd], [], [], 0.01)[0]:
            ch += os.read(fd, 2).decode("utf-8", "ignore")
    return ch


def main():
    rospy.init_node("keyboard_command", anonymous=False)
    topic = rospy.get_param("~command_topic", "/senpai/command")
    pub = rospy.Publisher(topic, String, queue_size=1, latch=True)

    command = rospy.get_param("~initial_command", "FORWARD")
    sys.stdout.write(HELP)
    sys.stdout.write(f"[keyboard_command] publishing to {topic}\n")

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        pub.publish(String(data=command))
        sys.stdout.write(f"\r[command] {command:8s}"); sys.stdout.flush()

        while not rospy.is_shutdown():
            key = read_key(fd)
            if key is None:
                continue
            if key in QUIT_KEYS:
                break
            new_command = KEY_TO_COMMAND.get(key.lower() if len(key) == 1 else key)
            if new_command is None:
                continue
            if new_command != command:
                command = new_command
                pub.publish(String(data=command))
            sys.stdout.write(f"\r[command] {command:8s}"); sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        sys.stdout.write("\n[keyboard_command] stopped\n")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
