#!/usr/bin/env python3
"""Replay a rosbag through realtime_planner_node_ff.py without a ROS master.

The node is written against rospy, but everything it needs from rospy at
runtime is parameters, logging, and publishers. This entry point supplies those
from a shim, reads /odom and the image topic straight out of the bag with the
rosbag API, and drives the same process() the live callback would.

Two deliberate differences from the live node:

* No frame is ever dropped. Live, cb_image returns early while a plan is still
  running; here every 0.5 s sample is planned in full, so a run is reproducible
  and two segmentation backends can be compared on exactly the same frames.
* Nothing is published. Publishers are no-ops, so build_path() and friends still
  execute (and would still raise on a bug) but no master is required.

Usage mirrors the live command lines -- the same ``_name:=value`` parameters
work, plus ``--bag``::

    python offline_bag_runner_ff.py --bag /path/to.bag \\
        _checkpoint:=model/0805_hybrid/last.ckpt \\
        _expected_model_variant:=hybrid \\
        _ego_input_mode:=fixed_speed _fixed_speed_mps:=1.0 \\
        _segmentation_backend:=twinlitenet \\
        _in_topic:=/zed2i/zed_node/right_raw/image_raw_color
"""

from pathlib import Path
import argparse
import sys
import time

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# --------------------------------------------------------------------------
# rospy shim
# --------------------------------------------------------------------------
def _parse_params(tokens):
    """Turn the ROS-style ``_name:=value`` argv into a {'~name': value} dict."""
    params = {}
    for token in tokens:
        if ":=" not in token:
            raise ValueError(f"Expected _name:=value, got {token!r}")
        name, _, raw = token.partition(":=")
        name = name.lstrip("_")
        params["~" + name] = _coerce(raw)
    return params


def _coerce(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


class _OfflineRospy:
    """Just enough rospy for the planner to construct and run off a master."""

    def __init__(self, params, verbose=True):
        self._params = dict(params)
        self._verbose = verbose
        self._throttled = {}
        # Duration/Time are plain genpy types and work with no master at all,
        # so the real ones are reused rather than reimplemented.
        import rospy as _real_rospy

        self.Duration = _real_rospy.Duration
        self.Time = _real_rospy.Time

    # ---- parameters ----
    def get_param(self, name, default=None):
        return self._params.get(name, default)

    def set_param(self, name, value):
        self._params[name] = value

    # ---- logging ----
    def _log(self, level, message):
        if self._verbose or level != "INFO":
            print(f"[{level}] {message}", flush=True)

    def loginfo(self, message):
        self._log("INFO", message)

    def logwarn(self, message):
        self._log("WARN", message)

    def logerr(self, message):
        self._log("ERROR", message)

    def _throttle(self, level, period, message):
        key = (level, message.split("\n", 1)[0][:80])
        now = time.monotonic()
        previous = self._throttled.get(key)
        if previous is None or now - previous >= period:
            self._throttled[key] = now
            self._log(level, message)

    def loginfo_throttle(self, period, message):
        self._throttle("INFO", period, message)

    def logwarn_throttle(self, period, message):
        self._throttle("WARN", period, message)

    def logerr_throttle(self, period, message):
        self._throttle("ERROR", period, message)

    # ---- node lifecycle and I/O, all inert offline ----
    def init_node(self, *args, **kwargs):
        return None

    def on_shutdown(self, *args, **kwargs):
        return None

    def spin(self):
        return None

    def is_shutdown(self):
        return False

    class Publisher:
        def __init__(self, *args, **kwargs):
            pass

        def publish(self, *args, **kwargs):
            return None

        def unregister(self):
            return None

    class Subscriber:
        def __init__(self, *args, **kwargs):
            pass

        def unregister(self):
            return None


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a rosbag through the FF planner with no ROS master."
    )
    parser.add_argument("--bag", required=True, help="path to the .bag file")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="stop after this many planned samples (0 = whole bag)",
    )
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="skip this many seconds from the start of the bag",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the node's INFO logging"
    )
    args, param_tokens = parser.parse_known_args()

    bag_path = Path(args.bag).expanduser()
    if not bag_path.is_file():
        parser.error(f"bag not found: {bag_path}")

    params = _parse_params(param_tokens)
    # Offline plots are the whole point of the run, so default them on.
    params.setdefault("~save_plots", True)
    shim = _OfflineRospy(params, verbose=not args.quiet)

    # Import the node entry point first: its module preamble is what fixes
    # sys.path and registers realtime_runtime_ff as park_L2_ASAP, and importing
    # realtime_planner_node ahead of it would pull the incompatible original.
    # Nothing touches rospy at import time, so patching afterwards is in time
    # for the constructor, which is the first thing to read a parameter.
    import realtime_planner_node_ff as ff

    # ff, twinlite_backend and the legacy node all reach rospy through this one
    # module object, so a single rebind covers all three.
    ff.legacy.rospy = shim

    node = ff.RealtimePlannerNodeFF()

    in_topic = shim.get_param("~in_topic", "/zed2i/zed_node/left/image_rect_color")
    odom_topic = shim.get_param("~odom_topic", "/odom")
    print(f"[offline] bag={bag_path}", flush=True)
    print(f"[offline] image={in_topic} odom={odom_topic}", flush=True)

    import rosbag

    planned = 0
    seen_images = 0
    t_start = time.perf_counter()
    with rosbag.Bag(str(bag_path), "r") as bag:
        available = set(bag.get_type_and_topic_info().topics)
        for name, label in ((in_topic, "~in_topic"), (odom_topic, "~odom_topic")):
            if name not in available:
                raise SystemExit(
                    f"{label}={name} is not in the bag. Available topics:\n  "
                    + "\n  ".join(sorted(available))
                )
        bag_start = bag.get_start_time()
        begin = None
        if args.start > 0.0:
            import rospy as _real_rospy

            begin = _real_rospy.Time.from_sec(bag_start + args.start)

        for topic, msg, _ in bag.read_messages(
            topics=[in_topic, odom_topic], start_time=begin
        ):
            if topic == odom_topic:
                node.cb_odom(msg)
                continue

            seen_images += 1
            stamp = msg.header.stamp.to_sec()
            # The camera-rate pre-pass runs on every image, exactly as cb_image
            # does live; only the cadence gate below differs.
            node._twinlite_prepass(msg, stamp)

            if node.last_odom is None:
                continue
            if (
                node.last_sample_time is not None
                and stamp - node.last_sample_time < node.sample_interval
            ):
                continue

            node.process(msg)
            node.last_sample_time = stamp
            planned += 1
            if args.max_samples and planned >= args.max_samples:
                print(f"[offline] stopping at --max-samples={args.max_samples}", flush=True)
                break

    node._flush_pending(force=True)
    elapsed = time.perf_counter() - t_start
    print(
        f"[offline] images={seen_images} planned={planned} "
        f"wall={elapsed:.1f}s ({elapsed / max(planned, 1):.2f}s per sample)",
        flush=True,
    )
    plot_dir = getattr(node, "_plot_dir", None)
    if plot_dir is not None:
        print(f"[offline] plots -> {plot_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
