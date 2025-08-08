import argparse, os, json
import numpy as np
import cv2

DEFAULT_ACTION_NAMES = [f"a{i}" for i in range(17)]


def load_episode(npz_path):
    d = np.load(npz_path)
    return d["observations"], d["actions"], d["rewards"], d["dones"]


def draw_text(img, text, org, font_scale=0.5, thickness=1):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def overlay_hud(frame_bgr, step, action_id, action_name, reward, ret, done, bar_h=10):
    h, w = frame_bgr.shape[:2]
    y0 = 20
    draw_text(frame_bgr, f"step: {step}", (8, y0))
    draw_text(frame_bgr, f"action: {action_id} ({action_name})", (8, y0 + 18))
    draw_text(frame_bgr, f"reward: {reward:+.3f}", (8, y0 + 36))
    draw_text(frame_bgr, f"return: {ret:+.3f}", (8, y0 + 54))
    if done:
        draw_text(frame_bgr, "DONE", (w - 70, 20), font_scale=0.6, thickness=2)
    rw = max(-1.0, min(1.0, float(reward)))
    mid = w // 2
    pos_len = int(max(0, rw) * (w // 2))
    neg_len = int(max(0, -rw) * (w // 2))
    cv2.rectangle(frame_bgr, (0, h - bar_h), (w, h), (35, 35, 35), -1)
    if neg_len > 0:
        cv2.rectangle(frame_bgr, (mid - neg_len, h - bar_h), (mid, h), (0, 0, 180), -1)
    if pos_len > 0:
        cv2.rectangle(frame_bgr, (mid, h - bar_h), (mid + pos_len, h), (0, 180, 0), -1)
    cv2.line(frame_bgr, (mid, h - bar_h), (mid, h), (200, 200, 200), 1)
    return frame_bgr


def _open_writer(path, w, h, fps, try_codecs):
    for ext, fourcc_str in try_codecs:
        out_path = (
            path if path.lower().endswith(ext) else os.path.splitext(path)[0] + ext
        )
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, out_path, fourcc_str
    return None, None, None


def visualize_episode(
    npz_path,
    out_path=None,
    fps=15,
    scale=3,
    action_names=None,
    preview=False,
    codec="auto",
):
    obs, actions, rewards, dones = load_episode(npz_path)
    T, H, W, C = obs.shape
    assert C == 3, f"Expected (T,64,64,3) RGB, got {obs.shape}"
    action_names = action_names or DEFAULT_ACTION_NAMES
    if len(action_names) < int(actions.max()) + 1:
        action_names = list(action_names) + [
            f"a{i}" for i in range(len(action_names), int(actions.max()) + 1)
        ]

    out_h, out_w = H * scale, W * scale

    if codec == "auto":
        tries = [
            (".mp4", "mp4v"),
            (".mp4", "avc1"),
            (".avi", "MJPG"),
        ]
    elif codec.lower() == "avi":
        tries = [(".avi", "MJPG")]
    elif codec.lower() in ("mp4v", "avc1", "h264"):
        ext = ".mp4"
        four = "avc1" if codec.lower() in ("avc1", "h264") else "mp4v"
        tries = [(ext, four)]
    else:
        raise ValueError(f"Unknown codec option: {codec}")

    target = out_path or os.path.splitext(npz_path)[0] + ".mp4"
    writer, final_out, used = _open_writer(target, out_w, out_h, fps, tries)
    if writer is None:
        raise RuntimeError(
            "Failed to open VideoWriter. Try:\n"
            " - Installing FFmpeg so OpenCV can write MP4s\n"
            " - Using --codec avi (writes Motion-JPEG .avi)\n"
            " - Matching extension to codec (e.g., .avi for MJPG)"
        )
    print(f"[writer] -> {final_out} ({used}) @ {fps} fps, {out_w}x{out_h}")

    ret = 0.0
    frames_written = 0
    for t in range(T):
        frame_rgb = obs[t]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(
            frame_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
        ret += float(rewards[t])
        a_id = int(actions[t])
        a_name = action_names[a_id] if 0 <= a_id < len(action_names) else f"a{a_id}"
        frame_bgr = overlay_hud(
            frame_bgr, t, a_id, a_name, float(rewards[t]), ret, bool(dones[t])
        )
        frame_bgr = np.ascontiguousarray(frame_bgr)
        writer.write(frame_bgr)
        frames_written += 1

        if preview:
            cv2.imshow("episode", frame_bgr)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break

    writer.release()
    if preview:
        cv2.destroyAllWindows()

    if frames_written == 0:
        raise RuntimeError(
            "No frames written. Check that observations are uint8 RGB and the writer opened correctly."
        )
    print(f"Wrote {frames_written} frames -> {final_out}")


def main():
    p = argparse.ArgumentParser(
        description="Visualize a saved Crafter episode_XXXX.npz as a video with overlays."
    )
    p.add_argument("input", help="Path to episode_XXXX.npz")
    p.add_argument(
        "--out",
        default=None,
        help="Output video path (extension may change based on codec)",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument(
        "--action-names",
        default=None,
        help='JSON list or path to JSON file for action names, e.g. \'["noop","up",...]\' or file.json',
    )
    p.add_argument("--preview", action="store_true")
    p.add_argument(
        "--codec",
        default="auto",
        choices=["auto", "avi", "mp4v", "avc1", "h264"],
        help="Force a specific container/codec or let it auto-fallback",
    )
    args = p.parse_args()

    action_names = None
    if args.action_names:
        if os.path.isfile(args.action_names):
            with open(args.action_names, "r") as f:
                action_names = json.load(f)
        else:
            action_names = json.loads(args.action_names)

    os.makedirs(os.path.dirname(args.out or args.input) or ".", exist_ok=True)
    visualize_episode(
        args.input,
        out_path=args.out,
        fps=args.fps,
        scale=args.scale,
        action_names=action_names,
        preview=args.preview,
        codec=args.codec,
    )


if __name__ == "__main__":
    main()
