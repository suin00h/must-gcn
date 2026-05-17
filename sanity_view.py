"""Visual sanity check: overlay HRNet 2D skeletons on the 3 NTU camera views.

Run from the project root:
    python sanity_view.py                              # default sample (xsub_val[0])
    python sanity_view.py --idx 42                     # pick by dataloader index
    python sanity_view.py --frame_dir S001P001R001A001 # pick by (S,P,R,A)
    python sanity_view.py --idx 42 --anim --traj       # add GIF + trajectory plots

Data locations are resolved from $DATA_ROOT (default: ./data); see README.
Outputs (in sanity_output/):
    multiview_<key>.png   static 3-view skeleton overlay on RGB (always)
    anim_<key>.mp4/.gif   animated 3-view skeleton over the action, on the
                          original RGB video where available (--anim)
    traj_<key>.png        per-joint x/y/HRNet-score over time, per camera (--traj)
"""

import argparse
import os
import pickle
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feeders.feeder_ntu_multiview import MultiviewFeeder, parse_name


DATA_ROOT = os.environ.get('DATA_ROOT', 'data')
PKL = os.path.join(DATA_ROOT, 'ntu-rgbd', 'processed', 'hrnet', 'ntu60_hrnet.pkl')
VIDEO_ROOT = os.path.join(DATA_ROOT, 'ntu-rgbd', 'videos')

# COCO-17 skeleton edges + colour scheme
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),                  # face
    (5, 7), (7, 9), (6, 8), (8, 10),                 # arms
    (5, 6), (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # legs
]
LIMB_COLOURS = (['#ffd700'] * 4              # face
              + ['#00bfff'] * 4              # arms
              + ['#ffffff'] * 4              # torso
              + ['#ff69b4'] * 4)             # legs
SCORE_THR = 0.1

# COCO-17 joint names (0-indexed), for the trajectory plot.
COCO17_NAMES = [
    'nose', 'eye-L', 'eye-R', 'ear-L', 'ear-R',
    'shoulder-L', 'shoulder-R', 'elbow-L', 'elbow-R',
    'wrist-L', 'wrist-R', 'hip-L', 'hip-R',
    'knee-L', 'knee-R', 'ankle-L', 'ankle-R',
]

# NTU60 action names (0-indexed).
NTU60 = [
    'drink water', 'eat meal', 'brushing teeth', 'brushing hair', 'drop',
    'pickup', 'throw', 'sitting down', 'standing up', 'clapping',
    'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket',
    'wear a shoe', 'take off a shoe', 'wear glasses', 'take off glasses', 'put on hat',
    'take off hat', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket',
    'hopping', 'jump up', 'phone call', 'play with phone', 'typing',
    'pointing', 'taking selfie', 'check time', 'rub hands', 'nod head',
    'shake head', 'wipe face', 'salute', 'palms together', 'cross hands stop',
    'sneeze/cough', 'staggering', 'falling', 'touch head', 'touch chest',
    'touch back', 'touch neck', 'nausea', 'use fan', 'punch/slap other',
    'kick other', 'push other', 'pat back of other', 'point at other', 'hug other',
    'give to other', 'touch other pocket', 'handshake', 'walk towards', 'walk apart',
]


def load_pkl_groups(pkl_path: str, split: str):
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    annos = {a['frame_dir']: a for a in raw['annotations']}
    wanted = set(raw['split'][split])
    groups = defaultdict(dict)
    for name in wanted:
        parsed = parse_name(name)
        if parsed is None:
            continue
        s, c, p, r, a = parsed
        groups[(s, p, r, a)][c] = name
    complete = {k: v for k, v in groups.items() if {1, 2, 3}.issubset(v.keys())}
    return annos, complete


def load_video_frame(frame_dir: str, frame_idx: int):
    path = f'{VIDEO_ROOT}/{frame_dir}_rgb.avi'
    if not os.path.exists(path):
        return None, f'video not found: {path}'
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None, f'cv2 failed to read frame {frame_idx} from {path}'
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None


def draw_view(ax, anno, frame_dir, frame_idx, person_colours=('red', 'orange')):
    img, err = load_video_frame(frame_dir, frame_idx)
    H, W = anno['img_shape']
    if img is not None:
        ax.imshow(img)
    else:
        ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax.text(0.5, 0.5, err, transform=ax.transAxes, color='red',
                ha='center', va='center', fontsize=9)

    kp = anno['keypoint'][:, frame_idx].astype(np.float32)            # (M, 17, 2)
    score = anno['keypoint_score'][:, frame_idx].astype(np.float32)   # (M, 17)
    for m in range(kp.shape[0]):
        valid = score[m] > SCORE_THR
        # bones
        for (i, j), col in zip(SKELETON, LIMB_COLOURS):
            if valid[i] and valid[j]:
                ax.plot([kp[m, i, 0], kp[m, j, 0]],
                        [kp[m, i, 1], kp[m, j, 1]],
                        color=col, lw=2.0, alpha=0.9)
        # joints
        ax.scatter(kp[m, valid, 0], kp[m, valid, 1],
                   s=18, c=person_colours[m % 2],
                   edgecolors='black', linewidths=0.4, zorder=5)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_pose(ax, kp, score, H, W, label, bg=None, person_colours=('red', 'orange')):
    """Draw COCO-17 skeletons for one frame.
    kp: (M, 17, 2)   score: (M, 17)   bg: optional (H, W, 3) RGB background."""
    ax.clear()
    if bg is not None:
        ax.imshow(bg, extent=[0, W, H, 0], aspect='auto')
    else:
        ax.set_facecolor('#11131f')
    for m in range(kp.shape[0]):
        valid = score[m] > SCORE_THR
        for (i, j), col in zip(SKELETON, LIMB_COLOURS):
            if valid[i] and valid[j]:
                ax.plot([kp[m, i, 0], kp[m, j, 0]],
                        [kp[m, i, 1], kp[m, j, 1]], color=col, lw=2.2, alpha=0.9)
        ax.scatter(kp[m, valid, 0], kp[m, valid, 1], s=16,
                   c=person_colours[m % 2], edgecolors='black',
                   linewidths=0.3, zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect('equal'); ax.axis('off')
    # label as in-axes text — no title margin
    ax.text(0.015, 0.04, label, transform=ax.transAxes, color='white',
            fontsize=8.5, va='top', family='monospace',
            bbox=dict(facecolor='black', alpha=0.55, pad=1.6, edgecolor='none'))


def _load_video_frames(frame_dir, n):
    """Sequentially read up to n RGB frames from <frame_dir>_rgb.avi, or None."""
    path = f'{VIDEO_ROOT}/{frame_dir}_rgb.avi'
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames or None


def save_multiview_anim(annos_per_view, cam_to_name, base, label_name, out_dir,
                        fps=12, max_frames=150):
    """Animated clip: the 3 camera views' HRNet skeletons over the whole action,
    overlaid on the original RGB video wherever the video files are available.
    Saved as .mp4 when ffmpeg is available (small), else a downscaled .gif."""
    import shutil
    import matplotlib.animation as animation
    cams = sorted(annos_per_view)
    T = min(min(a['total_frames'] for a in annos_per_view.values()), max_frames)
    kp = {c: annos_per_view[c]['keypoint'].astype(np.float32) for c in cams}
    sc = {c: annos_per_view[c]['keypoint_score'].astype(np.float32) for c in cams}
    hw = {c: annos_per_view[c]['img_shape'] for c in cams}
    vid = {c: _load_video_frames(cam_to_name[c], T) for c in cams}
    n_vid = sum(v is not None for v in vid.values())

    # mp4 compresses video content ~10x better than gif; the gif fallback is
    # downscaled and frame-subsampled to keep the file size reasonable.
    use_ff = shutil.which('ffmpeg') is not None
    ext, writer = ('mp4', 'ffmpeg') if use_ff else ('gif', 'pillow')
    stride = 1 if use_ff else max(1, round(T / 40))
    mult = 6.2 if use_ff else 4.2
    frame_ids = list(range(0, T, stride))
    print(f'  anim: {T} frames (stride {stride} -> {len(frame_ids)}), '
          f'{n_vid}/{len(cams)} views with video -> .{ext}')

    # figure aspect = the frame's H/W, so equal-aspect axes leave no whitespace
    asp = hw[cams[0]][0] / hw[cams[0]][1]
    fig, axes = plt.subplots(1, len(cams), figsize=(mult * len(cams), mult * asp),
                             facecolor='black')
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.015)

    def update(t):
        for ax, c in zip(axes, cams):
            H, W = hw[c]
            bg = vid[c][t] if (vid[c] is not None and t < len(vid[c])) else None
            tag = f'cam {c}  ·  {t + 1}/{T}'
            if c == cams[0]:
                tag = f'{base}   {label_name}\n' + tag
            _draw_pose(ax, kp[c][:, t], sc[c][:, t], H, W, tag, bg=bg)
        return list(axes)

    ani = animation.FuncAnimation(fig, update, frames=frame_ids,
                                  interval=1000 // fps, blit=False)
    out = f'{out_dir}/anim_{base}.{ext}'
    ani.save(out, writer=writer, fps=fps)
    plt.close(fig)
    print(f'saved  : {out}')


def save_trajectory(annos_per_view, base, out_dir, joints=(0, 9, 10, 15, 16),
                    person=0):
    """Per-joint x / y / HRNet-score over time, one line per camera — reveals
    jitter, dropout, and per-view confidence loss (the occlusion signal)."""
    cams = sorted(annos_per_view)
    T = min(a['total_frames'] for a in annos_per_view.values())
    cam_col = {1: 'tab:blue', 2: 'tab:green', 3: 'tab:red'}
    t = np.arange(T)

    fig, axes = plt.subplots(len(joints), 3, figsize=(15, 2.6 * len(joints)),
                             sharex=True, squeeze=False)
    for row, jid in enumerate(joints):
        for c in cams:
            kp = annos_per_view[c]['keypoint'][person].astype(np.float32)       # (T,17,2)
            sc = annos_per_view[c]['keypoint_score'][person].astype(np.float32) # (T,17)
            axes[row, 0].plot(t, kp[:T, jid, 0], lw=1, color=cam_col.get(c, 'gray'),
                              label=f'cam {c}')
            axes[row, 1].plot(t, kp[:T, jid, 1], lw=1, color=cam_col.get(c, 'gray'))
            axes[row, 2].plot(t, sc[:T, jid],    lw=1, color=cam_col.get(c, 'gray'))
        axes[row, 0].set_ylabel(COCO17_NAMES[jid], fontsize=9)
        axes[row, 2].axhline(SCORE_THR, color='gray', ls='--', lw=0.6)
        axes[row, 2].set_ylim(0, 1)
    axes[0, 0].set_title('x (px)'); axes[0, 1].set_title('y (px)')
    axes[0, 2].set_title('HRNet score')
    axes[0, 0].legend(fontsize=7, ncol=3)
    for k in range(3):
        axes[-1, k].set_xlabel('frame')
    fig.suptitle(f'{base}  —  per-joint trajectory & confidence (person {person})',
                 fontsize=11)
    plt.tight_layout()
    out = f'{out_dir}/traj_{base}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved  : {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='xsub_val')
    p.add_argument('--idx',   type=int, default=0,
                   help='index into the multiview dataset (after 3-view filtering)')
    p.add_argument('--frame_dir', default=None,
                   help='override --idx with a specific S###P###R###A### key')
    p.add_argument('--out_dir', default='sanity_output')
    p.add_argument('--anim', action='store_true',
                   help='also save an animated multi-view skeleton GIF')
    p.add_argument('--traj', action='store_true',
                   help='also save per-joint x/y/HRNet-score trajectory plots')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    annos, complete = load_pkl_groups(PKL, args.split)
    keys = sorted(complete.keys())

    if args.frame_dir is not None:
        import re
        m = re.match(r'^S(\d{3})(?:C\d{3})?P(\d{3})R(\d{3})A(\d{3})$', args.frame_dir)
        assert m, f'bad --frame_dir format: {args.frame_dir!r} (expected e.g. S001P003R001A050)'
        s, p_, r, a = (int(x) for x in m.groups())
        key = (s, p_, r, a)
        assert key in complete, f'{key} not a 3-view-complete sample in {args.split}'
    else:
        key = keys[args.idx]

    cam_to_name = complete[key]
    annos_per_view = {c: annos[name] for c, name in cam_to_name.items()}
    min_T = min(a['total_frames'] for a in annos_per_view.values())
    frame_idx = min_T // 2
    label = annos_per_view[1]['label']
    label_name = NTU60[label] if label < len(NTU60) else f'class {label}'

    s, p_, r, a = key
    base = f'S{s:03d}P{p_:03d}R{r:03d}A{a:03d}'
    print(f'sample : {base}    (label={label} · {label_name!r})')
    print(f'frames : {[a["total_frames"] for a in annos_per_view.values()]}    '
          f'using frame_idx={frame_idx}')

    # cross-check via the dataloader (no random crop, want it deterministic)
    ds = MultiviewFeeder(PKL, split=args.split, window_size=64, random_crop=False)
    # find the same key in the dataset list
    ds_idx = next(i for i, (k, _) in enumerate(ds.groups) if k == key)
    data, ds_label, ds_key = ds[ds_idx]
    print(f'loader : data.shape={data.shape}  label={ds_label}  key={ds_key}')

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='black')
    fig.suptitle(f'{base}   |   action {label}: {label_name}',
                 color='white', fontsize=14, y=0.97)
    for col, c in enumerate([1, 2, 3]):
        anno = annos_per_view[c]
        draw_view(axes[col], anno, cam_to_name[c], frame_idx)
        axes[col].set_title(f'cam {c}   ({anno["total_frames"]} frames)',
                            color='white', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = f'{args.out_dir}/multiview_{base}.png'
    plt.savefig(out, dpi=140, facecolor='black', bbox_inches='tight')
    print(f'saved  : {out}')

    if args.anim:
        save_multiview_anim(annos_per_view, cam_to_name, base, label_name, args.out_dir)
    if args.traj:
        save_trajectory(annos_per_view, base, args.out_dir)


if __name__ == '__main__':
    main()
