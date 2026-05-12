"""Visual sanity check: overlay HRNet 2D skeletons on the 3 NTU camera views.

Run from the project root:
    python sanity_view.py                              # default sample (xsub_val[0])
    python sanity_view.py --idx 42                     # pick by dataloader index
    python sanity_view.py --frame_dir S001P001R001A001 # pick by (S,P,R,A)

Data locations are resolved from $DATA_ROOT (default: ./data); see README.
Output: sanity_output/multiview_<frame_dir>.png
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='xsub_val')
    p.add_argument('--idx',   type=int, default=0,
                   help='index into the multiview dataset (after 3-view filtering)')
    p.add_argument('--frame_dir', default=None,
                   help='override --idx with a specific S###P###R###A### key')
    p.add_argument('--out_dir', default='sanity_output')
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


if __name__ == '__main__':
    main()
