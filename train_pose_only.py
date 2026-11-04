"""Pose-only pretraining on matched GT cylinders for the joint model.

Place beside the existing matched_pair_pose.py, dataset.py and losses.py.
Only MatchedPairPoseModel is constructed. No RGB images are opened.
Tokens contain GT angles/radii/depths paired by GT cylinder ID.
Training starts from scratch; --eval-only evaluates a saved pose checkpoint.
The saved model_state_dict can be loaded using the joint model's
load_pose_checkpoint() in the later fine-tuning stage.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import SceneTwoPairsDataset
from losses import relative_pose_loss_2d
from matched_pair_pose import MatchedPairPoseModel, matched_tokens, has_distinct_centers


class MatchedCylinderDataset(Dataset):
    """Reuse the existing sample order and pose convention, without opening RGB.

    Uses samples and _relative_pose_2d from the supplied SceneTwoPairsDataset.
    Each kept original sample index is recorded for fair subset comparisons.
    """
    def __init__(self, root, fov_degrees=90., occ_thresh=0.5):
        base = SceneTwoPairsDataset(root_dir=root, return_two_pairs=False)
        self.items, self.manifest = [], []
        stats = dict(original_pairs=len(base), kept_pairs=0, too_few_matches=0,
                     degenerate_centers=0, zero_baseline=0, ambiguous_cylinder_ids=0)
        for index, sample in enumerate(base.samples):
            pair = sample.pair
            tokens, ambiguous = matched_tokens(pair['vision1'], pair['vision2'], fov_degrees, occ_thresh)
            stats['ambiguous_cylinder_ids'] += ambiguous
            if len(tokens) < 2:
                stats['too_few_matches'] += 1
                continue
            if not has_distinct_centers(tokens):
                stats['degenerate_centers'] += 1
                continue
            camera_a = pair.get('camera1', sample.scene_camera1)
            camera_b = pair.get('camera2', sample.scene_camera2)
            if camera_a is None or camera_b is None:
                raise ValueError(f'Missing cameras for sample {index}')
            pose = base._relative_pose_2d(camera_a, camera_b)
            if not torch.isfinite(pose).all():
                raise ValueError(f'Nonfinite pose for sample {index}')
            if pose[:2].norm() <= 1e-6:
                stats['zero_baseline'] += 1
                continue
            self.items.append((torch.from_numpy(tokens), pose, index))
            self.manifest.append(dict(sample_index=index, scene=str(sample.scene_dir),
                                      image1=str(pair['image1']), image2=str(pair['image2']),
                                      matched_cylinders=len(tokens)))
        stats['kept_pairs'] = len(self.items)
        self.stats = stats
        print(f'Dataset {root}: {stats}', flush=True)
        if not self.items:
            raise ValueError('No eligible pairs: need >=2 unique shared cylinder IDs and a nonzero camera baseline')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_pairs(batch):
    pairs, poses, original_indices = zip(*batch)
    lengths = torch.tensor([len(x) for x in pairs])
    padded = pad_sequence(pairs, batch_first=True)
    valid = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, valid, torch.stack(poses), torch.tensor(original_indices)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals = dict(pose=0., translation=0., rotation=0.)
    n = 0
    for pairs, valid, target, _ in loader:
        pairs, valid, target = pairs.to(device), valid.to(device), target.to(device)
        
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(pairs, valid)
            loss, t_loss, r_loss = relative_pose_loss_2d(pred, target)
            
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pose loss')
            
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        
        n += len(target)
        
        for key, value in zip(totals, (loss, t_loss, r_loss)):
            totals[key] += value.detach().item() * len(target)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def evaluate_pose(model, loader, device):
    model.eval()
    t_errors, yaw_errors, indices = [], [], []
    total = failures = 0
    for pairs, valid, target, original_indices in loader:
        pred = model(pairs.to(device), valid.to(device)).cpu()
        for p, gt, index in zip(pred, target, original_indices):
            total += 1
            if (not torch.isfinite(p).all() or p[:2].norm() < 1e-8
                    or p[2:].norm() < 1e-8):
                failures += 1
                continue
            t_pred = torch.nn.functional.normalize(p[:2], dim=0)
            t_gt = torch.nn.functional.normalize(gt[:2], dim=0)
            t_errors.append(torch.rad2deg(torch.acos((t_pred * t_gt).sum().clamp(-1, 1))).item())
            delta = torch.atan2(p[2], p[3]) - torch.atan2(gt[2], gt[3])
            yaw_errors.append(torch.rad2deg(torch.atan2(delta.sin(), delta.cos())).abs().item())
            indices.append(index.item())
    t, yaw = np.asarray(t_errors), np.asarray(yaw_errors)
    def stat(x, fn):
        return float(fn(x)) if len(x) else float('nan')
    print('CAMERA POSE PERFORMANCE\n' + '=' * 60)
    print(f'\nTest pairs:             {total}')
    print(f'Successful poses:       {len(t)}')
    print(f'Prediction failures:    {failures}')
    print(f'Success rate:           {len(t) / max(total, 1):.3f}')
    result = dict(total=total, failures=failures)
    for prefix, label, values in [('translation', 'TRANSLATION DIRECTION ERROR', t),
                                  ('yaw', 'YAW ERROR', yaw)]:
        print('\n' + label + '\n' + '-' * 40)
        for key, title, fn in [('mean', 'Mean', np.mean), ('median', 'Median', np.median),
                               ('p90', 'P90', lambda x: np.percentile(x, 90))]:
            result[f'{prefix}_{key}'] = stat(values, fn)
            print(f'{title + ":":24s}{result[f"{prefix}_{key}"]:.2f}°')
    print('\nACCURACY THRESHOLDS\n' + '-' * 40)
    for threshold in (5, 10, 20, 30):
        ta = stat(t, lambda x: np.mean(x < threshold))
        ya = stat(yaw, lambda x: np.mean(x < threshold))
        print(f'< {threshold:2d}°: T={100*ta:5.1f}% | Yaw={100*ya:5.1f}%')
    return result, dict(translation_errors=t, yaw_errors=yaw, sample_indices=np.asarray(indices))


def model_config(args):
    return {key: getattr(args, key) for key in (
        'embed_dim', 'num_heads', 'depth', 'dropout', 'fov_degrees',
        'radius_scale', 'distance_scale')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/nobackup/proj/disk/midlevel_representations/personal/johanna/data')
    parser.add_argument('--output-dir', default='runs/pose_only')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-interval', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fov-degrees', type=float, default=90.)
    parser.add_argument('--radius-scale', type=float, default=1.)
    parser.add_argument('--distance-scale', type=float, default=10.)
    parser.add_argument('--occ-thresh', type=float, default=0.5)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split', default='testdataset', help='Dataset subfolder for --eval-only')
    args = parser.parse_args()
    if args.checkpoint and not args.eval_only:
        parser.error('--checkpoint is for --eval-only; training starts from scratch')
    if min(args.epochs, args.batch_size, args.val_interval, args.save_every) < 1:
        parser.error('Epochs, batch size and intervals must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print('Device:', device)
    print('Seed:', args.seed)
    print('Training stage: pose only, GT geometry and GT correspondences')

    config = model_config(args)
    occ_thresh = args.occ_thresh
    checkpoint = None
    if args.eval_only:
        if not args.checkpoint:
            parser.error('--eval-only requires --checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint['model_config']
        occ_thresh = checkpoint['args']['occ_thresh']
    model = MatchedPairPoseModel(**config).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    def make_loader(split, shuffle):
        ds = MatchedCylinderDataset(Path(args.data_dir) / split,
                                    config['fov_degrees'], occ_thresh)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                            num_workers=0, collate_fn=collate_pairs,
                            pin_memory=(device.type == 'cuda'))
        # Preprocessed labels are already in memory; no RGB/data workers needed.
        with (output / f'{Path(split).name}_subset.json').open('w') as f:
            json.dump(dict(stats=ds.stats, kept_samples=ds.manifest), f, indent=2)
        return loader

    if args.eval_only:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print('Checkpoint epoch:', checkpoint['epoch'])
        loader = make_loader(args.split, False)
        result, arrays = evaluate_pose(model, loader, device)
        with (output / 'pose_metrics.json').open('w') as f:
            json.dump(result, f, indent=2)
        np.savez(output / 'pose_errors.npz', **arrays)
        return

    train_loader, val_loader = make_loader('dataset', True), make_loader('valdataset', False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    ckpt_dir = output / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    history = {key: [] for key in ('epoch', 'pose', 'translation', 'rotation',
                                  'val_epoch', 'val_pose', 'val_translation', 'val_rotation')}
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, train_loader, device, optimizer)
        history['epoch'].append(epoch)
        
        for key, value in train.items():
            history[key].append(value)
        log = f'Epoch {epoch}: ' + ' | '.join(f'{k}={v:.4f}' for k, v in train.items())
        improved = False
        
        if epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs:
            val = run_epoch(model, val_loader, device)
            history['val_epoch'].append(epoch)
            for key, value in val.items():
                history['val_' + key].append(value)
            log += ' | ' + ' | '.join(f'val_{k}={v:.4f}' for k, v in val.items())
            improved = val['pose'] < best
            if improved:
                best = val['pose']
                log += ' | BEST'
        
        state = dict(epoch=epoch, model_state_dict=model.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(), model_config=config,
                     args=vars(args), history=history, best_val_loss=best,
                     training_stage='pose_only_gt')
        torch.save(state, ckpt_dir / 'latest.pt')
        
        if improved:
            torch.save(state, ckpt_dir / 'best.pt')
        if epoch % args.save_every == 0:
            torch.save(state, ckpt_dir / f'epoch_{epoch:03d}.pt')
        with (output / 'history.json').open('w') as f:
            json.dump(history, f, indent=2)
        print(log + f' | time={time.perf_counter() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
