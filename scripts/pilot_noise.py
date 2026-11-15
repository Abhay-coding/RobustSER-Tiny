"""Paired fresh-head MUSAN pilot on a fixed subset of CREMA-D training speakers."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from robustser.augmentation.noise import mix_at_snr, noise_rows_from_manifest
from robustser.data.cremad import TASK_LABELS, load_split_rows, read_pcm16_mono, read_pcm16_mono_segment
from robustser.features.log_mel import fit_audio_window
from robustser.training import set_global_seed
from robustser.training.metrics import classification_metrics

LABELS = TASK_LABELS['crema_d_six_emotion']
MANIFEST = 'data/manifests/crema_d_speaker_splits.csv'
NOISE = 'data/manifests/musan_noise_splits.csv'
SPLIT = Path('configs/experiments/musan_pilot_split.json')

def get_split(rows):
    speakers = sorted({r['speaker_id'] for r in rows})
    if SPLIT.exists():
        split = json.loads(SPLIT.read_text(encoding='utf-8'))
    else:
        if len(speakers) != 64:
            raise ValueError(f'Expected 64 training speakers, found {len(speakers)}')
        shuffled = np.random.default_rng(20261108).permutation(speakers).tolist()
        split = {'train_speakers': shuffled[:51], 'validation_speakers': shuffled[51:], 'split_seed': 20261108}
        SPLIT.write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')
    a, b = set(split['train_speakers']), set(split['validation_speakers'])
    if len(a) != 51 or len(b) != 13 or a & b or a | b != set(speakers):
        raise ValueError('Pilot speaker split is invalid or does not match the manifest')
    for group in (a, b):
        if {r['emotion'] for r in rows if r['speaker_id'] in group} != set(LABELS):
            raise ValueError('Pilot split lacks an emotion label')
    return a, b

class Audio(Dataset):
    def __init__(self, rows, seed, noise_mode):
        self.rows, self.seed, self.epoch, self.noise_mode = rows, seed, 0, noise_mode
        self.noise = noise_rows_from_manifest(NOISE, split='augmentation_train' if noise_mode == 'train' else 'evaluation_holdout') if noise_mode else []
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        audio = fit_audio_window(read_pcm16_mono(row['path']), target_samples=48000, policy='center_crop_or_zero_pad')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 42]))
        if self.noise_mode and (self.noise_mode == 'fixed' or rng.random() < 0.5):
            source = self.noise[int(rng.integers(len(self.noise)))]
            noise = read_pcm16_mono_segment(source['relative_path'], length=len(audio), rng=rng)
            snr = 10.0 if self.noise_mode == 'fixed' else float(rng.uniform(5, 20))
            audio = mix_at_snr(audio, noise, snr_db=snr, rng=rng)
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        return torch.from_numpy(audio), LABELS.index(row['emotion'])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', choices=['clean', 'noise'], required=True)
    p.add_argument('--seed', type=int, default=20261108)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.run_dir.exists():
        raise SystemExit(f'Run directory already exists: {args.run_dir}')
    if args.epochs < 1 or args.patience < 1:
        p.error('epochs and patience must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable')
    from transformers import Wav2Vec2Model
    set_global_seed(args.seed, deterministic_algorithms=True)
    rows = load_split_rows(MANIFEST, split='train', task='crema_d_six_emotion')
    train_ids, val_ids = get_split(rows)
    train_rows = [r for r in rows if r['speaker_id'] in train_ids]
    val_rows = [r for r in rows if r['speaker_id'] in val_ids]
    train = Audio(train_rows, args.seed, 'train' if args.branch == 'noise' else None)
    clean_val = Audio(val_rows, 20261108, None)
    noisy_val = Audio(val_rows, 20261108, 'fixed')
    encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', local_files_only=True).to(device)
    for param in encoder.parameters(): param.requires_grad = False
    for layer in encoder.encoder.layers[-8:]:
        for param in layer.parameters(): param.requires_grad = True
    head = nn.Sequential(nn.Linear(encoder.config.hidden_size * 2, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, len(LABELS))).to(device)
    optimizer = torch.optim.AdamW([{'params': [p for p in encoder.parameters() if p.requires_grad], 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-4}], weight_decay=.01)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=2, shuffle=True, generator=generator)
    args.run_dir.mkdir(parents=True)
    best, best_epoch = -1., 0
    def evaluate(data):
        encoder.eval(); head.eval(); predictions, targets = [], []
        with torch.no_grad():
            # DataLoader draws a base seed even without shuffling. Keep that
            # draw separate from the model's training/dropout random stream.
            eval_generator = torch.Generator().manual_seed(20261108)
            for audio, labels in DataLoader(data, batch_size=2, generator=eval_generator):
                hidden = encoder(audio.to(device)).last_hidden_state
                features = torch.cat((hidden.mean(1), hidden.std(1, unbiased=False)), dim=-1)
                predictions.append(head(features).argmax(1).cpu()); targets.append(labels)
        return classification_metrics(total_loss=0, targets=torch.cat(targets), predictions=torch.cat(predictions), num_classes=len(LABELS))
    for epoch in range(1, args.epochs + 1):
        train.epoch = epoch
        encoder.train(); head.train(); optimizer.zero_grad(); count = 0
        print(f'{args.branch} epoch {epoch}: {len(loader)} batches', flush=True)
        for i, (audio, labels) in enumerate(loader):
            hidden = encoder(audio.to(device)).last_hidden_state
            features = torch.cat((hidden.mean(1), hidden.std(1, unbiased=False)), dim=-1)
            nn.functional.cross_entropy(head(features), labels.to(device), reduction='sum').backward()
            count += len(labels)
            if (i + 1) % 8 == 0 or i + 1 == len(loader):
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None: param.grad.div_(count)
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 1.)
                optimizer.step(); optimizer.zero_grad(); count = 0
            if (i + 1) % 100 == 0: print(f'  batch {i+1}/{len(loader)}', flush=True)
        metrics = evaluate(clean_val)
        noisy_metrics = evaluate(noisy_val)
        record = {'epoch': epoch, 'clean_validation_macro_f1': metrics.macro_f1, 'per_class_f1': metrics.per_class_f1, 'noisy_10db_macro_f1': noisy_metrics.macro_f1, 'noisy_per_class_f1': noisy_metrics.per_class_f1}
        with (args.run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as out: out.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if metrics.macro_f1 > best:
            best, best_epoch = metrics.macro_f1, epoch
            torch.save({'encoder': encoder.state_dict(), 'head': head.state_dict(), 'epoch': epoch}, args.run_dir / 'best.pt')
        elif epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(args.run_dir / 'best.pt', map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder']); head.load_state_dict(checkpoint['head'])
    clean, noisy = evaluate(clean_val), evaluate(noisy_val)
    summary = {'status': 'completed_pilot', 'branch': args.branch, 'seed': args.seed, 'best_epoch': best_epoch, 'clean_macro_f1': clean.macro_f1, 'noisy_10db_macro_f1': noisy.macro_f1, 'labels': list(LABELS), 'clean_per_class_f1': clean.per_class_f1, 'noisy_per_class_f1': noisy.per_class_f1, 'split': str(SPLIT), 'noise_evaluation_partition_previously_used': True, 'test_split_loaded': False}
    (args.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__': main()
