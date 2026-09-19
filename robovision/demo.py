"""Render matched cup-grasping episodes with a camera inset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from robovision.control import action_to_target, execute_approach, run_neural_grasp
from robovision.env import VisionCupEnv
from robovision.io import atomic_json, file_hash
from robovision.models import load_grasp, load_target

ROOT = Path(__file__).resolve().parents[1]
FPS = 20
WIDTH = 640
HEADER = 124
HEIGHT = HEADER + 480 + 86
BACKGROUND = '#111820'
MUTED = '#aab8c5'
WHITE = '#f0f5f9'


def font(size):
    return ImageFont.load_default(size=size)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def episode(approach, network, seed):
    env = VisionCupEnv(max_steps=160, render_images=True)
    try:
        observation, _ = env.reset(seed=seed)
        initial_image = observation['image'].copy()
        initial_time = float(env.data.time)
        frames, phases, elapsed, cameras = [], [], [], []

        def capture(current_env, phase):
            frames.append(current_env.render())
            phases.append(phase)
            elapsed.append(float(current_env.data.time) - initial_time)
            cameras.append(current_env.render_camera())

        capture(env, 'visual target')
        with torch.no_grad():
            prediction, _ = approach.predict(observation, deterministic=True)
        target = action_to_target(prediction)
        execute_approach(env, target, frame_callback=capture)
        alignment_error = float(np.linalg.norm(env.grasp_position[:2] - env.cup_position[:2]))
        env.render_images = False
        info = run_neural_grasp(env, target, network, frame_callback=capture, record=True)
        record = {
            'seed': seed,
            'target_xy': target.tolist(),
            'approach_action': prediction.tolist(),
            'initial_alignment_error': alignment_error,
            'initial_camera_sha256': hashlib.sha256(initial_image.tobytes()).hexdigest(),
            'success': bool(info['is_success']),
            'elapsed_seconds': elapsed[-1],
            'final_info': json_value(info),
        }
        return dict(frames=frames, phases=phases, elapsed=elapsed, cameras=cameras, record=record)
    finally:
        env.close()


def render_panel(rollout, index, name, accent, camera_inset):
    last = len(rollout['frames']) - 1
    position = min(index, last)
    terminal = index >= last
    result = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, WIDTH - 1, 5), fill=accent)
    draw.text((22, 21), name, font=font(27), fill=WHITE)
    draw.text((22, 60), 'PPO target policy + BC grasp controller', font=font(19), fill=MUTED)
    draw.text((22, 89), 'Shared grasp network controls vertical motion and jaws', font=font(17), fill=MUTED)
    result.paste(Image.fromarray(rollout['frames'][position]), (0, HEADER))
    if camera_inset:
        inset = Image.fromarray(rollout['cameras'][position]).resize((120, 120), Image.Resampling.NEAREST)
        x, y = WIDTH - 138, HEADER + 16
        draw.rectangle((x - 3, y - 3, x + 122, y + 145), fill=BACKGROUND)
        result.paste(inset, (x, y))
        draw.text((x + 1, y + 125), 'Camera view', font=font(13), fill=WHITE)
    footer_y = HEADER + 480
    success = rollout['record']['success']
    status = ('SUCCESS' if success else 'UNSUCCESSFUL') if terminal else rollout['phases'][position].upper()
    color = ('#7addaa' if success else '#f2b6a0') if terminal else WHITE
    draw.text((22, footer_y + 16), status, font=font(23), fill=color)
    timing = f"{rollout['elapsed'][position]:.2f} simulated seconds"
    if index > last:
        timing += ' | final frame held'
    draw.text((22, footer_y + 49), timing, font=font(17), fill=MUTED)
    return result


def compose(left, right, index, scenario, number, total, training_seed, inset):
    top = 78
    image = Image.new('RGB', (2 * WIDTH, HEIGHT + top), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((22, 15), 'roboVision / Learned cup grasping', font=font(26), fill=WHITE)
    label = f'Scenario {scenario} | {number}/{total} fixed scenes | training seed {training_seed}'
    draw.text((22, 49), label, font=font(17), fill=MUTED)
    image.paste(render_panel(left, index, 'A  Hard-only target training', '#99a9b7', inset), (0, top))
    image.paste(render_panel(right, index, 'B  Curriculum target training', '#64b8ce', inset), (WIDTH, top))
    draw.line((WIDTH, top, WIDTH, HEIGHT + top), fill=BACKGROUND, width=4)
    return np.asarray(image)


def make_video(args):
    torch.set_num_threads(args.threads)
    outputs = [args.out, args.out.with_suffix('.png'), args.out.with_suffix('.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f'Output already exists: {args.out}. Choose a new --out path.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model_paths = {'hard': args.hard_model, 'curriculum': args.curriculum_model}
    models = {name: load_target(path) for name, path in model_paths.items()}
    network = load_grasp(args.grasp_model)
    evidence = {
        'scenario_seeds': args.seeds,
        'training_seed': args.training_seed,
        'fps': FPS,
        'playback': 'real time; terminal frames held for comparison',
        'target_model_sha256': {name: file_hash(path) for name, path in model_paths.items()},
        'grasp_model_sha256': file_hash(args.grasp_model),
        'episodes': [],
    }
    writer = imageio.get_writer(str(args.out), fps=FPS, codec='libx264', quality=8,
                               macro_block_size=2, ffmpeg_params=['-movflags', '+faststart'])
    try:
        for number, seed in enumerate(args.seeds, 1):
            left = episode(models['hard'], network, seed)
            right = episode(models['curriculum'], network, seed)
            if left['record']['initial_camera_sha256'] != right['record']['initial_camera_sha256']:
                raise RuntimeError('Paired episodes have different initial camera observations')
            count = max(len(left['frames']), len(right['frames']))
            for index in range(count + FPS):
                frame = compose(left, right, index, seed, number, len(args.seeds),
                                args.training_seed, not args.no_inset)
                writer.append_data(frame)
                if number == 1 and index == count - 1:
                    Image.fromarray(frame).save(args.out.with_suffix('.png'))
            evidence['episodes'].append({'seed': seed, 'hard': left['record'], 'curriculum': right['record']})
            atomic_json(args.out.with_suffix('.json'), evidence)
            print(json.dumps({'seed': seed, 'hard_success': left['record']['success'],
                              'curriculum_success': right['record']['success']}), flush=True)
    finally:
        writer.close()
    evidence['video_sha256'] = file_hash(args.out)
    evidence['preview_sha256'] = file_hash(args.out.with_suffix('.png'))
    atomic_json(args.out.with_suffix('.json'), evidence)
    print(f'Saved {args.out}')
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hard-model', type=Path, default=ROOT / 'models/target_seed101_hard.pt')
    parser.add_argument('--curriculum-model', type=Path, default=ROOT / 'models/target_seed101_curriculum.pt')
    parser.add_argument('--grasp-model', type=Path, default=ROOT / 'models/grasp.pt')
    parser.add_argument('--out', type=Path, default=Path('runs/demo.mp4'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[81000, 81001, 81002, 81040])
    parser.add_argument('--training-seed', type=int, default=101)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--no-inset', action='store_true')
    make_video(parser.parse_args(argv))


if __name__ == '__main__':
    main()
