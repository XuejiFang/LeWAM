"""Evaluate one LeWAM task/phase on the fixed RoboTwin benchmark cases."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import hydra
from omegaconf import DictConfig
import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def task_arguments(robotwin, task_name, phase):
    args = read_yaml(robotwin / 'env_cfg/task_config' / f'{phase}.yml')
    if args['embodiment'] != ['aloha-agilex']:
        raise ValueError('LeWAM checkpoints use the aloha-agilex embodiment.')
    embodiment = read_yaml(robotwin / 'env_cfg/task_config/_embodiment_config.yml')['aloha-agilex']
    robot = embodiment['file_path']
    camera = read_yaml(robotwin / 'env_cfg/task_config/_camera_config.yml')[args['camera']['head_camera_type']]
    args.update(task_name=task_name, task_config=phase, eval_mode=True,
                left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True,
                left_embodiment_config=read_yaml(Path(robot) / 'config.yml'),
                right_embodiment_config=read_yaml(Path(robot) / 'config.yml'),
                head_camera_h=camera['h'], head_camera_w=camera['w'],
                render_freq=0, save_data=False)
    return args


def save_results(path, metadata, results):
    payload = {**metadata, 'evaluated_num': len(results),
               'success_num': sum(row['success'] for row in results), 'results': results}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def start_video(task, path, args):
    process = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
         '-video_size', f"{args['head_camera_w']}x{args['head_camera_h']}",
         '-framerate', '10', '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
         '-crf', '23', str(path)], stdin=subprocess.PIPE)
    task._set_eval_video_ffmpeg(process)
    return process


@hydra.main(version_base='1.3', config_path='../../configs/eval', config_name='sim_robotwin')
def main(cfg: DictConfig):
    options = cfg.EVALUATION
    task_name, phase = options.task_name, options.task_config
    if not isinstance(task_name, str) or phase not in ('demo_clean', 'demo_randomized'):
        raise ValueError('This worker needs one task and one phase; use the manager for multiple jobs.')
    shard, shards = int(options.shard_id), int(options.num_shards)
    if not 0 <= shard < shards:
        raise ValueError('Expected 0 <= shard_id < num_shards.')
    count = int(options.eval_num_episodes)
    if count < 1:
        raise ValueError('eval_num_episodes must be positive.')
    checkpoint = resolve_path(cfg.ckpt)
    robotwin = resolve_path(options.robotwin_root)
    output = resolve_path(options.output_root) / task_name / phase
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f'shard_{shard}.json'
    cases = None
    case_hash = None
    if options.cases_dir is not None:
        case_path = resolve_path(options.cases_dir) / f'{task_name}__{phase}.json'
        case_bytes = case_path.read_bytes()
        cases = json.loads(case_bytes)
        if count > len(cases):
            raise ValueError(f'{case_path} contains only {len(cases)} cases.')
        case_hash = hashlib.sha256(case_bytes).hexdigest()
    elif shards != 1:
        raise ValueError('Generated cases use one worker per task/phase; use the manager.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    os.environ.setdefault('VK_ICD_FILENAMES', str(ROOT / 'environments/robotwin/nvidia_egl_icd.json'))
    os.chdir(robotwin)
    sys.path[:0] = [str(ROOT), str(robotwin), str(robotwin / 'description/utils')]

    import numpy as np
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions
    from experiments.robotwin.lewam_policy.deploy_policy import get_model

    stats = resolve_path(options.dataset_stats_path) if options.dataset_stats_path else checkpoint / 'stats/dataset_stats.json'
    encoder = checkpoint / 'vision_encoder'
    model = get_model({'ckpt_setting': str(checkpoint), 'dataset_stats_path': str(stats),
                       'encoder_model_name_or_path': str(encoder), 'device': 'cuda',
                       'task_name': task_name,
                       'task_name_to_id_path': str(ROOT / 'configs/eval/task_name_to_id.json'),
                       'replan_steps': options.replan_steps, 'diffusion_shift': options.diffusion_shift,
                       'generator_seed': options.generator_seed, 'random_sparse_offsets': False})
    args = task_arguments(robotwin, task_name, phase)
    save_video = bool(options.get('save_video', True))
    if save_video:
        videos = output / f'videos_{shard}'
        videos.mkdir(exist_ok=True)
        args['eval_video_save_dir'] = str(videos)
    task = getattr(importlib.import_module(f'envs.{task_name}'), task_name)()
    task.suc = task.test_num = 0
    metadata = {'task_name': task_name, 'task_config': phase, 'checkpoint': str(checkpoint),
                'case_sha256': case_hash, 'seed': int(options.seed),
                'case_source': 'fixed' if cases is not None else 'expert', 'test_num': count,
                'num_shards': shards, 'shard_id': shard,
                'generator_seed': options.generator_seed, 'replan_steps': options.replan_steps,
                'diffusion_shift': options.diffusion_shift}
    results = []
    generated_cases = []
    next_seed = 100000 * (int(options.seed) + 1)
    for position, index in enumerate(range(shard, count, shards)):
        if cases is None:
            for attempt in range(1000):
                seed = next_seed
                next_seed += 1
                try:
                    task.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
                    episode_info = task.play_once()
                    valid = task.plan_success and task.check_success()
                except UnStableError:
                    valid = False
                except Exception as error:
                    # Match RoboTwin: failed expert plans do not define evaluation cases.
                    print(f'Expert rejected seed={seed}: {type(error).__name__}: {error}', flush=True)
                    valid = False
                finally:
                    task.close_env()
                if valid:
                    case = {'seed': seed, 'info': episode_info['info']}
                    generated_cases.append(case)
                    (output / 'cases.json').write_text(json.dumps(generated_cases, indent=2) + '\n')
                    break
            else:
                raise RuntimeError(f'No expert-valid scene found in 1000 attempts for episode {index}.')
        else:
            case = cases[index]
        seed = int(case['seed'])
        try:
            task.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
        except UnStableError:
            task.close_env()
            results.append({'global_idx': index, 'seed': seed, 'success': False, 'status': 'unstable'})
            save_results(result_path, metadata, results)
            continue
        process = None
        try:
            descriptions = generate_episode_descriptions(task_name, [case['info']], count)
            instruction = str(np.random.default_rng(seed).choice(descriptions[0]['unseen']))
            task.set_instruction(instruction=instruction)
            if save_video:
                video_path = videos / f'episode_{index:03d}.tmp.mp4'
                process = start_video(task, video_path, args)
            model.reset()
            while task.take_action_cnt < task.step_lim:
                observation = task.get_obs() if model.should_request_observation() else None
                model.step(task, observation)
                if task.eval_success:
                    break
            success = bool(task.eval_success)
            steps = int(task.take_action_cnt)
        finally:
            if process is not None:
                task._del_eval_video_ffmpeg()
            task.close_env(clear_cache=(position + 1) % args['clear_cache_freq'] == 0)
        if process is not None:
            if process.returncode != 0:
                raise RuntimeError(f'ffmpeg failed with exit code {process.returncode}')
            video_path.rename(videos / f'episode_{index:03d}_success-{str(success).lower()}.mp4')
        results.append({'global_idx': index, 'seed': seed, 'success': success, 'status': 'evaluated',
                        'steps': steps, 'instruction': instruction})
        save_results(result_path, metadata, results)
        successes = sum(row['success'] for row in results)
        print(f'{task_name} {phase} shard={shard}/{shards} episode={index} seed={seed} '
              f'success={success} total={successes}/{len(results)}', flush=True)


if __name__ == '__main__':
    main()
