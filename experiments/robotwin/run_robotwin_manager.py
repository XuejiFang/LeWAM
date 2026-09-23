"""Schedule RoboTwin task/phase shards and aggregate complete evaluation results."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'experiments/robotwin/eval_robotwin_single.py'
PHASE_NAMES = {'demo_clean': 'clean', 'demo_randomized': 'random'}


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def as_list(value):
    return [value] if isinstance(value, str) else list(value)


@hydra.main(version_base='1.3', config_path='../../configs/eval', config_name='sim_robotwin')
def main(cfg: DictConfig):
    robotwin = resolve_path(cfg.EVALUATION.robotwin_root)
    known = yaml.safe_load((robotwin / 'env_cfg/task_config/_eval_step_limit.yml').read_text())
    tasks = list(known) if cfg.EVALUATION.task_name is None else as_list(cfg.EVALUATION.task_name)
    phases = as_list(cfg.EVALUATION.task_config)
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(known):
        raise ValueError('Select unique task names from the RoboTwin benchmark.')
    if not phases or len(set(phases)) != len(phases) or set(phases) - set(PHASE_NAMES):
        raise ValueError('Select demo_clean and/or demo_randomized.')
    gpus = list(cfg.MULTIRUN.gpu_ids)
    slots = int(cfg.MULTIRUN.workers_per_gpu)
    count = int(cfg.EVALUATION.eval_num_episodes)
    shards = min(int(cfg.MULTIRUN.shards_per_task), count)
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0 or min(slots, shards) < 1:
        raise ValueError('Use unique GPU IDs and positive worker/shard counts.')
    if count < 1:
        raise ValueError('eval_num_episodes must be positive.')
    if cfg.EVALUATION.cases_dir is None:
        shards = 1  # Preserve the sequential expert-selected seed stream.
    output = resolve_path(cfg.EVALUATION.output_root)
    configs, logs = output / 'configs', output / 'logs'
    configs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    cfg.ckpt = str(resolve_path(cfg.ckpt))
    cfg.EVALUATION.output_root = str(output)
    shared = OmegaConf.to_container(cfg, resolve=True)
    jobs = queue.Queue()
    for task in tasks:
        for phase in phases:
            for shard in range(shards): jobs.put((task, phase, shard))
    total = jobs.qsize()
    stopped = threading.Event()
    lock = threading.Lock()
    running = set()
    completed = 0

    def consume(gpu):
        nonlocal completed
        while not stopped.is_set():
            try:
                task, phase, shard = jobs.get_nowait()
            except queue.Empty:
                return
            name = f'{task}__{phase}__{shard}'
            job = OmegaConf.create(shared)
            job.gpu_id = gpu
            job.EVALUATION.task_name = task
            job.EVALUATION.task_config = phase
            job.EVALUATION.num_shards = shards
            job.EVALUATION.shard_id = shard
            OmegaConf.save(job, configs / f'{name}.yaml')
            command = [sys.executable, '-u', str(WORKER), '--config-path', str(configs),
                       '--config-name', name, 'hydra.run.dir=.', 'hydra.output_subdir=null',
                       'hydra.job.chdir=false', 'hydra/job_logging=disabled']
            log_path = logs / f'{name}.log'
            with log_path.open('w') as log:
                with lock:
                    if stopped.is_set():
                        return
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    running.add(process)
                try:
                    code = process.wait()
                finally:
                    with lock: running.discard(process)
            if code:
                stopped.set()
                raise RuntimeError(f'{name} failed with exit code {code}: {log_path}')
            with lock:
                completed += 1
                print(f'[{completed}/{total}] GPU {gpu}: {name}', flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus) * slots) as pool:
        futures = [pool.submit(consume, gpu) for gpu in gpus for _ in range(slots)]
        try:
            for future in as_completed(futures): future.result()
        except BaseException:
            stopped.set()
            with lock:
                for process in running:
                    if process.poll() is None:
                        try: os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError: pass
            raise

    task_results = []
    overall = {PHASE_NAMES[p]: {'successes': 0, 'episodes': 0} for p in phases}
    for task in tasks:
        for phase in phases:
            rows = []
            for shard in range(shards):
                result = json.loads((output / task / phase / f'shard_{shard}.json').read_text())
                assert result['checkpoint'] == cfg.ckpt and result['test_num'] == count
                assert result['num_shards'] == shards and result['shard_id'] == shard
                rows.extend(result['results'])
            assert len(rows) == count and {r['global_idx'] for r in rows} == set(range(count))
            case_path = (resolve_path(cfg.EVALUATION.cases_dir) / f'{task}__{phase}.json'
                         if cfg.EVALUATION.cases_dir is not None
                         else output / task / phase / 'cases.json')
            cases = json.loads(case_path.read_text())
            assert all(r['seed'] == cases[r['global_idx']]['seed'] for r in rows)
            successes = sum(r['success'] for r in rows)
            phase_name = PHASE_NAMES[phase]
            task_results.append({'task': task, 'phase': phase_name, 'successes': successes,
                                 'episodes': count, 'success_rate': successes / count})
            overall[phase_name]['successes'] += successes
            overall[phase_name]['episodes'] += count
    for stats in overall.values():
        stats['success_rate'] = stats['successes'] / stats['episodes']
    summary = {'checkpoint': cfg.ckpt, 'tasks': task_results, 'overall': overall}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['task', 'phase', 'successes', 'episodes', 'success_rate'])
        writer.writeheader()
        writer.writerows(task_results)
    for phase, stats in overall.items():
        print(f"{phase}: {stats['successes']}/{stats['episodes']} = {stats['success_rate']:.2%}")
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
