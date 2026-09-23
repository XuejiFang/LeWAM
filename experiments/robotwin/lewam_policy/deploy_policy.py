"""I-JEPA-Huge LeWAM policy adapter for RoboTwin's three RGB cameras."""
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from lewam import LeWAMPipeline
from lewam.models.vision_encoder import load_vision_encoder
from experiments.robotwin.image_utils import load_action_stats, compose_robotwin_triplet


class LeWAMRobotWinPolicy:
    def __init__(self, model, stats_path, task_id, replan_steps=24,
                 diffusion_shift=1.0, generator_seed=42):
        self.model = model
        parameter = next(model.predictor.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.task_id = torch.tensor([task_id], device=self.device, dtype=torch.long)
        self.replan_steps = int(replan_steps)
        if not 1 <= self.replan_steps <= model.action_horizon:
            raise ValueError(f'replan_steps must be within [1, {model.action_horizon}].')
        self.diffusion_shift = float(diffusion_shift)
        self.generator_seed = int(generator_seed)
        action_mean, action_std = load_action_stats(Path(stats_path))
        self.action_mean = torch.asarray(action_mean, dtype=torch.float32)
        self.action_std = torch.asarray(np.clip(action_std, 1e-6, None), dtype=torch.float32)
        self.pending_actions = deque()
        self.reset()

    def reset(self):
        # Restart the same action-noise stream per episode, independent of worker order.
        self.pending_actions.clear()
        self.generator = torch.Generator(device=self.device).manual_seed(self.generator_seed)

    def should_request_observation(self):
        return not self.pending_actions

    def _encode_image(self, observation):
        cameras = observation['observation']
        images = [torch.as_tensor(cameras[name]['rgb']).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                  for name in ('head_camera', 'left_camera', 'right_camera')]
        return compose_robotwin_triplet(*images).squeeze(0)

    @torch.inference_mode()
    def step(self, task_env, observation):
        if not self.pending_actions:
            if observation is None:
                raise ValueError('A new observation is required when replanning.')
            image = self._encode_image(observation).unsqueeze(0).to(device=self.device, dtype=self.dtype)
            actions = self.model(image=image, task_id=self.task_id, shift=self.diffusion_shift,
                                 random_sparse_offsets=False, generator=self.generator).action
            if actions.shape[1] != self.model.action_horizon:
                raise RuntimeError('Policy returned an incorrect action horizon.')
            self.pending_actions.extend(actions[0, :self.replan_steps].detach())
        action = self.pending_actions.popleft().cpu().float() * self.action_std + self.action_mean
        task_env.take_action(action.numpy(), action_type='qpos')


def get_model(args):
    encoder, _ = load_vision_encoder(args['encoder_model_name_or_path'], local_files_only=True)
    model = LeWAMPipeline.from_pretrained(
        args['ckpt_setting'], vision_encoder=encoder.to(torch.bfloat16), torch_dtype=torch.bfloat16
    ).to('cuda').eval()
    model.vision_encoder.requires_grad_(False)
    model.predictor.requires_grad_(False)
    task_ids = json.loads(Path(args['task_name_to_id_path']).read_text())
    return LeWAMRobotWinPolicy(model, args['dataset_stats_path'], task_ids[args['task_name']],
                              replan_steps=args.get('replan_steps', 24),
                              diffusion_shift=args.get('diffusion_shift', 1.0),
                              generator_seed=args.get('generator_seed', 42))


def eval(task_env, model, observation):
    model.step(task_env, observation)


def reset_model(model):
    model.reset()
