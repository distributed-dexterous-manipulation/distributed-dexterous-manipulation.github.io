import numpy as np
from copy import deepcopy
import itertools
import wandb

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import utils.MATSAC.gpt_core_no_autoreg as core
import utils.multi_agent_replay_buffer as MARB

class MATSAC:
    def __init__(self, env_dict, hp_dict, train_or_test="train"):
        self.train_or_test = train_or_test
        self.hp_dict = hp_dict.copy()
        self.env_dict = env_dict
        self.obs_dim = self.env_dict['pi_obs_space']['dim']
        self.dec = self.hp_dict['dec']
        self.cec = self.hp_dict['cec']
        self.act_dim = 2 if not(self.dec or self.cec) else 3
        self.hp_dict['action_dim'] = self.act_dim
        self.act_limit = self.env_dict['action_space']['high']
        self.device = self.hp_dict['dev_rl']
        self.gauss = hp_dict['gauss']
        self.log_dict = {
            'Q loss': [],
            'Pi loss': [],
            'alpha': [],
            'mu': [],
            'std': [],
            'Reward': []
        }
        self.max_avg_rew = 0
        self.batch_size = hp_dict['batch_size']

        
        self.tf = core.Transformer(self.hp_dict)
        self.tf_target = deepcopy(self.tf)

        self.tf = self.tf.to(self.device)
        self.tf_target = self.tf_target.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.tf_target.parameters():
            p.requires_grad = False
        self.ma_replay_buffer = MARB.MultiAgentReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=hp_dict['rblen'], max_agents=self.tf.max_agents)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.tf.decoder_actor, self.tf.decoder_critic1, self.tf.decoder_critic2])
        if self.train_or_test == "train":
            print(f"\nNumber of parameters: {np.sum(var_counts)}\n")

        self.critic_params = itertools.chain(self.tf.decoder_critic1.parameters(), self.tf.decoder_critic2.parameters())
        
        if self.hp_dict['optim']=="adam":
            self.optimizer_actor = optim.Adam(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.Adam(filter(lambda p: p.requires_grad, self.critic_params), lr=hp_dict['q_lr'])
        elif self.hp_dict['optim']=="adamW":
            self.optimizer_actor = optim.AdamW(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.AdamW(filter(lambda p: p.requires_grad, self.critic_params), lr=hp_dict['q_lr'])
        elif self.hp_dict['optim']=="sgd":
            self.optimizer_actor = optim.SGD(filter(lambda p: p.requires_grad, self.tf.decoder_actor.parameters()), lr=hp_dict['pi_lr'])
            self.optimizer_critic = optim.SGD(filter(lambda p: p.requires_grad, self.critic_params), lr=hp_dict['q_lr'])

        self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=10, T_mult=2, eta_min=hp_dict['eta_min'])
        self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=10, T_mult=2, eta_min=hp_dict['eta_min'])
        
        self.q_loss = None
        self.internal_updates_counter = 0
        
        self.q_loss_scaler = torch.amp.GradScaler('cuda')
        self.pi_loss_scaler = torch.amp.GradScaler('cuda')

        if self.gauss:
            if self.hp_dict['learned_alpha']:
                self.log_alpha = torch.tensor([-1.6094], requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp_dict['pi_lr'])
            else:
                self.alpha = torch.tensor([self.hp_dict['alpha']], requires_grad=False, device=self.device)

    def compute_q_loss(self, s1, a, s2, r, d, pos):
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q1 = self.tf.decoder_critic1(s1, a, pos).squeeze().mean(dim=1)
            q2 = self.tf.decoder_critic2(s1, a, pos).squeeze().mean(dim=1)
            
            with torch.no_grad():
                # if self.gauss:
                #     next_actions, log_probs, _, _= self.tf(s2, pos)
                # else:z
                #     next_actions = self.tf(s2, pos)
                
                # next_q1 = self.tf_target.decoder_critic1(s2, next_actions, pos).squeeze()
                # next_q2 = self.tf_target.decoder_critic2(s2, next_actions, pos).squeeze()
                q_next = r # + self.hp_dict['gamma'] * ((1 - d.unsqueeze(1)) * (torch.min(next_q1, next_q2) - self.alpha * log_probs)).mean(dim=1)
                # q_next = r.unsqueeze(1)
            q_loss1 = F.mse_loss(q1, q_next)
            q_loss2 = F.mse_loss(q2, q_next)
            
        self.q_loss_scaler.scale(q_loss1).backward()
        self.q_loss_scaler.scale(q_loss2).backward()
        
        self.q_loss_scaler.unscale_(self.optimizer_critic)
        torch.nn.utils.clip_grad_norm_(self.critic_params, self.hp_dict['max_grad_norm'])
        
        self.q_loss_scaler.step(self.optimizer_critic)
        self.q_loss_scaler.update()
        
        self.log_dict['Q loss'].append(q_loss1.item() + q_loss2.item())

    def compute_pi_loss(self, s1, pos):
        for p in self.critic_params:
            p.requires_grad = False
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, n_agents, _ = s1.size()
            if self.gauss:
                actions, log_probs, mu, std = self.tf(s1, pos)
            else:
                actions = self.tf(s1, pos)
            # actions = self.tf.get_actions(s1, pos)
            
            q1_pi = self.tf.decoder_critic1(s1, actions, pos).squeeze()
            q2_pi = self.tf.decoder_critic2(s1, actions, pos).squeeze()
            q_pi = torch.minimum(q1_pi, q2_pi)
            
            if self.gauss:
                pi_loss = (self.alpha * log_probs - q_pi).mean()
            else:
                pi_loss = -q_pi.mean()
            
        self.pi_loss_scaler.scale(pi_loss).backward()
        self.pi_loss_scaler.unscale_(self.optimizer_actor)
        # pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tf.decoder_actor.parameters(), self.hp_dict['max_grad_norm'])
        self.pi_loss_scaler.step(self.optimizer_actor)
        self.pi_loss_scaler.update()
        
        # Update alpha
        if self.hp_dict['learned_alpha']:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_probs - self.act_dim*n_agents).detach()).mean() # Target entropy is -act_dim
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        for p in self.critic_params:
            p.requires_grad = True
            
        self.log_dict['Pi loss'].append(pi_loss.item())
        if self.gauss:
            self.log_dict['alpha'].append(self.alpha.item())
            self.log_dict['mu'].append(mu.mean().item())
            self.log_dict['std'].append(std.mean().item())

    def update(self, current_episode, n_updates, avg_reward):
        self.log_dict['Reward'].append(avg_reward)
        for j in range(n_updates):
            self.internal_updates_counter += 1
            # if self.internal_updates_counter == 1:
            #     for param_group in self.optimizer_critic.param_groups:
            #         param_group['lr'] = 1e-6
            #     for param_group in self.optimizer_actor.param_groups:
            #         param_group['lr'] = 1e-6
            # elif self.internal_updates_counter == self.hp_dict['warmup_epochs']:
            #     for param_group in self.optimizer_critic.param_groups:
            #         param_group['lr'] = self.hp_dict['q_lr']
            #     for param_group in self.optimizer_actor.param_groups:
            #         param_group['lr'] = self.hp_dict['pi_lr']
            self.scheduler_actor.step()
            self.scheduler_critic.step()

            data = self.ma_replay_buffer.sample_batch(self.batch_size)
            n_agents = int(torch.max(data['num_agents']))
            states = data['obs'][:,:n_agents].to(self.device)
            actions = data['act'][:,:n_agents].to(self.device)
            rews = data['rew'].to(self.device)
            new_states = data['obs2'][:,:n_agents].to(self.device)
            dones = data['done'].to(self.device)
            pos = data['pos'][:,:n_agents].to(self.device)

            # Critic Update
            # self.optimizer_critic.zero_grad()
            for param in self.critic_params:
                param.grad = None
            self.compute_q_loss(states, actions, new_states, rews, dones, pos)

            # Actor Update
            # with torch.autograd.set_detect_anomaly(True):
            # self.optimizer_actor.zero_grad()
            for param in self.tf.decoder_actor.parameters():
                param.grad = None
            self.compute_pi_loss(states, pos)

            # Target Update
            with torch.no_grad():
                for p, p_target in zip(self.tf.parameters(), self.tf_target.parameters()):
                    p_target.data.mul_(self.hp_dict['tau'])
                    p_target.data.add_((1 - self.hp_dict['tau']) * p.data)

            if (self.train_or_test == "train") and (self.internal_updates_counter % 5000) == 0:
                if self.max_avg_rew < avg_reward:
                    print("ckpt saved @ ", current_episode, self.internal_updates_counter)
                    self.max_avg_rew = avg_reward
                    dicc = {
                        'model': self.tf.state_dict(),
                        'actor_optimizer': self.optimizer_actor.state_dict(),
                        'critic_optimizer': self.optimizer_critic.state_dict(),
                    }
                    torch.save(dicc, f"{self.hp_dict['data_dir']}/{self.hp_dict['exp_name']}/pyt_save/model.pt")
        
            if (not self.hp_dict["dont_log"]) and (self.internal_updates_counter % 100) == 0:
                wandb.log({k: np.mean(v) if isinstance(v, list) and len(v) > 0 else v for k, v in self.log_dict.items()})
                self.log_dict = {
                    'Q loss': [],
                    'Pi loss': [],
                    'alpha': [],
                    'mu': [],
                    'std': [],
                    'Reward': []
                }
                
    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).to(self.device)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
            pos = pos.unsqueeze(0)
            
        actions = self.tf.get_actions(obs, pos, deterministic=deterministic)
        return actions.detach().to(torch.float32).cpu().numpy()
        
    def load_saved_policy(self, path='./data/rl_data/backup/matsac_expt_grasp/pyt_save/model.pt'):
        dicc = torch.load(path, map_location=self.hp_dict['dev_rl'])
        
        self.tf.load_state_dict(dicc['model'])
        self.optimizer_actor.load_state_dict(dicc['actor_optimizer'])
        self.optimizer_critic.load_state_dict(dicc['critic_optimizer'])
        self.tf_target = deepcopy(self.tf)