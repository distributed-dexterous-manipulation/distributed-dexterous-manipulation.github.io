import torch
from utils.MATBC.gpt_adaln_core import Transformer

class MATBC:
    def __init__(self, parent_hp_dict):
        self.hp_dict = {
            "exp_name"          : parent_hp_dict['exp_name'],
            "data_dir"          : "./data/rl_data",
            "dont_log"          : parent_hp_dict['dont_log'],
            "replay_size"       : 500001,
            'warmup_epochs'     : 1000,
            'pi_lr'             : parent_hp_dict['pi_lr'],
            'q_lr'              : parent_hp_dict['q_lr'],
            "q_eta_min"         : parent_hp_dict['q_eta_min'],
            "pi_eta_min"        : parent_hp_dict['pi_eta_min'],
            "tau"               : 0.005,
            "gamma"             : 0.99,

            # DiT Params:
            'state_dim'         : 6,
            'obj_name_enc_dim'  : 9,
            'action_dim'        : 2,
            'act_limit'         : 0.03,
            "dev_rl"            : parent_hp_dict['dev_rl'],
            'optim'             : 'adam',
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : 512,
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : parent_hp_dict['max_grad_norm'],
            "alpha"             : 0.2,
            "attn_mech"         : parent_hp_dict['attn_mech'],
            'masked'            : parent_hp_dict['masked'],
            'gauss'             : parent_hp_dict['gauss'],
            'learned_alpha'     : parent_hp_dict['learned_alpha'],
            'sce'               : './utils/MATBC/sce.pth',
        }
        self.device = self.hp_dict['dev_rl']
        self.tf = Transformer(self.hp_dict)
        self.tf.to(self.device)
        self.gauss = self.hp_dict['gauss']

    @torch.no_grad()
    def get_actions(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).unsqueeze(0).to(self.device)
            
        actions = self.tf.get_actions(obs, pos, deterministic).squeeze()
        return actions.detach().to(torch.float32).cpu().numpy()
    
    @torch.no_grad()
    def get_actions_batch(self, obs, pos, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        pos = torch.as_tensor(pos, dtype=torch.int32).to(self.device)
            
        actions = self.tf.get_actions(obs, pos, deterministic)
        return actions.detach().cpu().numpy().mean(axis=0)

    def load_saved_policy(self, path):
        print("HAKUNA")
        expt_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.tf.load_state_dict(expt_dict['model'])
