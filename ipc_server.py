import numpy as np
import os
import time
import wandb
from threading import Lock

from utils.logger import MetricLogger
import utils.MATSAC.matsac as matsac
import utils.MATBC.matbc_test as matbc
import utils.MATBC.matbc_finetune as matbc_finetune

update_lock = Lock()

class DeltaArrayServer():
    def __init__(self, config):
        self.train_or_test = "test" if config['test'] else "train"
        os.makedirs(f'./data/rl_data/{config['name']}/pyt_save', exist_ok=True)
        ma_env_dict = {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 3},
                    'pi_obs_space'  : {'dim': 6},
                    'q_obs_space'   : {'dim': 6},
                    "max_agents"    : 64,}
        
        self.hp_dict = {
            "env_dict"          : {'action_space': {'low': -0.03, 'high': 0.03, 'dim': 2},
                                    'pi_obs_space'  : {'dim': 6},
                                    'q_obs_space'   : {'dim': 6},
                                    "max_agents"    : 64,},
            "exp_name"          : config['name'],
            "diff_exp_name"     : "expt_1",
            'algo'              : config['algo'],
            'data_type'         : config['data_type'],
            "dont_log"          : config['dont_log'],
            "rblen"             : config['rblen'],
            'seed'              : 69420,
            "data_dir"          : "./data/rl_data",
            "real"              : config['real'],
            "infer_every"       : config['infer_every'],
            
            # RL params
            "tau"               : config['tau'],
            "gamma"             : config['gamma'],
            "q_lr"              : config['q_lr'],
            "pi_lr"             : config['pi_lr'],
            "q_eta_min"         : config['q_etamin'],
            "pi_eta_min"        : config['pi_etamin'],
            "eta_min"           : config['q_etamin'],
            "alpha"             : config['alpha'],
            'optim'             : config['optim'],
            'epsilon'           : 1.0,
            "batch_size"        : config['bs'],
            "warmup_epochs"     : config['warmup'],
            "policy_delay"      : config['policy_delay'],
            'act_limit'         : 0.03,

            # Multi Agent Part Below:
            'state_dim'         : 6,
            'action_dim'        : 3,
            "dev_rl"            : config['dev_rl'],
            "model_dim"         : 256,
            "num_heads"         : 8,
            "dim_ff"            : config['dim_ff'],
            "n_layers_dict"     : {'encoder':5, 'actor': 10, 'critic': 10},
            "dropout"           : 0,
            "max_grad_norm"     : config['gradnorm'],
            "adaln"             : config['adaln'],
            "delta_array_size"  : [8,8],
            'masked'            : config['masked'],
            'gauss'             : config['gauss'],
            'learned_alpha'     : config['la'],
            'test_traj'         : config['test_traj'],
            'resume'            : config['resume'] != "No",
            'attn_mech'         : config['attn_mech'],
            'pos_embed'         : config['pos_embed'],
            'sce'               : './utils/MATBC/sce.pth',
            'dec'               : config['dec'],
            'cec'               : config['cec'],
        }
        # For pure MATBC, MATSAC, and MATBC_FT, we use number of actions as 2 and intentionally set a_z = z_low.
        hp_dict_old = self.hp_dict.copy()
        hp_dict_old['action_dim'] = 2
        self.logger = MetricLogger(dontlog=self.hp_dict["dont_log"])

        if self.hp_dict['test_traj']:
            self.pushing_agents = {
                "Vis Servo" : None,
                "MATSAC" : matsac.MATSAC(ma_env_dict, self.hp_dict, train_or_test="test"),
                "MATBC" : matbc.MATBC(self.hp_dict),
                "MATBC_FT" : matbc_finetune.MATBC_FT(hp_dict_old, self.logger),
                "MATBC_FT_OG" : matbc_finetune.MATBC_FT(self.hp_dict, self.logger),
                "MATBC_FT_DEC" : matbc_finetune.MATBC_FT(self.hp_dict, self.logger),
                "MATBC_FT_CEC" : matbc_finetune.MATBC_FT(self.hp_dict, self.logger),
                "MATBC_FT_MEC" : matbc_finetune.MATBC_FT(self.hp_dict, self.logger),
            }
            self.pushing_agents["MATSAC"].load_saved_policy("./pretrained_ckpts/matsac.pt")
            self.pushing_agents["MATBC"].load_saved_policy("./pretrained_ckpts/matbc.pt")
            self.pushing_agents["MATBC_FT"].load_saved_policy("./pretrained_ckpts/matbc_ft.pt")
            self.pushing_agents["MATBC_FT_OG"].load_saved_policy("./pretrained_ckpts/matbc_ft_og.pt")
            self.pushing_agents["MATBC_FT_DEC"].load_saved_policy("./pretrained_ckpts/matbc_ft_dec.pt")
            self.pushing_agents["MATBC_FT_CEC"].load_saved_policy("./pretrained_ckpts/matbc_ft_cec.pt")
            self.pushing_agents["MATBC_FT_MEC"].load_saved_policy("./pretrained_ckpts/matbc_ft_mec.pt")
        else:
            if config['algo']=="MATSAC":
                self.pushing_agent = matsac.MATSAC(ma_env_dict, self.hp_dict, train_or_test="train")
                if config['resume'] != "No":
                    self.pushing_agent.load_saved_policy(config['resume'])
            elif config['algo']=="MATBC":
                self.pushing_agent = matbc.MATBC()
            elif config['algo']=="MATBC_FT":
                self.pushing_agent = matbc_finetune.MATBC_FT(self.hp_dict, self.logger)
                if config['resume'] != "No":
                    self.pushing_agent.load_saved_policy(config['resume'])
                else:
                    self.pushing_agent.load_saved_policy(f'./pretrained_ckpts/{config['finetune_name']}.pt')

            if (self.train_or_test=="test") and (config['algo']=="MATSAC"):
                if config['t_path'] is not None:
                    self.pushing_agent.load_saved_policy(config['t_path'])
                else:
                    self.pushing_agent.load_saved_policy(f'./data/rl_data/{config['name']}/pyt_save/model.pt')
            elif (self.train_or_test=="test") and (config['algo'] in ["MATBC", "MATBC_FT"]):
                self.pushing_agent.load_saved_policy(f'./pretrained_ckpts/{config['name']}.pth')
        
            if (self.train_or_test=="train") and (not self.hp_dict["dont_log"]):
                if config['resume'] != "No":
                    wandb.init(project="Distributed_Dexterous_Manipulation",
                            config=self.hp_dict,
                            name = self.hp_dict['exp_name'],
                            id=config['wb_resume'],
                            resume=True)
                else:
                    wandb.init(project="Distributed_Dexterous_Manipulation",
                            config=self.hp_dict,
                            name = self.hp_dict['exp_name'])

def server_process_main(pipe_conn, batched_queue, response_dict, config):
    """
    Runs in the child process:
        1) Creates the DeltaArrayServer instance
        2) Loops, waiting for requests from the parent process
        3) Executes the request (get_actions, update, etc.) and replies

        MA_GET_ACTION           1   : data_args = [states, pos, deterministic]
        MA_UPDATE_POLICY        2   : data_args = [bs, curr_ep, n_upd, avg_rew]
        MARB_STORE              3   : data_args = [replay_data]
        MARB_SAVE               4   : data_args = None
        SAVE_MODEL              5   : data_args = None
        LOAD_MODEL              6   : data_args = None
        LOG_INFERENCE           7   : data_args = [inference rewards]
        TT_GET_ACTION           8   : data_args = [algo_name, states, pos, deterministic]
        SET_BATCH_SIZE          9   : data_args = [batch_size]
        
    """
        
    MA_GET_ACTION        = 1
    MA_UPDATE_POLICY     = 2
    MARB_STORE           = 3
    MARB_SAVE            = 4
    SAVE_MODEL           = 5
    LOAD_MODEL           = 6
    LOG_INFERENCE        = 7
    TT_GET_ACTION        = 8
    SET_BATCH_SIZE       = 9
    
    global_batch_size = 1
    
    server = DeltaArrayServer(config)

    collecting_batch = False
    batch_start_time = None
    max_batch_wait = 5
    while True:
        if pipe_conn.poll(0.00005):
            try:
                request = pipe_conn.recv()
            except EOFError:
                continue
            
            endpoint, data = request
            response = {}

            if (endpoint == -1) or (endpoint is None):
                pipe_conn.send({"status": True})
                break
            
            elif endpoint == SET_BATCH_SIZE:
                global_batch_size = data
                
            elif endpoint == TT_GET_ACTION:
                algo, data = data[0], data[1:]
                with update_lock:
                    action = server.pushing_agents[algo].get_actions(*data)
                response = action

            elif endpoint == MA_UPDATE_POLICY:
                server.pushing_agent.update(*data)

            elif endpoint == SAVE_MODEL:
                server.pushing_agent.save_model()

            elif endpoint == LOAD_MODEL:
                server.pushing_agent.load_model()

            elif endpoint == MARB_STORE:
                server.pushing_agent.ma_replay_buffer.store(data)

            elif endpoint == MARB_SAVE:
                server.pushing_agent.ma_replay_buffer.save_RB()

            elif endpoint == LOG_INFERENCE:
                if not server.hp_dict["dont_log"]:
                    for reward in data:
                        server.logger.add_data("Inference Reward", reward)

            else:
                print(f"[Child] Invalid endpoint: {endpoint}")

            pipe_conn.send(response)
        
        
        if (not batched_queue.empty()) and (global_batch_size > 1):
            if not collecting_batch:
                collecting_batch = True
                batched_states = []
                batched_poses = []
                req_ids = []
                batch_start_time = time.time()
            
            while len(batched_states) < global_batch_size:
                try:
                    if not batched_queue.empty():
                        req = batched_queue.get_nowait()
                        endpoint, data, req_id = req
                    if endpoint == MA_GET_ACTION:
                        batched_states.append(data[0])
                        batched_poses.append(data[1])
                        inference = data[2]
                        req_ids.append(req_id)
                    else:
                        break  # No more requests in queue
                except Exception as e:
                    print(f"Error in batch collection: {e}")
                    break
                    
                if (time.time() - batch_start_time) > max_batch_wait:
                    break
                
            if len(batched_states) > 0:
                bs = len(batched_states)
                max_agents = max([len(s) for s in batched_states])
                
                states = np.zeros((bs, max_agents, server.pushing_agent.obs_dim), dtype=np.float32)
                poses = np.zeros((bs, max_agents, 1), dtype=np.float32)
                
                # For loop because states and pos dims are inhomogeneous
                for i, (state_batch, pose_batch) in enumerate(zip(batched_states, batched_poses)):
                    states[i, :len(state_batch)] = state_batch
                    poses[i, :len(pose_batch)] = pose_batch[:, None]

                with update_lock:
                    outputs = server.pushing_agent.get_actions(states, poses, inference)
                    if outputs.shape[0] == 4:
                        a_kNone = False
                        actions, a_ks, log_ps, ents = outputs
                    else:
                        a_kNone = True
                        actions, a_ks, log_ps, ents = outputs, None, None, None
                    
                # Send each individual action back via its corresponding response Queue.
                for i, req_id in enumerate(req_ids):
                    if a_kNone:
                        response_dict[req_id] = (actions[i], None, None, None)
                    else:
                        response_dict[req_id] = (actions[i], a_ks[i], log_ps[i], ents[i])

                collecting_batch = False

    # Cleanup
    pipe_conn.close()
    print("[Child] Exiting child process safely.")
