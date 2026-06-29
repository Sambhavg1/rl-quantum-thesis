#!/usr/bin/env python3
"""
PPO for Linear Cluster State Preparation — Heterogeneous Error Model
Includes unique sampled error rates per gate and tuned stability hyperparameters.
"""

import argparse, os, csv, json, time, datetime, random, sys, psutil
from collections import Counter
from functools import reduce

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

import gymnasium as gym
from gymnasium import spaces


# ============================================================================
# Auto-scaling hyperparameters per n
# ============================================================================
def default_hparams(n):
    if n <= 3:
        return {'steps_per_episode': 50, 'update_timestep': 4096, 'actor_hidden': [512, 256], 'critic_hidden': [512, 256], 'batch_size': 256, 'policy_epochs': 10, 'actor_lr': 1e-4, 'critic_lr': 5e-4, 'entropy_coef': 0.015, 'gamma': 0.99}
    elif n <= 6:
        # Tuned hyperparameters for stability
        return {'steps_per_episode': 180, 'update_timestep': 16384, 'actor_hidden': [1536, 768], 'critic_hidden': [1536, 768], 'batch_size': 512, 'policy_epochs': 20, 'actor_lr': 1e-4, 'critic_lr': 4e-4, 'entropy_coef': 0.005, 'gamma': 0.995}
    else:  
        return {'steps_per_episode': 300, 'update_timestep': 20480, 'actor_hidden': [2048, 1024, 512], 'critic_hidden': [2048, 1024, 512], 'batch_size': 512, 'policy_epochs': 20, 'actor_lr': 1e-4, 'critic_lr': 4e-4, 'entropy_coef': 0.015, 'gamma': 0.995}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True, help='Number of qubits')
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noisy', action='store_true', help='Enable Pauli noise')
    # Default noise_std updated to 0.15
    p.add_argument('--noise_std', type=float, default=0.15, help='Std of Pauli rotation angle')

    p.add_argument('--steps_per_episode', type=int, default=None)
    p.add_argument('--update_timestep', type=int, default=None)
    p.add_argument('--actor_hidden', type=int, nargs='+', default=None)
    p.add_argument('--critic_hidden', type=int, nargs='+', default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--policy_epochs', type=int, default=None)
    p.add_argument('--actor_lr', type=float, default=None)
    p.add_argument('--critic_lr', type=float, default=None)
    p.add_argument('--entropy_coef', type=float, default=None)
    p.add_argument('--gamma', type=float, default=None)
    # Tighter clipping to prevent policy collapse
    p.add_argument('--clip_ratio', type=float, default=0.1) 

    p.add_argument('--output_root', type=str, default='./results')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--allow_cpu', action='store_true')
    return p.parse_args()

def resolve_hparams(args):
    hp = default_hparams(args.n)
    for key in ['steps_per_episode', 'update_timestep', 'actor_hidden', 'critic_hidden', 'batch_size', 'policy_epochs', 'actor_lr', 'critic_lr', 'entropy_coef', 'gamma']:
        v = getattr(args, key, None)
        if v is not None:
            hp[key] = v
    return hp


# ============================================================================
# Environment
# ============================================================================
class QuantumEnv(gym.Env):
    def __init__(self, num_qubits=6, noisy=False, noise_std=0.15):
        super().__init__()
        self.num_qubits = num_qubits
        self.is_noisy = noisy
        self.noise_std = noise_std
        self.hilbert_dim = 2 ** self.num_qubits

        self._I2 = np.eye(2, dtype=np.complex128)
        self._X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._SX = np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128)
        self._Rx_pi_2 = np.array([[np.cos(np.pi/4), -1j*np.sin(np.pi/4)],
                                  [-1j*np.sin(np.pi/4), np.cos(np.pi/4)]], dtype=np.complex128)
        self._Rz_pi_2 = np.array([[np.exp(-1j*np.pi/4), 0],
                                  [0, np.exp(1j*np.pi/4)]], dtype=np.complex128)
        self._CZ4 = np.diag([1, 1, 1, -1]).astype(np.complex128).reshape(2, 2, 2, 2)
        
        self._pauli = {
            'X': np.array([[0,1],[1,0]], dtype=np.complex128),
            'Y': np.array([[0,-1j],[1j,0]], dtype=np.complex128),
            'Z': np.array([[1,0],[0,-1]], dtype=np.complex128),
        }

        # -------------------------------------------------------------
        # GENERATE HETEROGENEOUS ERROR RATES
        # -------------------------------------------------------------
        self.gate_error_rates = {}
        
        # Mean 1.5%, Std 0.3% for Single Qubit Gates (Clips negative to 0)
        sq_mean, sq_std = 0.015, 0.003  
        # Mean 5.0%, Std 1.0% for Two Qubit Gates
        cz_mean, cz_std = 0.050, 0.010  

        for op_name in ['X', 'SX', 'Rx(pi/2)']:
            for i in range(self.num_qubits):
                err = float(np.clip(np.random.normal(sq_mean, sq_std), 0.0, 1.0))
                self.gate_error_rates[f'{op_name}_{i}'] = err
                
        for i in range(self.num_qubits):
            self.gate_error_rates[f'Rz(pi/2)_{i}'] = 0.0 # Virtual gates

        for i in range(self.num_qubits - 1):
            err = float(np.clip(np.random.normal(cz_mean, cz_std), 0.0, 1.0))
            self.gate_error_rates[f'CZ_{i}_{i+1}'] = err

        # Build actions list
        self.actions_list = []  
        self.gate_names = []    
        single_ops = {'X': self._X, 'SX': self._SX, 'Rx(pi/2)': self._Rx_pi_2, 'Rz(pi/2)': self._Rz_pi_2}
        
        for op_name, op_mat in single_ops.items():
            for i in range(self.num_qubits):
                name = f'{op_name}_{i}'
                self.actions_list.append({
                    'type': 'single', 'op': op_mat, 'targets': (i,),
                    'name': name, 'error_rate': self.gate_error_rates[name],
                })
                self.gate_names.append(name)
                
        for i in range(self.num_qubits - 1):
            name = f'CZ_{i}_{i+1}'
            self.actions_list.append({
                'type': 'cz', 'op': self._CZ4, 'targets': (i, i+1),
                'name': name, 'error_rate': self.gate_error_rates[name],
            })
            self.gate_names.append(name)

        self.actions = {i: name for i, name in enumerate(self.gate_names)}
        self.action_space = spaces.Discrete(len(self.actions_list))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * self.hilbert_dim,), dtype=np.float64)

        self.target_state = self._build_target_state_via_full_matrices()
        self.state = None         
        self.last_fidelity = 0
        self.reset()

    def _build_target_state_via_full_matrices(self):
        h_single = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        target = np.zeros(self.hilbert_dim, dtype=np.complex128)
        target[0] = 1.0
        target = target.reshape((2,) * self.num_qubits)
        for q in range(self.num_qubits):
            target = np.tensordot(h_single, target, axes=([1], [q]))
            target = np.moveaxis(target, 0, q)
        for q in range(self.num_qubits - 1):
            target = np.tensordot(self._CZ4, target, axes=([2, 3], [q, q+1]))
            target = np.moveaxis(target, [0, 1], [q, q+1])
        return target.ravel().astype(np.complex128)

    def _apply_single_gate(self, op2, qubit):
        self.state = np.tensordot(op2, self.state, axes=([1], [qubit]))
        self.state = np.moveaxis(self.state, 0, qubit)

    def _apply_two_gate(self, op4, q1, q2):
        self.state = np.tensordot(op4, self.state, axes=([2, 3], [q1, q2]))
        self.state = np.moveaxis(self.state, [0, 1], [q1, q2])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((2,) * self.num_qubits, dtype=np.complex128)
        self.state[(0,) * self.num_qubits] = 1.0 + 0j
        self.last_fidelity = float(np.abs(np.vdot(self.target_state, self.state.ravel())) ** 2)
        return self._get_obs(), {}

    def step(self, action):
        info = self.actions_list[action]
        if info['type'] == 'single':
            self._apply_single_gate(info['op'], info['targets'][0])
        else:
            self._apply_two_gate(info['op'], *info['targets'])

        # Apply specific heterogeneous gate error
        if self.is_noisy and info['error_rate'] > 0.0 and random.random() < info['error_rate']:
            qt = random.choice(info['targets'])
            axis = random.choice(['X', 'Y', 'Z'])
            angle = np.random.normal(0, self.noise_std)
            err = (np.cos(angle/2) * self._I2 - 1j * np.sin(angle/2) * self._pauli[axis])
            self._apply_single_gate(err, qt)

        flat = self.state.ravel()
        norm_sq = np.vdot(flat, flat).real
        if norm_sq > 1e-18:
            self.state /= np.sqrt(norm_sq)

        fidelity = float(np.abs(np.vdot(self.target_state, self.state.ravel())) ** 2)
        reward = (fidelity - self.last_fidelity) * 10
        self.last_fidelity = fidelity
        reward -= 0.01
        done = fidelity >= 0.98
        if done:
            reward += 20.0
        return self._get_obs(), reward, done, False, {"fidelity": fidelity}

    def _get_obs(self):
        flat = self.state.ravel()
        return np.concatenate([flat.real, flat.imag])


# ============================================================================
# PPO Agent
# ============================================================================
class PPOAgent:
    def __init__(self, n_actions, n_features, actor_lr, critic_lr,
                 gamma, clip_ratio, policy_epochs, batch_size, entropy_coef,
                 actor_hidden, critic_hidden):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        self.actor = self._build_net(actor_hidden, self.n_actions, output_activation='softmax')
        self.critic = self._build_net(critic_hidden, 1, output_activation=None)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()
        self.clear_memory()

    def _build_net(self, hidden, n_out, output_activation):
        inp = layers.Input(shape=(self.n_features,))
        x = inp
        for h in hidden:
            x = layers.Dense(h, activation='relu')(x)
        out = layers.Dense(n_out, activation=output_activation)(x)
        return tf.keras.Model(inp, out)

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []

    def store_transition(self, s, a, r, ns, d, lp):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.next_states.append(ns); self.dones.append(d); self.log_probs.append(lp)

    def choose_action(self, state):
        t = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(t)
        action = tf.random.categorical(tf.math.log(probs + 1e-10), 1).numpy().item()
        log_prob = tf.math.log(probs[0, action] + 1e-10)
        return action, log_prob

    def choose_best_action(self, state):
        t = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(t)
        return int(tf.argmax(probs[0]).numpy())

    @tf.function
    def train_step(self, states, actions, advantages, old_log_probs):
        with tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
            values = tf.squeeze(self.critic(states, training=True))
            critic_loss = self.critic_loss_fn(advantages + values, values)

            new_probs = self.actor(states, training=True)
            action_indices = tf.stack(
                [tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            new_log_probs = tf.math.log(tf.gather_nd(new_probs, action_indices) + 1e-10)

            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            s1 = ratio * advantages
            s2 = clipped * advantages

            entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
            entropy_mean = tf.reduce_mean(entropy)

            actor_loss = -tf.reduce_mean(tf.minimum(s1, s2)) - (self.entropy_coef * entropy_mean)

        a_grads = a_tape.gradient(actor_loss, self.actor.trainable_variables)
        c_grads = c_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(a_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))

    def learn(self):
        if not self.states:
            return
        S = np.array(self.states, dtype=np.float32)
        A = np.array(self.actions, dtype=np.int32)
        R = np.array(self.rewards, dtype=np.float32)
        NS = np.array(self.next_states, dtype=np.float32)
        D = np.array(self.dones, dtype=np.float32)
        OL = np.array(self.log_probs, dtype=np.float32)

        values = self.critic(S).numpy().flatten()
        next_values = self.critic(NS).numpy().flatten()
        deltas = R + self.gamma * next_values * (1 - D) - values
        advantages = np.zeros_like(R)
        last_adv = 0.0
        for t in reversed(range(len(R))):
            advantages[t] = deltas[t] + self.gamma * 0.95 * last_adv * (1 - D[t])
            last_adv = advantages[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(S)
        idx = np.arange(n)
        for _ in range(self.policy_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                bi = idx[start:start + self.batch_size]
                self.train_step(
                    tf.convert_to_tensor(S[bi]),
                    tf.convert_to_tensor(A[bi]),
                    tf.convert_to_tensor(advantages[bi]),
                    tf.convert_to_tensor(OL[bi]),
                )
        self.clear_memory()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    hp = resolve_hparams(args)

    # Seeds must be set before generating the random error map!
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError: pass
        print(f"[GPU] {len(gpus)} device(s) visible")
    else:
        if not args.allow_cpu:
            print("[FATAL] No GPU. Use --allow_cpu to override.")
            sys.exit(2)
        print("[GPU] CPU only")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_tag = 'noisy' if args.noisy else 'ideal'
    run_name = args.run_name or f"cluster{args.n}_{noise_tag}_s{args.seed}_{ts}"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    config = {
        'n': args.n, 'episodes': args.episodes, 'seed': args.seed,
        'noisy': args.noisy, 'noise_std': args.noise_std,
        'clip_ratio': args.clip_ratio, **hp
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({k: (v if not isinstance(v, list) else list(v)) for k, v in config.items()}, f, indent=2)

    # Env + agent
    env = QuantumEnv(num_qubits=args.n, noisy=args.noisy, noise_std=args.noise_std)
    print(f"[ENV] n={args.n} | hilbert_dim={env.hilbert_dim} | actions={env.action_space.n}")

    # Export the randomly generated gate error map for post-analysis
    with open(os.path.join(out_dir, 'gate_error_map.json'), 'w') as f:
        json.dump(env.gate_error_rates, f, indent=2)
    print(f"[WROTE] gate_error_map.json to {out_dir}")

    agent = PPOAgent(
        n_actions=env.action_space.n, n_features=env.observation_space.shape[0],
        actor_lr=hp['actor_lr'], critic_lr=hp['critic_lr'], gamma=hp['gamma'], 
        clip_ratio=args.clip_ratio, policy_epochs=hp['policy_epochs'], 
        batch_size=hp['batch_size'], entropy_coef=hp['entropy_coef'],
        actor_hidden=hp['actor_hidden'], critic_hidden=hp['critic_hidden'],
    )

    all_fidelities = []
    all_action_sequences = []
    timestep_counter = 0

    csv_path = os.path.join(out_dir, 'cluster_results.csv')
    f_csv = open(csv_path, 'w', newline='', buffering=1)
    writer = csv.writer(f_csv)
    writer.writerow(['episode', 'final_fidelity', 'length', 'total_reward', 'action_sequence'])

    start_time = time.time()
    print(f"--- Starting Heterogeneous PPO Cluster({args.n} qubits) ---")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_actions = []
        episode_reward = 0.0
        step = 0
        info = {'fidelity': 0.0}

        for step in range(hp['steps_per_episode']):
            timestep_counter += 1
            action, log_prob = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done, log_prob)
            episode_actions.append(action)
            episode_reward += float(reward)
            obs = next_obs
            
            if timestep_counter % hp['update_timestep'] == 0:
                agent.learn()
            if done:
                break

        final_fidelity = float(info.get('fidelity', 0.0))
        all_fidelities.append(final_fidelity)
        all_action_sequences.append(episode_actions)

        writer.writerow([episode + 1, final_fidelity, step + 1, episode_reward, '-'.join(map(str, episode_actions))])

        if (episode + 1) % 100 == 0:
            avg_f = np.mean(all_fidelities[-100:])
            elapsed = time.time() - start_time
            print(f'Ep {episode+1}/{args.episodes} | Avg F {avg_f:.4f} | {elapsed:.0f}s')

    f_csv.close()
    
    agent.actor.save(os.path.join(out_dir, 'actor_final.keras'))
    agent.critic.save(os.path.join(out_dir, 'critic_final.keras'))
    print(f"\n[DONE] Saved models and logs to {out_dir}")

if __name__ == '__main__':
    sys.exit(main())