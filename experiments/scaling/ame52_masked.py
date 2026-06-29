#!/usr/bin/env python3
import os
import csv
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from itertools import combinations
from collections import Counter
import gym
from gym import spaces

# --- SERVER SAFE PLOTTING ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- GPU SETUP ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s) active.")
    except RuntimeError as e:
        print(e)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_ame52_masked_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. ENVIRONMENT: Fast Tensor Engine + Dynamic Action Masking
# ==============================================================================
class UniversalQuantumEnv_Masked(gym.Env):
    def __init__(self, num_qudits=5, dim=2):
        super().__init__()
        self.n = num_qudits
        self.d = dim
        self.hilbert_dim = self.d ** self.n
        self.k = self.n // 2
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, 
                                            shape=(2 * self.hilbert_dim,), dtype=np.float64)
        
        self.gates, self.gate_names = self._create_tensor_gate_set()
        self.action_space = spaces.Discrete(len(self.gates))
        
        # Fast evaluation precomputations
        self.target_metric_max = self.k * np.log(self.d)
        subsystems = list(combinations(range(self.n), self.k))
        self.trace_perms = []
        self.dim_keep = self.d ** self.k
        self.dim_trace = self.d ** (self.n - self.k)
        for keep_indices in subsystems:
            trace_indices = tuple(i for i in range(self.n) if i not in set(keep_indices))
            self.trace_perms.append(list(keep_indices) + list(trace_indices))

        self.state = None
        self.last_action = None
        self.reset()

    def reset(self):
        self.state = np.zeros((self.d,) * self.n, dtype=np.complex128)
        self.state[(0,) * self.n] = 1.0 + 0j
        self.last_action = None
        return self._get_obs(), self.get_action_mask()

    def step(self, action):
        gate_info = self.gates[action]
        
        # Fast Tensor Contraction
        if gate_info['type'] == 'single':
            mat, target = gate_info['tensor'], gate_info['targets'][0]
            self.state = np.tensordot(mat, self.state, axes=([1], [target]))
            self.state = np.moveaxis(self.state, 0, target)
        elif gate_info['type'] == 'two':
            mat, c, t = gate_info['tensor'], gate_info['targets'][0], gate_info['targets'][1]
            self.state = np.tensordot(mat, self.state, axes=([2, 3], [c, t]))
            self.state = np.moveaxis(self.state, [0, 1], [c, t])

        # Normalize
        flat_state = self.state.ravel()
        norm_sq = np.vdot(flat_state, flat_state).real
        if norm_sq > 1e-18:
            self.state /= np.sqrt(norm_sq)
            
        entropies = self._calculate_ame_metric()
        min_ent = np.min(entropies)
        
        # Reward shaping (Softmin + Power Law)
        beta = 10.0 
        softmin_ent = - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (entropies - min_ent)))) + min_ent
        quality = softmin_ent / self.target_metric_max
        reward = (quality ** 10.0) * 100.0 
        
        done = min_ent >= (self.target_metric_max * 0.999) 
        if done: reward += 500.0 
        reward -= 0.05 
        
        self.last_action = action
        return self._get_obs(), reward, done, {"entropy": min_ent, "mask": self.get_action_mask()}

    def _get_obs(self):
        flat = self.state.ravel()
        return np.concatenate([flat.real, flat.imag])

    def get_action_mask(self):
        """PHYSICS-INFORMED INDUCTIVE BIAS"""
        mask = np.ones(self.action_space.n, dtype=np.float32)
        if self.last_action is not None:
            last_info = self.gates[self.last_action]
            
            # 1. Identity Pruner: In d=2, all native gates are self-inverses.
            if self.d == 2:
                mask[self.last_action] = 0.0
                
            # 2. Canonical Commutation Order: Prevent disjoint single-qubit permutations
            if last_info['type'] == 'single':
                q_last = last_info['targets'][0]
                for a in range(self.action_space.n):
                    if self.gates[a]['type'] == 'single':
                        q_curr = self.gates[a]['targets'][0]
                        if q_curr < q_last:
                            mask[a] = 0.0
        return mask

    def _calculate_ame_metric(self):
        entropies = np.zeros(len(self.trace_perms))
        eps = 1e-12
        for idx, perm in enumerate(self.trace_perms):
            permuted = np.transpose(self.state, perm)
            psi_mat = permuted.reshape(self.dim_keep, self.dim_trace)
            rho_reduced = psi_mat @ psi_mat.conj().T
            evals = np.linalg.eigvalsh(rho_reduced)
            evals = np.abs(evals) + eps
            entropies[idx] = -np.sum(evals * np.log(evals))
        return entropies

    def _create_tensor_gate_set(self):
        gates, names = [], []
        omega_d = np.exp(2 * np.pi * 1j / self.d)
        
        X_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        Z_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        F_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        
        for i in range(self.d):
            X_mat[(i + 1) % self.d, i] = 1.0
            Z_mat[i, i] = omega_d ** i
            for k in range(self.d):
                F_mat[i, k] = omega_d ** (i * k)
        F_mat /= np.sqrt(self.d)

        # Single Qudit Gates
        for name, mat in {'X': X_mat, 'Z': Z_mat, 'F': F_mat}.items():
            for i in range(self.n):
                gates.append({'type': 'single', 'targets': (i,), 'tensor': mat})
                names.append(f"{name}_{i}")

        # Bidirectional SUM Gates
        SUM_tensor = np.zeros((self.d, self.d, self.d, self.d), dtype=np.complex128)
        for c in range(self.d):
            for t in range(self.d):
                t_out = (t + c) % self.d
                SUM_tensor[c, t_out, c, t] = 1.0

        for i, j in combinations(range(self.n), 2):
            gates.append({'type': 'two', 'targets': (i, j), 'tensor': SUM_tensor})
            names.append(f"SUM_{i}_{j}")
            gates.append({'type': 'two', 'targets': (j, i), 'tensor': SUM_tensor})
            names.append(f"SUM_{j}_{i}")
            
        return gates, names

# ==============================================================================
# 2. PPO AGENT (Logit Masking Architecture)
# ==============================================================================
class MaskedPPOAgent:
    def __init__(self, n_actions, n_features, initial_lr=3e-4, decay_steps=10000):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.clip_ratio = 0.1
        self.policy_epochs = 10
        self.batch_size = 512
        self.initial_entropy = 0.05
        
        # --- FIX: Decay steps initialized properly ---
        self.decay_steps = decay_steps
        
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr, decay_steps=decay_steps,
            end_learning_rate=1e-6, power=1.0)
        self.step_counter = 0

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.clear_memory()

    def _build_actor(self):
        inputs = layers.Input(shape=(self.n_features,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        # CRITICAL CHANGE: Output raw logits (linear), NO SOFTMAX yet!
        outputs = layers.Dense(self.n_actions, activation='linear')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.n_features,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def store_transition(self, state, action, mask, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.masks.append(mask) 
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def choose_action(self, state, mask):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor([mask], dtype=tf.float32)
        
        # INDUCTIVE BIAS APPLIED HERE
        logits = self.actor(state_tensor)
        masked_logits = logits + (mask_tensor - 1.0) * 1e9
        probs = tf.nn.softmax(masked_logits)
        
        action = tf.random.categorical(tf.math.log(probs + 1e-10), 1).numpy().item()
        log_prob = tf.math.log(probs[0, action] + 1e-10)
        return action, log_prob

    def choose_best_action(self, state, mask):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor([mask], dtype=tf.float32)
        logits = self.actor(state_tensor)
        masked_logits = logits + (mask_tensor - 1.0) * 1e9
        return tf.argmax(masked_logits[0]).numpy().item() 

    def clear_memory(self):
        self.states, self.actions, self.masks, self.rewards = [], [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []
    
    @tf.function
    def train_step(self, states, actions, masks, advantages, old_log_probs, entropy_coef):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values = tf.squeeze(self.critic(states, training=True))
            critic_loss = tf.reduce_mean((advantages + values - values)**2) 
            
            # Reconstruct masked probabilities during backprop
            logits = self.actor(states, training=True)
            masked_logits = logits + (masks - 1.0) * 1e9
            new_probs = tf.nn.softmax(masked_logits)
            
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            new_log_probs = tf.math.log(tf.gather_nd(new_probs, action_indices) + 1e-10)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            
            dist_entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
            entropy_mean = tf.reduce_mean(dist_entropy)
            
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2)) - (entropy_coef * entropy_mean)
            
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def learn(self):
        if not self.states: return
        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int32)
        masks_arr = np.array(self.masks, dtype=np.float32)
        rewards_arr = np.array(self.rewards, dtype=np.float32)
        next_states_arr = np.array(self.next_states, dtype=np.float32)
        dones_arr = np.array(self.dones, dtype=np.float32)
        old_log_probs_arr = np.array(self.log_probs, dtype=np.float32)

        values = self.critic(states_arr).numpy().flatten()
        next_values = self.critic(next_states_arr).numpy().flatten()
        deltas = rewards_arr + self.gamma * next_values * (1 - dones_arr) - values
        advantages = np.zeros_like(rewards_arr)
        last_advantage = 0
        for t in reversed(range(len(rewards_arr))):
            advantages[t] = deltas[t] + self.gamma * 0.95 * last_advantage * (1 - dones_arr[t])
            last_advantage = advantages[t]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        self.step_counter += 1
        # --- FIX: Prevent division by zero with max(1, ...) ---
        divisor = max(1, self.decay_steps // 2048)
        entropy_coef = max(0.0, self.initial_entropy * (1 - self.step_counter / divisor))
        
        num_samples = len(self.states)
        indices = np.arange(num_samples)
        for _ in range(self.policy_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                self.train_step(
                    tf.convert_to_tensor(states_arr[batch_indices]),
                    tf.convert_to_tensor(actions_arr[batch_indices]),
                    tf.convert_to_tensor(masks_arr[batch_indices]),
                    tf.convert_to_tensor(advantages[batch_indices]),
                    tf.convert_to_tensor(old_log_probs_arr[batch_indices]),
                    tf.constant(entropy_coef, dtype=tf.float32)
                )
        self.clear_memory()

# ==============================================================================
# 3. UTILS & MAIN EXECUTION
# ==============================================================================
def save_results(all_entropies, successful_sequences, env):
    max_theoretical = env.target_metric_max
    plt.figure(figsize=(12, 5))
    plt.plot(all_entropies, alpha=0.5, label='Episode Max Entropy')
    if len(all_entropies) >= 100:
        moving_avg = np.convolve(all_entropies, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_entropies)), moving_avg, color='red', label='Moving Avg (100)')
    plt.axhline(y=max_theoretical, color='green', linestyle='--', label=f'AME Limit ({max_theoretical:.4f})')
    plt.title(f'Physics-Informed Training (Masked) (N={env.n}, D={env.d})')
    plt.xlabel('Episode')
    plt.ylabel('Metric Score')
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    plot_path = os.path.join(OUTPUT_DIR, "masked_training_curve.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "masked_best_sequences.csv")
    successful_sequences.sort(key=lambda x: x[1], reverse=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode_Entropy', 'Sequence_Length', 'Gate_Sequence'])
        for seq_tuple, ent in successful_sequences:
            writer.writerow([f"{ent:.6f}", len(seq_tuple), "-".join(seq_tuple)])
            
    print(f"\nResults securely saved to directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    EPISODES = 10000            
    STEPS_PER_EPISODE = 50
    UPDATE_TIMESTEP = 4096      
    
    env = UniversalQuantumEnv_Masked(num_qudits=5, dim=2)
    agent = MaskedPPOAgent(
        n_actions=env.action_space.n, n_features=env.observation_space.shape[0],
        initial_lr=3e-4, decay_steps=EPISODES
    )

    print(f"--- Starting Action-Masked AME(5,2) Search ---")
    start_time = time.time()
    
    all_entropies = []
    successful_sequences = [] 
    timestep_counter = 0
    global_max_entropy = 0.0

    for episode in range(EPISODES):
        observation, current_mask = env.reset()
        episode_max_entropy = 0
        episode_actions = [] 
        
        for step in range(STEPS_PER_EPISODE):
            timestep_counter += 1
            action, log_prob = agent.choose_action(observation, current_mask) 
            episode_actions.append(env.gate_names[action])
            
            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, current_mask, reward, next_observation, done, log_prob)
            
            observation = next_observation
            current_mask = info["mask"]
            episode_max_entropy = max(episode_max_entropy, info['entropy'])

            if timestep_counter % UPDATE_TIMESTEP == 0:
                agent.learn()
            if done: break
        
        all_entropies.append(episode_max_entropy)
        
        if episode_max_entropy > global_max_entropy:
            global_max_entropy = episode_max_entropy
            if global_max_entropy > (env.target_metric_max * 0.95):
                print(f"!!! NEW RECORD: {global_max_entropy:.4f} (Episode {episode+1}) !!!")

        if episode_max_entropy > (env.target_metric_max * 0.99):
            successful_sequences.append((tuple(episode_actions), episode_max_entropy))

        if (episode + 1) % 100 == 0:
            avg_ent = np.mean(all_entropies[-100:])
            curr_lr = agent.lr_schedule(agent.actor_optimizer.iterations).numpy()
            print(f'Ep: {episode + 1}/{EPISODES} | Avg Max: {avg_ent:.4f} | LR: {curr_lr:.2e} | Time: {time.time()-start_time:.1f}s')

    print("\n--- Running Deterministic Final Exam ---")
    observation, current_mask = env.reset()
    for _ in range(STEPS_PER_EPISODE):
        action = agent.choose_best_action(observation, current_mask) 
        observation, _, done, info = env.step(action)
        current_mask = info["mask"]
        if done: break
    print(f"Final Exam Metric: {info['entropy']:.4f} / {env.target_metric_max:.4f}")

    save_results(all_entropies, successful_sequences, env)