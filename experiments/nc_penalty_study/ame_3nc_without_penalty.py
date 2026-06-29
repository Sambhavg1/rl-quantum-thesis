#!/usr/bin/env python3
import os

# --- CRITICAL HPC FIX: Force memory growth BEFORE TensorFlow loads ---
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from functools import reduce
from itertools import combinations
from collections import Counter

# --- SERVER SAFE PLOTTING ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- GYM IMPORTS (Reverted to standard gym for Bulbasaur compat) ---
import gym
from gym import spaces

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. NVIDIA GPU SETUP
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Bulbasaur NVIDIA GPU Detected: {len(gpus)} device(s) active.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ WARNING: Running on CPU. GPU not detected.")

# Output Directory
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_ame46_bulbasaur_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. Universal Quantum Environment (AME 4,6)
# ==============================================================================
class UniversalQuantumEnv(gym.Env):
    def __init__(self, num_qudits=4, dim=6):
        super().__init__()
        self.n = num_qudits
        self.d = dim
        self.hilbert_dim = self.d ** self.n

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.hilbert_dim,),
            dtype=np.float32 
        )

        self.gates, self.gate_names = self._create_universal_gate_set()
        self.action_space = spaces.Discrete(len(self.gates))
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.zeros((self.d,) * self.n, dtype=np.complex128)
        self.state[(0,) * self.n] = 1.0 + 0j
        return self._get_obs()

    def step(self, action):
        gate_info = self.gates[action]
        gate_type = gate_info['type']
        
        if gate_type == 'single':
            mat, target = gate_info['tensor'], gate_info['targets'][0]
            self.state = np.tensordot(mat, self.state, axes=([1], [target]))
            self.state = np.moveaxis(self.state, 0, target)
            
        elif gate_type == 'two':
            mat, c, t = gate_info['tensor'], gate_info['targets'][0], gate_info['targets'][1]
            self.state = np.tensordot(mat, self.state, axes=([2, 3], [c, t]))
            self.state = np.moveaxis(self.state, [0, 1], [c, t])

        flat_state = self.state.ravel()
        norm_sq = np.vdot(flat_state, flat_state).real
        if norm_sq > 1e-18:
            self.state /= np.sqrt(norm_sq)

        entropy = self._calculate_ame_metric()
        k = self.n // 2
        max_possible_entropy = k * np.log(self.d)

        # Power-law Reward Shaping
        quality = entropy / max_possible_entropy
        reward = (quality ** 10.0) * 100.0

        done = bool(entropy >= (max_possible_entropy * 0.995))
        if done:
            reward += 500.0

        #reward -= 0.05

        return self._get_obs(), float(reward), done, {"entropy": entropy}

    def _get_obs(self):
        flat_state = self.state.ravel()
        return np.concatenate([flat_state.real, flat_state.imag]).astype(np.float32)

    def _calculate_ame_metric(self):
        k = self.n // 2
        subsystems = list(combinations(range(self.n), k))
        entropies = []
        state_tensor = self.state

        for keep_indices in subsystems:
            rho_reduced = self._get_reduced_density_matrix(state_tensor, keep_indices)
            evals = np.linalg.eigvalsh(rho_reduced)
            evals = evals[evals > 1e-10]
            vn_entropy = -np.sum(evals * np.log(evals))
            entropies.append(vn_entropy)

        return np.min(entropies)

    def _get_reduced_density_matrix(self, state_tensor, keep_indices):
        keep_set = set(keep_indices)
        trace_indices = tuple([i for i in range(self.n) if i not in keep_set])
        perm = list(keep_indices) + list(trace_indices)
        permuted = np.transpose(state_tensor, perm)

        dim_keep = self.d ** len(keep_indices)
        dim_trace = self.d ** len(trace_indices)

        psi_mat = permuted.reshape(dim_keep, dim_trace)
        return psi_mat @ psi_mat.conj().T

    def _create_universal_gate_set(self):
        gates = []
        names = []

        omega_d = np.exp(2 * np.pi * 1j / self.d)
        omega_gold_1 = np.exp(1j * np.pi / 10.0) 
        omega_gold_2 = np.exp(1j * np.pi / 5.0)   
        omega_gold_3 = np.exp(1j * np.pi / 7.0)   

        X_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        Z_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        F_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        P1_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        P2_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        P3_mat = np.zeros((self.d, self.d), dtype=np.complex128)

        for i in range(self.d):
            X_mat[(i + 1) % self.d, i] = 1.0
            Z_mat[i, i] = omega_d ** i
            P1_mat[i, i] = omega_gold_1 ** i
            P2_mat[i, i] = omega_gold_2 ** i
            P3_mat[i, i] = omega_gold_3 ** i
            for k in range(self.d):
                F_mat[i, k] = omega_d ** (i * k)
        F_mat /= np.sqrt(self.d)

        primitives = {'X': X_mat, 'Z': Z_mat, 'F': F_mat, 'P_pi10': P1_mat, 'P_pi5': P2_mat, 'P_pi7': P3_mat}
        for name, mat in primitives.items():
            for i in range(self.n):
                gates.append({'type': 'single', 'targets': (i,), 'tensor': mat})
                names.append(f"{name}_{i}")

        SUM_tensor = np.zeros((self.d, self.d, self.d, self.d), dtype=np.complex128)
        for c in range(self.d):
            for t in range(self.d):
                t_out = (t + c) % self.d
                SUM_tensor[c, t_out, c, t] = 1.0

        pairs = list(combinations(range(self.n), 2))
        for i, j in pairs:
            gates.append({'type': 'two', 'targets': (i, j), 'tensor': SUM_tensor})
            names.append(f"SUM_{i}_{j}")

        return gates, names

# ==============================================================================
# 3. PPO Agent (NVIDIA CUDA Compiled)
# ==============================================================================
class PPOAgent:
    def __init__(
        self,
        n_actions,
        n_features,
        initial_lr=3e-4,       
        steps_per_cycle=10000, 
        gamma=0.99,
        clip_ratio=0.1,
        policy_epochs=10,
        batch_size=4096,
        initial_entropy=0.1    
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size

        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=steps_per_cycle,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.01
        )

        self.initial_entropy = initial_entropy
        self.update_counter = 0  

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Standard Optimizers for NVIDIA GPUs
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()

        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []

    def _build_actor(self):
        inputs = layers.Input(shape=(self.n_features,), dtype=tf.float32)
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(self.n_actions, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.n_features,), dtype=tf.float32)
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    @tf.function(reduce_retracing=True)
    def _compiled_choose_action(self, state):
        probs = self.actor(state)
        dist = tf.random.categorical(tf.math.log(probs + 1e-10), 1)
        action = dist[0, 0]
        log_prob = tf.math.log(probs[0, action] + 1e-10)
        return action, log_prob

    def choose_action(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action, log_prob = self._compiled_choose_action(state_tensor)
        return int(action.numpy()), float(log_prob.numpy())

    @tf.function(reduce_retracing=True)
    def _train_step(self, states, actions, advantages, old_log_probs, entropy_coef):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values = tf.squeeze(self.critic(states, training=True))
            critic_loss = self.critic_loss_fn(advantages + values, values)

            new_probs = self.actor(states, training=True)
            action_indices = tf.stack(
                [tf.range(tf.shape(actions)[0], dtype=tf.int32), actions],
                axis=1
            )
            new_log_probs = tf.math.log(tf.gather_nd(new_probs, action_indices) + 1e-10)

            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages

            dist_entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
            entropy_mean = tf.reduce_mean(dist_entropy)

            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            total_actor_loss = actor_loss - (entropy_coef * entropy_mean)

        actor_grads = actor_tape.gradient(total_actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def learn(self):
        if not self.states:
            return

        states_arr = np.array(self.states, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.int32)
        rewards_arr = np.array(self.rewards, dtype=np.float32)
        next_states_arr = np.array(self.next_states, dtype=np.float32)
        dones_arr = np.array(self.dones, dtype=np.float32)
        old_log_probs_arr = np.array(self.log_probs, dtype=np.float32)

        # <--- CRITICAL HPC FIX: Batched Critic Evaluation restored --->
        v_list = [self.critic(states_arr[i:i+self.batch_size]).numpy().flatten() for i in range(0, len(states_arr), self.batch_size)]
        values = np.concatenate(v_list)
        
        nv_list = [self.critic(next_states_arr[i:i+self.batch_size]).numpy().flatten() for i in range(0, len(next_states_arr), self.batch_size)]
        next_values = np.concatenate(nv_list)

        deltas = rewards_arr + self.gamma * next_values * (1 - dones_arr) - values
        advantages = np.zeros_like(rewards_arr)

        last_advantage = 0.0
        for t in reversed(range(len(rewards_arr))):
            advantages[t] = deltas[t] + self.gamma * 0.95 * last_advantage * (1 - dones_arr[t])
            last_advantage = advantages[t]

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        self.update_counter += 1
        entropy_coef = self.initial_entropy * (0.999 ** self.update_counter)

        num_samples = len(self.states)
        indices = np.arange(num_samples)

        for _ in range(self.policy_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                self._train_step(
                    tf.convert_to_tensor(states_arr[batch_indices]),
                    tf.convert_to_tensor(actions_arr[batch_indices]),
                    tf.convert_to_tensor(advantages[batch_indices]),
                    tf.convert_to_tensor(old_log_probs_arr[batch_indices]),
                    tf.constant(entropy_coef, dtype=tf.float32)
                )

        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.next_states.clear(); self.dones.clear(); self.log_probs.clear()

# ==============================================================================
# 4. Plot Saving 
# ==============================================================================
def save_plots(all_entropies, successful_sequences, env):
    k = env.n // 2
    max_theoretical = k * np.log(env.d)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6))
    plt.plot(all_entropies, alpha=0.3, color='royalblue', label='Episode Max Entropy')

    if len(all_entropies) >= 200:
        window = 200
        moving_avg = np.convolve(all_entropies, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(all_entropies)), moving_avg, color='navy', linewidth=2, label=f'Moving Avg ({window})')

    plt.axhline(y=max_theoretical, color='darkgreen', linestyle='--', linewidth=2, label=f'Theoretical AME Limit ({max_theoretical:.4f})')
    plt.title(f'PPO Convergence: AME({env.n},{env.d}) Search', fontsize=16, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Von Neumann Entropy (Nats)')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "ame46_training_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    if successful_sequences:
        seq_counts = Counter([s[0] for s in successful_sequences])
        top_10 = seq_counts.most_common(10)

        labels, counts = [], []
        for seq_tuple, count in top_10:
            full_str = "-".join(seq_tuple)
            label = full_str[:25] + "..." if len(full_str) > 25 else full_str
            labels.append(label)
            counts.append(count)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(labels, counts, color='teal', edgecolor='black')
        plt.xlabel('Gate Sequence Block')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent Near-AME Sequences', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        hist_path = os.path.join(OUTPUT_DIR, "ame46_sequence_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()

# ==============================================================================
# 5. Main
# ==============================================================================
if __name__ == '__main__':
    NUM_QUDITS = 4
    DIM = 6
    EPISODES = 70000
    STEPS_PER_EPISODE = 400
    UPDATE_TIMESTEP = 8192

    env = UniversalQuantumEnv(num_qudits=NUM_QUDITS, dim=DIM)

    agent = PPOAgent(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        initial_lr=3e-4,         
        steps_per_cycle=10000,   
        gamma=0.99,
        clip_ratio=0.1,
        policy_epochs=10,
        batch_size=4096,
        initial_entropy=0.1      
    )

    print(f"--- Starting Bulbasaur AME({NUM_QUDITS}, {DIM}) Search ---")

    all_entropies = []
    successful_sequences = []
    timestep_counter = 0

    k_half = NUM_QUDITS // 2
    max_possible = k_half * np.log(DIM)
    global_max_entropy = 0.0

    start_time = time.time()

    for episode in range(EPISODES):
        observation = env.reset()
        episode_max_entropy = 0.0
        episode_actions = []

        for step in range(STEPS_PER_EPISODE):
            timestep_counter += 1
            action, log_prob = agent.choose_action(observation)
            episode_actions.append(env.gate_names[action])

            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, done, log_prob)

            observation = next_observation
            episode_max_entropy = max(episode_max_entropy, info["entropy"])

            if timestep_counter % UPDATE_TIMESTEP == 0:
                agent.learn()

            if done:
                break

        all_entropies.append(episode_max_entropy)

        if episode_max_entropy > global_max_entropy:
            global_max_entropy = episode_max_entropy
            if global_max_entropy > 0.95 * max_possible:
                print(f"!!! NEW RECORD: {global_max_entropy:.4f} / {max_possible:.4f} (Episode {episode+1}) !!!")

        if episode_max_entropy > (max_possible * 0.99):
            successful_sequences.append((tuple(episode_actions), episode_max_entropy))

        if (episode + 1) % 100 == 0:
            avg_ent = float(np.mean(all_entropies[-100:]))
            curr_lr = float(agent.lr_schedule(agent.actor_optimizer.iterations).numpy())
            elapsed = time.time() - start_time
            print(f'Ep: {episode+1}/{EPISODES} | Avg(100): {avg_ent:.4f} | Best: {global_max_entropy:.4f}/{max_possible:.4f} | LR: {curr_lr:.2e} | Time: {elapsed:.1f}s')

    # --- SAVE RESULTS ---
    print("\nGenerating Thesis Figures...")
    save_plots(all_entropies, successful_sequences, env)

    csv_path = os.path.join(OUTPUT_DIR, "ame46_results.csv")
    print(f"Saving {len(successful_sequences)} successful sequences to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode_Entropy', 'Sequence_Length', 'Gate_Sequence'])
        for seq_tuple, ent in successful_sequences:
            writer.writerow([f"{ent:.6f}", len(seq_tuple), "-".join(seq_tuple)])

    print(f"All processing complete. Results saved to {OUTPUT_DIR}/")