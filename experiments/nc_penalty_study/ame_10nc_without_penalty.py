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

# --- GYM IMPORTS (Reverted to classic gym for Bulbasaur) ---
import gym
from gym import spaces

# --- BULBASAUR NVIDIA GPU SETUP ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Bulbasaur NVIDIA GPU Detected: {len(gpus)} device(s) active.")
    except RuntimeError as e:
        print(f"GPU Allocation Error: {e}")
else:
    print("⚠️ WARNING: Running on CPU. GPU not detected.")

# Output Directory
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_ame46_10angles_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. Universal Quantum Environment (AME 4,6) - 10 ANGLE EDITION
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
        self.state = np.zeros(self.hilbert_dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0j
        return self._get_obs()

    def step(self, action):
        gate_matrix = self.gates[action]
        self.state = gate_matrix @ self.state

        norm = np.linalg.norm(self.state)
        if norm > 1e-9:
            self.state /= norm

        entropy = self._calculate_ame_metric()
        k = self.n // 2
        max_possible_entropy = k * np.log(self.d)

        quality = entropy / max_possible_entropy
        reward = (quality ** 10.0) * 100.0

        done = bool(entropy >= (max_possible_entropy * 0.995))
        
        if done:
            reward += 500.0

        #reward -= 0.05

        return self._get_obs(), reward, done, {"entropy": entropy}

    def _get_obs(self):
        return np.concatenate([self.state.real, self.state.imag]).astype(np.float32)

    def _create_universal_gate_set(self):
        gates = []
        names = []

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

        primitives = {'X': X_mat, 'Z': Z_mat, 'F': F_mat}

        # <--- 10 NON-CLIFFORD ANGLES --->
        angles = [np.pi/x for x in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]]
        for idx, angle in enumerate(angles):
            P_mat = np.zeros((self.d, self.d), dtype=np.complex128)
            for i in range(self.d):
                P_mat[i, i] = np.exp(1j * angle) ** i
            primitives[f'P_angle_{idx}'] = P_mat

        for name, mat in primitives.items():
            for i in range(self.n):
                op_list = [np.eye(self.d, dtype=np.complex128)] * self.n
                op_list[i] = mat
                gates.append(reduce(np.kron, op_list))
                names.append(f"{name}_qudit_{i}")

        pairs = list(combinations(range(self.n), 2))
        for i, j in pairs:
            U = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=np.complex128)
            for basis_idx in range(self.hilbert_dim):
                digits = []
                temp = basis_idx
                for _ in range(self.n):
                    digits.insert(0, temp % self.d)
                    temp //= self.d

                c_val = digits[i]
                t_val = digits[j]
                digits[j] = (t_val + c_val) % self.d

                new_idx = 0
                for digit in digits:
                    new_idx = new_idx * self.d + digit
                U[new_idx, basis_idx] = 1.0

            gates.append(U)
            names.append(f"SUM_{i}_{j}")

        return gates, names

    def _calculate_ame_metric(self, state=None):
        if state is None:
            state = self.state

        k = self.n // 2
        subsystems = list(combinations(range(self.n), k))
        entropies = []
        state_tensor = state.reshape([self.d] * self.n)

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

# ==============================================================================
# 2. PPO Agent 
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

        # --- HPC FIX: Standard Optimizers for NVIDIA GPUs ---
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
        probs = self.actor(state, training=False)
        action = tf.random.categorical(tf.math.log(probs + 1e-10), 1)
        action_idx = action[0, 0]
        log_prob = tf.math.log(probs[0, action_idx] + 1e-10)
        return action_idx, log_prob

    def choose_action(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_idx, log_prob = self._compiled_choose_action(state_tensor)
        return action_idx.numpy().item(), log_prob.numpy().item()

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []

    @tf.function(reduce_retracing=True)
    def train_step(self, states, actions, advantages, old_log_probs, entropy_coef):
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

        # --- HPC FIX: Batched Critic Evaluation to prevent RAM spike ---
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
        
        current_decay = self.initial_entropy * (0.999 ** self.update_counter)
        entropy_coef = max(0.02, current_decay)

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
                    tf.convert_to_tensor(advantages[batch_indices]),
                    tf.convert_to_tensor(old_log_probs_arr[batch_indices]),
                    tf.constant(entropy_coef, dtype=tf.float32)
                )

        self.clear_memory()

# ==============================================================================
# 3. Plot Saving 
# ==============================================================================
def save_plots(all_entropies, successful_sequences, env):
    k = env.n // 2
    max_theoretical = k * np.log(env.d)

    plt.figure(figsize=(12, 5))
    plt.plot(all_entropies, alpha=0.5, label='Episode Max Entropy')

    if len(all_entropies) >= 200:
        window = 200
        moving_avg = np.convolve(all_entropies, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(all_entropies)), moving_avg, label=f'Moving Avg ({window})')

    plt.axhline(y=max_theoretical, color='green', linestyle='--', label=f'AME Limit ({max_theoretical:.4f})')
    plt.title(f'Training Progress: 10-Angle AME Search (N={env.n}, D={env.d})')
    plt.xlabel('Episode')
    plt.ylabel('Von Neumann Entropy (Nats)')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(OUTPUT_DIR, "ame46_training_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training plot to {plot_path}")

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
        plt.bar(labels, counts)
        plt.xlabel('Gate Sequence')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent Successful Sequences')
        plt.xticks(rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.35)

        hist_path = os.path.join(OUTPUT_DIR, "ame46_sequence_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram to {hist_path}")

# ==============================================================================
# 4. Main
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

    print(f"--- Starting 10-Angle AME({NUM_QUDITS}, {DIM}) Search on Bulbasaur ---")

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
    save_plots(all_entropies, successful_sequences, env)

    csv_path = os.path.join(OUTPUT_DIR, "ame46_results.csv")
    print(f"\nSaving {len(successful_sequences)} successful sequences to {csv_path}...")
    
    # Sort descending by entropy so best result is at the top
    successful_sequences.sort(key=lambda x: x[1], reverse=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode_Entropy', 'Sequence_Length', 'Gate_Sequence'])
        for seq_tuple, ent in successful_sequences:
            writer.writerow([f"{ent:.6f}", len(seq_tuple), "-".join(seq_tuple)])

    print(f"Results saved to {OUTPUT_DIR}")