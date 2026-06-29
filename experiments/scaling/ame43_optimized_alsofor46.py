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

# --- SERVER SAFE PLOTTING ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
from gym import spaces

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
OUTPUT_DIR = f"results_ame_optimized_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. HIGH-PERFORMANCE QUANTUM ENVIRONMENT (TENSOR CONTRACTION)
# ==============================================================================
class UniversalQuantumEnv(gym.Env):
    def __init__(self, num_qudits=4, dim=6):
        super().__init__()
        self.n = num_qudits
        self.d = dim
        self.hilbert_dim = self.d ** self.n

        # Observation space is a flat array (Real + Imaginary)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.hilbert_dim,),
            dtype=np.float64
        )

        # Build gates as lightweight tensors instead of full matrices
        self.gates, self.gate_names = self._create_tensor_gate_set()
        self.action_space = spaces.Discrete(len(self.gates))
        
        # --- PRECOMPUTE PHYSICS CONSTANTS FOR EXTREME SPEED ---
        self.k = self.n // 2
        self.max_possible_entropy = self.k * np.log(self.d)
        
        # Precompute partial trace permutations
        subsystems = list(combinations(range(self.n), self.k))
        self.trace_perms = []
        self.dim_keep = self.d ** self.k
        self.dim_trace = self.d ** (self.n - self.k)
        
        for keep_indices in subsystems:
            trace_indices = tuple(i for i in range(self.n) if i not in set(keep_indices))
            perm = list(keep_indices) + list(trace_indices)
            self.trace_perms.append(perm)

        self.state = None
        self.reset()

    def reset(self):
        # State is now an N-dimensional tensor: shape (D, D, ..., D)
        self.state = np.zeros((self.d,) * self.n, dtype=np.complex128)
        self.state[(0,) * self.n] = 1.0 + 0j
        return self._get_obs()

    def step(self, action):
        gate_info = self.gates[action]
        gate_type = gate_info['type']
        
        # --- FAST TENSOR CONTRACTION ---
        if gate_type == 'single':
            # Tensordot applies the DxD matrix only to the target qudit axis
            mat, target = gate_info['tensor'], gate_info['targets'][0]
            self.state = np.tensordot(mat, self.state, axes=([1], [target]))
            self.state = np.moveaxis(self.state, 0, target)
            
        elif gate_type == 'two':
            # Tensordot applies the (D,D,D,D) tensor to the control and target axes
            mat, c, t = gate_info['tensor'], gate_info['targets'][0], gate_info['targets'][1]
            self.state = np.tensordot(mat, self.state, axes=([2, 3], [c, t]))
            self.state = np.moveaxis(self.state, [0, 1], [c, t])

        # --- FAST NORM ---
        # Ravel creates a flat view instantly without copying memory
        flat_state = self.state.ravel()
        norm_sq = np.vdot(flat_state, flat_state).real
        if norm_sq > 1e-18:
            self.state /= np.sqrt(norm_sq)

        entropy = self._calculate_ame_metric()
        
        # Power-law reward shaping (x^10)
        quality = entropy / self.max_possible_entropy
        reward = (quality ** 10.0) * 100.0

        done = entropy >= (self.max_possible_entropy * 0.995)
        if done:
            reward += 500.0

        reward -= 0.05

        return self._get_obs(), reward, done, {"entropy": entropy}

    def _get_obs(self):
        # Neural Network needs a flat 1D array
        flat_state = self.state.ravel()
        return np.concatenate([flat_state.real, flat_state.imag])

    def _calculate_ame_metric(self):
        """Highly optimized Von Neumann entropy calculation."""
        entropies = np.zeros(len(self.trace_perms))
        eps = 1e-12 # Vectorized epsilon prevents log(0) instantly

        for idx, perm in enumerate(self.trace_perms):
            # Transpose creates a fast memory view, reshape prepares it for matrix multiplication
            permuted = np.transpose(self.state, perm)
            psi_mat = permuted.reshape(self.dim_keep, self.dim_trace)
            
            # Density matrix
            rho_reduced = psi_mat @ psi_mat.conj().T
            
            # Fast Hermitian Eigensolver
            evals = np.linalg.eigvalsh(rho_reduced)
            
            # Fast vectorized entropy
            evals = np.abs(evals) + eps
            entropies[idx] = -np.sum(evals * np.log(evals))

        return np.min(entropies)

    def _create_tensor_gate_set(self):
        """Creates lightweight tensors instead of heavy np.kron matrices."""
        gates = []
        names = []

        omega_d = np.exp(2 * np.pi * 1j / self.d)
        omega_gold_1 = np.exp(1j * np.pi / 10.0)
        omega_gold_2 = np.exp(1j * np.pi / 5.0)
        omega_gold_3 = np.exp(1j * np.pi / 7.0)

        # Single Qudit Primitive Matrices (D x D)
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
        
        # Append Single Qudit Gates
        for name, mat in primitives.items():
            for i in range(self.n):
                gates.append({'type': 'single', 'targets': (i,), 'tensor': mat})
                names.append(f"{name}_{i}")

        # Two Qudit SUM Gates (D x D x D x D Tensor)
        # Represents mapping |c, t> -> |c, (t+c)%D>
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
# 2. PPO AGENT
# ==============================================================================
class PPOAgent:
    def __init__(self, n_actions, n_features, initial_lr=3e-4, steps_per_cycle=10000,
                 gamma=0.99, clip_ratio=0.1, policy_epochs=10, batch_size=4096, initial_entropy=0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size
        self.initial_entropy = initial_entropy
        self.update_counter = 0  

        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr, first_decay_steps=steps_per_cycle,
            t_mul=2.0, m_mul=0.9, alpha=0.01
        )

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()
        self.clear_memory()

    def _build_actor(self):
        inputs = layers.Input(shape=(self.n_features,))
        # <--- CHANGED HERE: Restored network width to 2048->1024 for the 2592-dimensional state vector --->
        x = layers.Dense(2048, activation='relu')(inputs)
        x = layers.Dense(1024, activation='relu')(x)
        outputs = layers.Dense(self.n_actions, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.n_features,))
        # <--- CHANGED HERE: Restored network width to 2048->1024 for the Critic as well --->
        x = layers.Dense(2048, activation='relu')(inputs)
        x = layers.Dense(1024, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state)
        action = tf.random.categorical(tf.math.log(probs + 1e-10), 1).numpy().item()
        log_prob = tf.math.log(probs[0, action] + 1e-10)
        return action, log_prob

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []

    @tf.function
    def train_step(self, states, actions, advantages, old_log_probs, entropy_coef):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values = tf.squeeze(self.critic(states, training=True))
            critic_loss = self.critic_loss_fn(advantages + values, values)

            new_probs = self.actor(states, training=True)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
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

        values = self.critic(states_arr).numpy().flatten()
        next_values = self.critic(next_states_arr).numpy().flatten()

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
                self.train_step(
                    tf.convert_to_tensor(states_arr[batch_indices]),
                    tf.convert_to_tensor(actions_arr[batch_indices]),
                    tf.convert_to_tensor(advantages[batch_indices]),
                    tf.convert_to_tensor(old_log_probs_arr[batch_indices]),
                    tf.constant(entropy_coef, dtype=tf.float32)
                )
        self.clear_memory()

# ==============================================================================
# 3. UTILS & MAIN
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
    plt.title(f'Training Progress: AME Search (N={env.n}, D={env.d})')
    plt.xlabel('Episode')
    plt.ylabel('Von Neumann Entropy (Nats)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    NUM_QUDITS = 4
    DIM = 3
    EPISODES = 100000
    STEPS_PER_EPISODE = 70
    UPDATE_TIMESTEP = 8192

    env = UniversalQuantumEnv(num_qudits=NUM_QUDITS, dim=DIM)

    agent = PPOAgent(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        initial_lr=3e-4, steps_per_cycle=10000, gamma=0.99,
        clip_ratio=0.1, policy_epochs=10, batch_size=4096, initial_entropy=0.1       
    )

    print(f"--- Starting Optimized AME({NUM_QUDITS}, {DIM}) Search ---")
    
    all_entropies = []
    successful_sequences = []
    timestep_counter = 0
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
            if done: break

        all_entropies.append(episode_max_entropy)

        if episode_max_entropy > global_max_entropy:
            global_max_entropy = episode_max_entropy
            if global_max_entropy > 0.95 * env.max_possible_entropy:
                print(f"!!! NEW RECORD: {global_max_entropy:.4f} / {env.max_possible_entropy:.4f} !!!")

        if episode_max_entropy > (env.max_possible_entropy * 0.99):
            successful_sequences.append((tuple(episode_actions), episode_max_entropy))

        if (episode + 1) % 100 == 0:
            avg_ent = float(np.mean(all_entropies[-100:]))
            curr_lr = float(agent.lr_schedule(agent.actor_optimizer.iterations).numpy())
            elapsed = time.time() - start_time
            print(f'Ep: {episode+1}/{EPISODES} | Avg(100): {avg_ent:.4f} | Best: {global_max_entropy:.4f} | LR: {curr_lr:.2e} | Time: {elapsed:.1f}s')

    save_plots(all_entropies, successful_sequences, env)
    csv_path = os.path.join(OUTPUT_DIR, "ame_results.csv")
    
    # <--- CHANGED HERE: Sort sequences by entropy descending before saving to CSV --->
    successful_sequences.sort(key=lambda x: x[1], reverse=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode_Entropy', 'Sequence_Length', 'Gate_Sequence'])
        for seq_tuple, ent in successful_sequences:
            writer.writerow([f"{ent:.6f}", len(seq_tuple), "-".join(seq_tuple)])