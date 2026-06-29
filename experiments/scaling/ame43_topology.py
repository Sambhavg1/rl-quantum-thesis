#!/usr/bin/env python3
import os
import csv
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from itertools import combinations
from collections import Counter
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

# ==============================================================================
# 1. UNIVERSAL QUANTUM ENVIRONMENT AME(4,3) WITH REWARD TOPOLOGIES
# ==============================================================================
class UniversalQuantumEnv(gym.Env):
    def __init__(self, num_qudits=4, dim=3, reward_strategy='min'):
        super(UniversalQuantumEnv, self).__init__()
        self.n = num_qudits
        self.d = dim
        self.hilbert_dim = self.d ** self.n
        self.reward_strategy = reward_strategy
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, 
                                            shape=(2 * self.hilbert_dim,), dtype=np.float64)
        
        self.gates, self.gate_names = self._create_universal_gate_set()
        self.action_space = spaces.Discrete(len(self.gates))
        self.state = None
        
        self.k = self.n // 2
        self.subsystems = list(combinations(range(self.n), self.k))
        self.max_possible_entropy = self.k * np.log(self.d)
        self.reset()

    def reset(self):
        self.state = np.zeros(self.hilbert_dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0j
        return self._get_obs()

    def step(self, action):
        gate_matrix = self.gates[action]
        self.state = gate_matrix @ self.state
        
        norm = np.linalg.norm(self.state)
        if norm > 1e-9: self.state /= norm
            
        entropies = self._calculate_all_entropies()
        actual_min_entropy = np.min(entropies)
        
        # --- REWARD TOPOLOGY SCALARIZATION ---
        if self.reward_strategy == 'min':
            metric = actual_min_entropy
        elif self.reward_strategy == 'sum':
            metric = np.mean(entropies)
        elif self.reward_strategy == 'all':
            metric = np.mean(entropies) - np.var(entropies)
        elif self.reward_strategy == 'softmin':
            beta = 10.0
            metric = actual_min_entropy - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (entropies - actual_min_entropy))))
        elif self.reward_strategy == 'product':
            fractions = entropies / self.max_possible_entropy
            metric = np.prod(fractions) * self.max_possible_entropy
        else:
            metric = actual_min_entropy

        reward = (metric / self.max_possible_entropy) * 10.0 
        
        done = actual_min_entropy >= (self.max_possible_entropy * 0.999) 
        if done: reward += 100.0 
        reward -= 0.01 
        
        return self._get_obs(), reward, done, {"actual_min_entropy": actual_min_entropy}

    def _get_obs(self):
        return np.concatenate([self.state.real, self.state.imag])

    def _calculate_all_entropies(self, state=None):
        if state is None: state = self.state
        entropies = []
        state_tensor = state.reshape([self.d] * self.n)
        for keep_indices in self.subsystems:
            rho_reduced = self._get_reduced_density_matrix(state_tensor, keep_indices)
            evals = np.linalg.eigvalsh(rho_reduced)
            evals = evals[evals > 1e-10]
            vn_entropy = -np.sum(evals * np.log(evals))
            entropies.append(vn_entropy)
        return np.array(entropies)

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
        gates, names = [], []
        omega = np.exp(2 * np.pi * 1j / self.d)
        
        X_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        Z_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        F_mat = np.zeros((self.d, self.d), dtype=np.complex128)
        
        for i in range(self.d):
            X_mat[(i + 1) % self.d, i] = 1.0
            Z_mat[i, i] = omega**i
            for k in range(self.d):
                F_mat[i, k] = omega**(i * k)
        F_mat /= np.sqrt(self.d)

        primitives = {'X': X_mat, 'Z': Z_mat, 'F': F_mat}
        for name, mat in primitives.items():
            for i in range(self.n):
                op_list = [np.eye(self.d, dtype=np.complex128)] * self.n
                op_list[i] = mat
                gates.append(reduce(np.kron, op_list))
                names.append(f"{name}_{i}")

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

# ==============================================================================
# 2. PPO AGENT (Hardware Maximized)
# ==============================================================================
class PPOAgent:
    def __init__(self, n_actions, n_features, decay_steps):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.clip_ratio = 0.2      # Relaxed for faster convergence
        self.policy_epochs = 20    # Increased to squeeze the larger batch
        self.batch_size = 512      # Increased batch size for GPU utilization
        self.initial_entropy = 0.05
        
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=3e-4, decay_steps=decay_steps, end_learning_rate=1e-6, power=1.0)
            
        self.step_counter = 0
        self.decay_steps = decay_steps

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()
        self.clear_memory()

    def _build_actor(self):
        inputs = layers.Input(shape=(self.n_features,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(self.n_actions, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.n_features,))
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
            
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2)) - (entropy_coef * entropy_mean)
        
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def learn(self):
        if not self.states: return
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
        last_advantage = 0
        for t in reversed(range(len(rewards_arr))):
            advantages[t] = deltas[t] + self.gamma * 0.95 * last_advantage * (1 - dones_arr[t])
            last_advantage = advantages[t]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        self.step_counter += 1
        
        # FIXED: Safe division to prevent division-by-zero crashes
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
                    tf.convert_to_tensor(advantages[batch_indices]),
                    tf.convert_to_tensor(old_log_probs_arr[batch_indices]),
                    tf.constant(entropy_coef, dtype=tf.float32)
                )
        self.clear_memory()

# ==============================================================================
# 3. EXPERIMENT RUNNER & VISUALIZATION SUITE
# ==============================================================================
def extract_and_save_matrix(best_sequence, env, strategy, output_dir, discovery_count):
    name_to_matrix = {name: matrix for name, matrix in zip(env.gate_names, env.gates)}
    
    U_total = np.eye(env.hilbert_dim, dtype=np.complex128)
    for gate_name in best_sequence:
        gate_matrix = name_to_matrix[gate_name]
        U_total = gate_matrix @ U_total
        
    filepath = os.path.join(output_dir, f"matrix_{strategy}.txt")
    with open(filepath, "w") as f:
        f.write(f"Strategy: {strategy.upper()}\n")
        f.write(f"Sequence Discovered {discovery_count} times\n")
        f.write(f"Gates ({len(best_sequence)}): {' -> '.join(best_sequence)}\n\n")
        f.write(f"Final {env.hilbert_dim}x{env.hilbert_dim} Unitary Matrix (Real parts rounded):\n")
        f.write(np.array2string(np.real(U_total), precision=3, separator=', ', suppress_small=True))
    
    return U_total

def run_single_strategy(strategy, episodes, steps_per_episode, update_timestep, output_dir):
    print(f"\n{'='*60}\n🚀 Launching Topology: {strategy.upper()}\n{'='*60}")
    
    env = UniversalQuantumEnv(num_qudits=4, dim=3, reward_strategy=strategy)
    agent = PPOAgent(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], decay_steps=episodes)
    
    all_entropies = []
    successful_sequences = []
    timestep_counter = 0
    start_time = time.time()
    
    for episode in range(episodes):
        observation = env.reset()
        episode_max_min_ent = 0
        episode_actions = []
        
        for step in range(steps_per_episode):
            timestep_counter += 1
            action, log_prob = agent.choose_action(observation)
            episode_actions.append(env.gate_names[action])
            
            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, next_observation, done, log_prob)
            
            observation = next_observation
            episode_max_min_ent = max(episode_max_min_ent, info['actual_min_entropy'])

            if timestep_counter % update_timestep == 0:
                agent.learn()
            if done: break
                
        all_entropies.append(episode_max_min_ent)
        
        if episode_max_min_ent >= (env.max_possible_entropy * 0.99):
            successful_sequences.append((tuple(episode_actions), episode_max_min_ent))
            
        if (episode + 1) % 500 == 0:
            avg_ent = np.mean(all_entropies[-500:])
            print(f"Ep: {episode+1}/{episodes} | Metric: {strategy.upper():<8} | Avg True Min-Ent: {avg_ent:.4f}/{env.max_possible_entropy:.4f}")
            
    run_time = time.time() - start_time
    print(f"✅ Finished {strategy.upper()} in {run_time:.1f} seconds.")
    
    if successful_sequences:
        seq_counts = Counter([s[0] for s in successful_sequences])
        top_10 = seq_counts.most_common(10)
        best_seq, count = top_10[0]
        
        extract_and_save_matrix(best_seq, env, strategy, output_dir, count)
        
        labels = [("-".join(s[0]))[:25]+"..." for s in top_10]
        counts = [c for s, c in top_10]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color='teal', edgecolor='black')
        plt.title(f'Top Sequences: {strategy.upper()}')
        plt.xticks(rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.35) 
        plt.savefig(os.path.join(output_dir, f"histogram_{strategy}.png"))
        plt.close()
    
    return {
        "strategy": strategy,
        "entropies": all_entropies,
        "successful_sequences": successful_sequences,
        "run_time": run_time,
        "max_possible": env.max_possible_entropy
    }

def generate_comparative_report(results_dict, output_dir):
    print("\n\n" + "="*80)
    print(f"📊 RL TOPOLOGY COMPARATIVE DASHBOARD")
    print("="*80)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(14, 6))
    max_theor = list(results_dict.values())[0]["max_possible"]
    
    for strategy, data in results_dict.items():
        ents = data["entropies"]
        if len(ents) >= 100:
            moving_avg = np.convolve(ents, np.ones(100)/100, mode='valid')
            plt.plot(np.arange(99, len(ents)), moving_avg, label=f"{strategy.upper()} (100-ep MA)", linewidth=2)
            
    plt.axhline(y=max_theor, color='black', linestyle='--', linewidth=2, label=f'AME Limit ({max_theor:.2f})')
    plt.title('Sample Efficiency & Convergence by Reward Topology', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Actual Minimum Cut Entropy', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_learning_curves.png"), dpi=300)
    plt.close()
    
    print(f"{'Strategy':<12} | {'Success Hits':<12} | {'Avg Circuit Length':<20} | {'Late-Stage Variance':<20}")
    print("-" * 75)
    
    stability_data = {}
    
    for strategy, data in results_dict.items():
        late_stage_ents = data["entropies"][-int(len(data["entropies"])*0.1):]
        variance = np.var(late_stage_ents)
        stability_data[strategy] = variance
        
        seqs = data["successful_sequences"]
        if seqs:
            avg_length = np.mean([len(s[0]) for s in seqs])
            len_str = f"{avg_length:.1f} gates"
        else:
            len_str = "N/A"
            
        print(f"{strategy.upper():<12} | {len(seqs):<12} | {len_str:<20} | {variance:.6f}")

    plt.figure(figsize=(10, 5))
    plt.bar(stability_data.keys(), stability_data.values(), color='coral', edgecolor='black')
    plt.title('Asymptotic Policy Stability (Lower Variance = Better)', fontsize=14)
    plt.ylabel('Variance (Last 10% of Episodes)')
    plt.savefig(os.path.join(output_dir, "comparative_stability.png"), dpi=300)
    plt.close()
    
    print(f"\n✅ All comparative plots and matrices saved to directory: {output_dir}/")


if __name__ == '__main__':
    # --- HARDWARE MAXIMIZED HYPERPARAMETERS ---
    EPISODES = 20000             # Increased to ensure all 5 strategies fully converge
    STEPS_PER_EPISODE = 50       
    UPDATE_TIMESTEP = 8192       # Wide memory buffer to utilize the 16 CPUs efficiently
    
    STRATEGIES = ['min', 'sum', 'all', 'softmin', 'product']
    
    OUTPUT_DIR = f"rl_topology_ame43_{datetime.datetime.now().strftime('%H%M%S')}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    experiment_results = {}
    
    for strat in STRATEGIES:
        res = run_single_strategy(strat, EPISODES, STEPS_PER_EPISODE, UPDATE_TIMESTEP, OUTPUT_DIR)
        experiment_results[strat] = res
        
    generate_comparative_report(experiment_results, OUTPUT_DIR)