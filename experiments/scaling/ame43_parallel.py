import os
import csv
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from functools import reduce
from itertools import combinations
from collections import Counter
import multiprocessing as mp

# --- SERVER CONFIGURATION ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check GPU Visibility
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s) active.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ WARNING: Running on CPU. GPU not detected.")

# Create Output Directories
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_parallel_ame43_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. Universal Quantum Environment (AME 4,3)
# ==============================================================================
import gym
from gym import spaces

class UniversalQuantumEnv(gym.Env):
    def __init__(self, num_qudits=4, dim=3):
        super().__init__()
        self.n = num_qudits
        self.d = dim
        self.hilbert_dim = self.d ** self.n

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.hilbert_dim,),
            dtype=np.float64
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

        reward = (entropy / max_possible_entropy) * 10.0
        done = entropy >= (max_possible_entropy * 0.999)
        if done:
            reward += 100.0
        reward -= 0.01

        return self._get_obs(), reward, done, {"entropy": entropy}

    def _get_obs(self):
        return np.concatenate([self.state.real, self.state.imag])

    def _create_universal_gate_set(self):
        gates = []
        names = []
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
# 2. PPO Agent (same as Code 1, plus batched action sampling)
# ==============================================================================
class PPOAgent:
    def __init__(self, n_actions, n_features,
                 initial_lr=1e-4, final_lr=1e-6, decay_steps=10000,
                 gamma=0.99, clip_ratio=0.1, policy_epochs=10,
                 batch_size=2048, initial_entropy=0.02):

        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size

        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=final_lr,
            power=1.0
        )

        self.initial_entropy = initial_entropy
        self.decay_steps = decay_steps
        self.step_counter = 0

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

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def choose_action_batch(self, states_np):
        """Vectorized action sampling for N envs at once."""
        states = tf.convert_to_tensor(states_np, dtype=tf.float32)  # [N, obs]
        probs = self.actor(states)                                 # [N, A]
        actions = tf.random.categorical(tf.math.log(probs + 1e-10), 1)[:, 0]  # [N]
        batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_idx = tf.stack([batch_idx, actions], axis=1)
        log_probs = tf.math.log(tf.gather_nd(probs, gather_idx) + 1e-10)      # [N]
        return actions.numpy().astype(np.int32), log_probs.numpy().astype(np.float32)

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

        self.step_counter += 1
        # same style as your code1 (kept)
        entropy_coef = max(0.0, self.initial_entropy * (1 - self.step_counter / max(1, (self.decay_steps // 2048))))

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
# 3. Multiprocessing Env Workers
# ==============================================================================
def _env_worker(conn, num_qudits, dim):
    env = UniversalQuantumEnv(num_qudits=num_qudits, dim=dim)
    while True:
        msg = conn.recv()
        if msg[0] == "reset":
            obs = env.reset()
            conn.send(obs)
        elif msg[0] == "step":
            action = msg[1]
            obs, reward, done, info = env.step(action)
            if done:
                # don’t auto-reset here; main loop controls episode bookkeeping
                pass
            conn.send((obs, reward, done, info))
        elif msg[0] == "close":
            conn.close()
            break
        else:
            raise RuntimeError(f"Unknown message: {msg[0]}")

class ParallelEnvs:
    def __init__(self, n_envs, num_qudits, dim):
        self.n_envs = n_envs
        self.parents, self.children = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.procs = []
        for child_conn in self.children:
            p = mp.Process(target=_env_worker, args=(child_conn, num_qudits, dim), daemon=True)
            p.start()
            self.procs.append(p)

    def reset_all(self):
        for conn in self.parents:
            conn.send(("reset",))
        obs = [conn.recv() for conn in self.parents]
        return np.stack(obs, axis=0)

    def step_all(self, actions):
        for conn, a in zip(self.parents, actions):
            conn.send(("step", int(a)))
        results = [conn.recv() for conn in self.parents]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs, axis=0), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool_), list(infos)

    def reset_one(self, idx):
        self.parents[idx].send(("reset",))
        return self.parents[idx].recv()

    def close(self):
        for conn in self.parents:
            conn.send(("close",))
        for p in self.procs:
            p.join(timeout=1)

# ==============================================================================
# 4. Visualization (same as Code 1)
# ==============================================================================
def save_plots(all_entropies, successful_sequences, env_n, env_d):
    k = env_n // 2
    max_theoretical = k * np.log(env_d)

    plt.figure(figsize=(12, 5))
    plt.plot(all_entropies, alpha=0.5, label='Episode Max Entropy')
    if len(all_entropies) >= 100:
        moving_avg = np.convolve(all_entropies, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_entropies)), moving_avg, color='red', label='Moving Avg (100)')
    plt.axhline(y=max_theoretical, color='green', linestyle='--', label=f'AME Limit ({max_theoretical:.2f})')
    plt.title(f'Training Progress: AME Search (N={env_n}, D={env_d})')
    plt.xlabel('Episode')
    plt.ylabel('Von Neumann Entropy (Nats)')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training plot to {plot_path}")

    if successful_sequences:
        seq_counts = Counter([s[0] for s in successful_sequences])
        top_10 = seq_counts.most_common(10)

        labels = []
        counts = []
        for seq_tuple, count in top_10:
            full_str = "-".join(seq_tuple)
            label = full_str[:25] + "..." if len(full_str) > 25 else full_str
            labels.append(label)
            counts.append(count)

        plt.figure(figsize=(12, 8))
        plt.bar(labels, counts, edgecolor='black')
        plt.xlabel('Gate Sequence')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Frequent Successful Sequences')
        plt.xticks(rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.35)

        hist_path = os.path.join(OUTPUT_DIR, "sequence_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram to {hist_path}")

# ==============================================================================
# 5. Main Execution (Parallel Rollouts)
# ==============================================================================
if __name__ == '__main__':
    # IMPORTANT: spawn is safest with TF + multiprocessing
    mp.set_start_method("spawn", force=True)

    NUM_QUDITS = 4
    DIM = 3
    EPISODES = 10000
    STEPS_PER_EPISODE = 50
    UPDATE_TIMESTEP = 8192

    # Parallel envs (tune this)
    N_ENVS = 8  # try 8 first; if you request 16 CPUs, try 16 envs

    # Create one local env just to access gate_names for logging/CSV
    env_ref = UniversalQuantumEnv(num_qudits=NUM_QUDITS, dim=DIM)

    agent = PPOAgent(
        n_actions=env_ref.action_space.n,
        n_features=env_ref.observation_space.shape[0],
        initial_lr=1e-4,
        final_lr=1e-6,
        decay_steps=EPISODES,
        gamma=0.99,
        clip_ratio=0.1,
        policy_epochs=10,
        batch_size=2048,
        initial_entropy=0.02
    )

    penv = ParallelEnvs(n_envs=N_ENVS, num_qudits=NUM_QUDITS, dim=DIM)

    print(f"--- Starting PARALLEL AME({NUM_QUDITS},{DIM}) with {N_ENVS} envs ---")

    # Per-env episode bookkeeping
    obs_batch = penv.reset_all()
    env_episode_steps = np.zeros(N_ENVS, dtype=np.int32)
    env_episode_max_entropy = np.zeros(N_ENVS, dtype=np.float64)
    env_episode_actions = [[] for _ in range(N_ENVS)]

    all_entropies = []
    successful_sequences = []
    timestep_counter = 0

    k_half = NUM_QUDITS // 2
    max_possible = k_half * np.log(DIM)
    global_max_entropy = 0.0

    start_time = time.time()
    completed_episodes = 0

    try:
        while completed_episodes < EPISODES:
            # Choose actions for all envs in one TF forward pass
            actions, logps = agent.choose_action_batch(obs_batch)

            next_obs_batch, rewards, dones, infos = penv.step_all(actions)

            # Store transitions + update per-env episode stats
            for i in range(N_ENVS):
                agent.store_transition(
                    obs_batch[i], actions[i], rewards[i],
                    next_obs_batch[i], float(dones[i]), logps[i]
                )

                timestep_counter += 1
                env_episode_steps[i] += 1
                env_episode_actions[i].append(env_ref.gate_names[actions[i]])
                env_episode_max_entropy[i] = max(env_episode_max_entropy[i], infos[i]["entropy"])

                # PPO update
                if timestep_counter % UPDATE_TIMESTEP == 0:
                    print(f"[Parallel Update] Training at step {timestep_counter} ...")
                    agent.learn()

                # End episode either by env done or max steps reached
                if dones[i] or env_episode_steps[i] >= STEPS_PER_EPISODE:
                    ep_max = float(env_episode_max_entropy[i])
                    all_entropies.append(ep_max)

                    if ep_max > global_max_entropy:
                        global_max_entropy = ep_max
                        if global_max_entropy > 2.1:
                            print(f"!!! NEW RECORD: {global_max_entropy:.4f} (Episode {completed_episodes+1}) !!!")

                    if ep_max > (max_possible * 0.99):
                        successful_sequences.append((tuple(env_episode_actions[i]), ep_max))

                    completed_episodes += 1

                    # Reset this env slot
                    obs_i = penv.reset_one(i)
                    next_obs_batch[i] = obs_i
                    env_episode_steps[i] = 0
                    env_episode_max_entropy[i] = 0.0
                    env_episode_actions[i] = []

                    if completed_episodes % 100 == 0:
                        avg_ent = float(np.mean(all_entropies[-100:])) if len(all_entropies) >= 100 else float(np.mean(all_entropies))
                        curr_lr = float(agent.lr_schedule(agent.actor_optimizer.iterations).numpy())
                        elapsed = time.time() - start_time
                        print(f'Episode: {completed_episodes}/{EPISODES} | Avg Max Ent: {avg_ent:.4f} | LR: {curr_lr:.2e} | Time: {elapsed:.1f}s')

            obs_batch = next_obs_batch

    finally:
        penv.close()

    # --- SAVE RESULTS ---
    save_plots(all_entropies, successful_sequences, NUM_QUDITS, DIM)

    csv_path = os.path.join(OUTPUT_DIR, "ame43_results.csv")
    print(f"\nSaving {len(successful_sequences)} successful sequences to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode_Entropy', 'Sequence_Length', 'Gate_Sequence'])
        for seq_tuple, ent in successful_sequences:
            writer.writerow([f"{ent:.6f}", len(seq_tuple), "-".join(seq_tuple)])
    print("Save successful.")
