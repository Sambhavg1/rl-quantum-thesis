#!/usr/bin/env python3
"""
AME PPO — simple version based on Sambhav's working laptop code.
Clifford-only gates (X, Z, F, SUM). No fancy optimizations.
Adds: CLI args, saves plots/CSV/Dirac/circuit to disk.

Run:
    python ame_simple.py --n 4 --d 3 --episodes 10000 --seed 42 --run_name ame43_s42
    python ame_simple.py --n 3 --d 2 --episodes 5000 --seed 42 --run_name ame32_s42
    python ame_simple.py --n 5 --d 2 --episodes 15000 --seed 42 --run_name ame52_s42
    python ame_simple.py --n 6 --d 2 --episodes 20000 --seed 42 --run_name ame62_s42
"""

import argparse, os, csv, json, time, datetime, random, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from functools import reduce
from itertools import combinations
from collections import Counter

import gym
from gym import spaces


# ============================================================================
# Environment (identical to Sambhav's laptop code - Clifford gates only)
# ============================================================================
class UniversalQuantumEnv(gym.Env):
    def __init__(self, num_qudits=4, dim=3, gate_set='full', entropy='vn'):
        super().__init__()
        self.n = num_qudits
        self.d = dim
        self.gate_set_name = gate_set
        self.entropy_type = entropy
        self.hilbert_dim = self.d ** self.n

        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(2 * self.hilbert_dim,),
                                            dtype=np.float64)
        self.gates, self.gate_names = self._create_universal_gate_set()
        self.action_space = spaces.Discrete(len(self.gates))

        # Precompute max entropy for reward normalization
        k = self.n // 2
        if self.entropy_type == 'vn':
            self.max_possible_entropy = k * np.log(self.d)
        else:  # linear
            self.max_possible_entropy = 1.0 - 1.0 / (self.d ** k)

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
        reward = (entropy / self.max_possible_entropy) * 10.0

        done = entropy >= (self.max_possible_entropy * 0.999)
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
            Z_mat[i, i] = omega ** i
            for k in range(self.d):
                F_mat[i, k] = omega ** (i * k)
        F_mat /= np.sqrt(self.d)

        # Choose primitives based on gate_set
        if self.gate_set_name == 'full':
            primitives = {'X': X_mat, 'Z': Z_mat, 'F': F_mat}
        else:  # minimal Clifford generators: {F (Hadamard), Z (phase), SUM}
            primitives = {'Z': Z_mat, 'F': F_mat}

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
            if self.entropy_type == 'vn':
                evals_pos = evals[evals > 1e-10]
                ent = -np.sum(evals_pos * np.log(evals_pos)) if len(evals_pos) > 0 else 0.0
            else:  # linear entropy: 1 - Tr(rho^2) = 1 - sum(lambda_i^2)
                ent = 1.0 - float(np.sum(evals ** 2))
            entropies.append(ent)
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


# ============================================================================
# PPO Agent (identical to laptop code)
# ============================================================================
class PPOAgent:
    def __init__(self, n_actions, n_features,
                 initial_lr=1e-4, final_lr=1e-6, decay_steps=10000,
                 gamma=0.99, clip_ratio=0.1, policy_epochs=10, batch_size=256,
                 initial_entropy=0.02):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
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
        action = tf.random.categorical(tf.math.log(probs), 1).numpy().item()
        log_prob = tf.math.log(probs[0, action])
        return action, log_prob

    def choose_best_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state)
        action = tf.argmax(probs[0]).numpy().item()
        return action

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs = [], [], []

    @tf.function
    def train_step(self, states, actions, advantages, old_log_probs, entropy_coef):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values = tf.squeeze(self.critic(states, training=True))
            critic_loss = self.critic_loss_fn(advantages + values, values)

            new_probs = self.actor(states, training=True)
            action_indices = tf.stack(
                [tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            new_log_probs = tf.math.log(tf.gather_nd(new_probs, action_indices))

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
        last_advantage = 0
        for t in reversed(range(len(rewards_arr))):
            advantages[t] = deltas[t] + self.gamma * 0.95 * last_advantage * (1 - dones_arr[t])
            last_advantage = advantages[t]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        self.step_counter += 1
        entropy_coef = max(0.0, self.initial_entropy * (1 - self.step_counter / (self.decay_steps // 2048)))

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


# ============================================================================
# Plot + output functions (saved to disk, no plt.show)
# ============================================================================
def save_training_curve(all_entropies, env, out_dir):
    max_theoretical = env.max_possible_entropy
    ylabel = 'Von Neumann Entropy (nats)' if env.entropy_type == 'vn' else 'Linear Entropy (1 - Tr[rho^2])'

    plt.figure(figsize=(12, 5))
    plt.plot(all_entropies, alpha=0.5, label='Episode Max Entropy')
    if len(all_entropies) >= 100:
        moving_avg = np.convolve(all_entropies, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_entropies)), moving_avg, color='red',
                 label='Moving Avg (100)')
    plt.axhline(y=max_theoretical, color='green', linestyle='--',
                label=f'AME Limit ({max_theoretical:.4f})')
    plt.title(f'Training Progress: AME({env.n}, {env.d}) — {env.entropy_type}, {env.gate_set_name}')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curve.png'), dpi=120)
    plt.close()


def save_top_sequences(successful_sequences, env, out_dir, topk=10):
    if not successful_sequences:
        return
    seq_counts = Counter([s[0] for s in successful_sequences])
    top = seq_counts.most_common(topk)

    labels, counts, avg_entropies = [], [], []
    for seq_tuple, count in top:
        full_str = "-".join(seq_tuple)
        label = full_str[:25] + "..." if len(full_str) > 25 else full_str
        labels.append(label)
        counts.append(count)
        ents = [ent for s, ent in successful_sequences if s == seq_tuple]
        avg_entropies.append(np.max(ents) if ents else 0)

    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, counts, color='teal', edgecolor='black')
    plt.xlabel('Gate Sequence')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Successful Sequences')
    plt.xticks(rotation=90, ha='center')

    for bar, entropy in zip(bars, avg_entropies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5,
                 f"S={entropy:.2f}", ha='center', va='bottom',
                 fontsize=9, color='darkred')

    plt.subplots_adjust(bottom=0.35)
    plt.savefig(os.path.join(out_dir, 'top_sequences.png'), dpi=120)
    plt.close()


def save_entropy_climb(best_seq_tuple, env, out_dir):
    if not best_seq_tuple:
        return
    state = np.zeros(env.hilbert_dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    name_to_matrix = {name: matrix for name, matrix in zip(env.gate_names, env.gates)}

    entropies = [0.0]
    theoretical_max = env.max_possible_entropy
    ylabel = 'Von Neumann Entropy' if env.entropy_type == 'vn' else 'Linear Entropy'

    step_log = []
    step_log.append(f"{'Step':<5} | {'Gate':<15} | {'Entropy':<10} | {'% of Max':<10}")
    step_log.append("-" * 55)
    step_log.append(f"{0:<5} | {'(Initial)':<15} | {0.0:<10.4f} | {0.0:<10.1f}%")

    for step, gate_name in enumerate(best_seq_tuple):
        matrix = name_to_matrix[gate_name]
        state = matrix @ state
        state /= np.linalg.norm(state)

        ent = env._calculate_ame_metric(state)
        entropies.append(ent)
        pct = (ent / theoretical_max) * 100
        step_log.append(f"{step+1:<5} | {gate_name:<15} | {ent:<10.4f} | {pct:<10.1f}%")

    with open(os.path.join(out_dir, 'entropy_climb_steps.txt'), 'w') as f:
        f.write(f"Step-by-step entropy evolution for best sequence\n")
        f.write(f"Gates: {' -> '.join(best_seq_tuple)}\n\n")
        f.write('\n'.join(step_log) + '\n')

    plt.figure(figsize=(10, 4))
    plt.plot(entropies, marker='o', linestyle='-', color='indigo', label='Achieved Entropy')
    plt.axhline(y=theoretical_max, color='r', linestyle='--',
                label=f'AME Limit ({theoretical_max:.4f})')
    plt.title(f'Entropy Climb (Step-by-Step) — {env.entropy_type}, {env.gate_set_name}')
    plt.xlabel('Gate Step')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'entropy_climb.png'), dpi=120)
    plt.close()

    return state  # Return final state for Dirac


def save_dirac(state, env, out_dir, threshold=1e-3):
    lines = []
    for i, amp in enumerate(state):
        if np.abs(amp) ** 2 > threshold:
            indices = []
            temp_i = i
            for _ in range(env.n):
                indices.append(str(temp_i % env.d))
                temp_i //= env.d
            basis_ket = "".join(reversed(indices))
            lines.append(f"{np.abs(amp):.4f}|{basis_ket}>")
    dirac = " + ".join(lines) if lines else "(no significant components)"
    with open(os.path.join(out_dir, 'best_state_dirac.txt'), 'w') as f:
        f.write(f"AME({env.n}, {env.d}) - best state\n")
        f.write(f"|psi> = {dirac}\n")


def try_save_circuit(action_names, env, out_dir):
    if env.d != 2:
        with open(os.path.join(out_dir, 'circuit.txt'), 'w') as f:
            f.write(f"Qudit circuit (d={env.d}, n={env.n}):\n\n")
            for i, g in enumerate(action_names):
                f.write(f"  step {i+1:2d}: {g}\n")
        return
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(env.n)
        for name in action_names:
            parts = name.split('_')
            head = parts[0]
            if head == 'X':
                qc.x(int(parts[1]))
            elif head == 'Z':
                qc.z(int(parts[1]))
            elif head == 'F':
                qc.h(int(parts[1]))
            elif head == 'SUM':
                qc.cx(int(parts[1]), int(parts[2]))
        fig = qc.draw(output='mpl', fold=60)
        fig.savefig(os.path.join(out_dir, 'circuit.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] circuit diagram failed: {e}")
        with open(os.path.join(out_dir, 'circuit.txt'), 'w') as f:
            for i, g in enumerate(action_names):
                f.write(f"  step {i+1:2d}: {g}\n")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--steps_per_episode', type=int, default=50)
    parser.add_argument('--update_timestep', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gate_set', type=str, default='full',
                        choices=['full', 'minimal'],
                        help='full = X,Z,F,SUM; minimal = F,Z,SUM (drops X)')
    parser.add_argument('--entropy', type=str, default='vn',
                        choices=['vn', 'linear'],
                        help='vn = Von Neumann (-sum p log p); linear = 1 - Tr(rho^2)')
    parser.add_argument('--output_root', type=str, default='./results')
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()

    # Seeds
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Output dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ame_{args.n}_{args.d}_s{args.seed}_{ts}"
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"[RUN] {run_name} -> {out_dir}")

    # Env + Agent
    env = UniversalQuantumEnv(num_qudits=args.n, dim=args.d,
                              gate_set=args.gate_set, entropy=args.entropy)
    max_possible = env.max_possible_entropy
    print(f"[ENV] n={args.n} d={args.d} | gate_set={args.gate_set} | "
          f"entropy={args.entropy} | actions={env.action_space.n} | "
          f"E_max={max_possible:.4f}")

    agent = PPOAgent(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        initial_lr=1e-4,
        final_lr=1e-6,
        decay_steps=args.episodes,
        gamma=0.99,
        clip_ratio=0.1,
        policy_epochs=10,
        batch_size=256,
        initial_entropy=0.02
    )

    print(f"--- Starting AME({args.n}, {args.d}) Search ---")

    all_entropies = []
    successful_sequences = []
    timestep_counter = 0

    global_max_entropy = 0.0
    start_time = time.time()

    # Per-episode CSV
    csv_path = os.path.join(out_dir, 'ame_results.csv')
    f_csv = open(csv_path, 'w', newline='', buffering=1)
    writer = csv.writer(f_csv)
    writer.writerow(['episode', 'max_entropy', 'length', 'sequence'])

    for episode in range(args.episodes):
        observation = env.reset()
        episode_max_entropy = 0
        episode_actions = []

        for step in range(args.steps_per_episode):
            timestep_counter += 1
            action, log_prob = agent.choose_action(observation)
            gate_name = env.gate_names[action]
            episode_actions.append(gate_name)

            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward,
                                   next_observation, done, log_prob)
            observation = next_observation
            episode_max_entropy = max(episode_max_entropy, info['entropy'])

            if timestep_counter % args.update_timestep == 0:
                agent.learn()
            if done:
                break

        all_entropies.append(episode_max_entropy)
        writer.writerow([episode + 1, episode_max_entropy,
                         len(episode_actions), '-'.join(episode_actions)])

        if episode_max_entropy > global_max_entropy:
            global_max_entropy = episode_max_entropy
            if global_max_entropy > max_possible * 0.95:
                print(f"!!! NEW RECORD: {global_max_entropy:.4f} (Episode {episode+1}) !!!")

        if episode_max_entropy > (max_possible * 0.99):
            successful_sequences.append((tuple(episode_actions), episode_max_entropy))

        if (episode + 1) % 100 == 0:
            avg_ent = np.mean(all_entropies[-100:])
            elapsed = time.time() - start_time
            try:
                curr_lr = agent.lr_schedule(agent.actor_optimizer.iterations).numpy()
            except Exception:
                curr_lr = float('nan')
            print(f'Episode: {episode+1}/{args.episodes} | Avg Max Ent: {avg_ent:.4f} | '
                  f'LR: {curr_lr:.2e} | Time: {elapsed:.0f}s')

    f_csv.close()
    wall_time = time.time() - start_time
    print(f"\n=== Done === episodes={args.episodes} wall_time={wall_time:.1f}s")

    # Deterministic final exam
    print("\n--- Running Deterministic Final Exam ---")
    observation = env.reset()
    exam_actions = []
    for _ in range(args.steps_per_episode):
        action = agent.choose_best_action(observation)
        observation, _, done, info = env.step(action)
        exam_actions.append(env.gate_names[action])
        if done:
            break

    final_entropy = info['entropy']
    final_state = env.state.copy()
    print(f"Final Exam Entropy: {final_entropy:.4f}")
    converged = final_entropy > 0.99 * max_possible
    if converged:
        print(f"CONVERGENCE CONFIRMED: Agent reproduces AME({args.n},{args.d}).")
    else:
        print("Agent still exploring.")

    # Save final exam
    with open(os.path.join(out_dir, 'final_exam.txt'), 'w') as f:
        f.write(f"Deterministic Final Exam\n")
        f.write(f"Entropy: {final_entropy:.6f} / {max_possible:.6f}\n")
        f.write(f"Ratio: {final_entropy/max_possible:.6f}\n")
        f.write(f"Converged: {converged}\n\n")
        f.write(f"Gate sequence ({len(exam_actions)} gates):\n")
        f.write(' -> '.join(exam_actions) + '\n')

    # All plots
    save_training_curve(all_entropies, env, out_dir)
    save_top_sequences(successful_sequences, env, out_dir)

    # Best sequence analysis (entropy climb + Dirac + circuit)
    if successful_sequences:
        seq_counts = Counter([s[0] for s in successful_sequences])
        best_seq_tuple, count = seq_counts.most_common(1)[0]

        final_state_from_best = save_entropy_climb(best_seq_tuple, env, out_dir)
        if final_state_from_best is not None:
            save_dirac(final_state_from_best, env, out_dir)
        try_save_circuit(list(best_seq_tuple), env, out_dir)

        with open(os.path.join(out_dir, 'best_sequence.txt'), 'w') as f:
            f.write(f"Best gate sequence (found {count} times):\n\n")
            f.write(' -> '.join(best_seq_tuple) + '\n')
    else:
        # Use the final exam state at least
        save_dirac(final_state, env, out_dir)
        try_save_circuit(exam_actions, env, out_dir)

    # Save models
    agent.actor.save(os.path.join(out_dir, 'actor_final.keras'))
    agent.critic.save(os.path.join(out_dir, 'critic_final.keras'))

    # CSV with just successful
    success_csv = os.path.join(out_dir, 'successful_sequences.csv')
    successful_sequences_sorted = sorted(successful_sequences, key=lambda x: x[1], reverse=True)
    with open(success_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Entropy', 'Length', 'Sequence'])
        for seq_tuple, ent in successful_sequences_sorted:
            w.writerow([f"{ent:.6f}", len(seq_tuple), '-'.join(seq_tuple)])

    # Summary
    summary = {
        'run_name': run_name,
        'n': args.n, 'd': args.d,
        'gate_set': args.gate_set,
        'entropy_type': args.entropy,
        'num_actions': env.action_space.n,
        'episodes_run': len(all_entropies),
        'wall_time_s': wall_time,
        'max_possible_entropy': float(max_possible),
        'global_best_entropy': float(global_max_entropy),
        'global_best_ratio': float(global_max_entropy / max_possible),
        'final_exam_entropy': float(final_entropy),
        'final_exam_ratio': float(final_entropy / max_possible),
        'converged': bool(converged),
        'num_successful_sequences': len(successful_sequences),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[WROTE] {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
