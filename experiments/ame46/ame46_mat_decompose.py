#!/usr/bin/env python3
import os
import csv
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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
OUTPUT_DIR = f"results_ame_hybrid_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. THE TARGET MATRIX GENERATOR (From Rather et al.)
# ==============================================================================
def generate_AME46_target():
    """Generates the exact 36x36 Golden AME(4,6) unitary matrix."""
    a = 1 / np.sqrt(5.0 + np.sqrt(5.0))
    b = np.sqrt(5.0 + np.sqrt(5.0)) / np.sqrt(20.0)
    c = 1 / np.sqrt(2.0)
    w = np.exp(1j * np.pi / 10.0)
    U = np.zeros((36, 36), dtype=np.complex128)

    # 20 non-zero entries of 1st six rows:
    U[1, 1] = c*(w**3); U[5, 0] = c/w; U[0, 7] = b/w
    U[2, 6] = a/(w**7); U[3, 6] = b*(w**2); U[5, 7] = a/(w**3)
    U[0, 15] = a/(w**2); U[1, 14] = b*(w**9); U[4, 14] = a*(w**5)
    U[5, 15] = b*(w**6); U[2, 20] = a/(w**6); U[3, 20] = b/w
    U[4, 21] = c/w; U[1, 29] = a; U[2, 28] = b*(w**7)
    U[3, 28] = a*(w**2); U[4, 29] = b*(w**6); U[0, 34] = c*(w**6)
    U[2, 35] = b/(w**5); U[3, 35] = a/(w**6)

    # 20 non-zero entries of 2nd six rows:
    U[6, 0] = c; U[10, 1] = c/(w**4); U[6, 7] = a*(w**8)
    U[8, 6] = b/(w**8); U[9, 6] = a/(w**9); U[11, 7] = b/(w**4)
    U[6, 15] = b/(w**3); U[7, 14] = a*(w**6); U[10, 14] = b/(w**8)
    U[11, 15] = a/(w**5); U[8, 20] = b*w; U[9, 20] = a/(w**4)
    U[7, 21] = c*(w**10); U[8, 28] = a/(w**4); U[9, 28] = b*w
    U[7, 29] = b*(w**7); U[10, 29] = a*(w**3); U[8, 35] = a/(w**8)
    U[9, 35] = b*w; U[11, 34] = c/(w**7)

    # 18 non-zero entries of 3rd six rows:
    U[13, 5] = b/(w**3); U[14, 4] = b/(w**4); U[15, 4] = a/w
    U[16, 5] = a*(w**9); U[12, 11] = a*(w**4); U[13, 10] = a*(w**6)
    U[16, 10] = b*(w**8); U[17, 11] = b; U[15, 13] = c/(w**5)
    U[16, 12] = c*w; U[14, 18] = c; U[17, 19] = c/(w**6)
    U[12, 26] = c*(w**8); U[14, 27] = a*(w**6); U[15, 27] = b/w
    U[12, 33] = b/(w**6); U[13, 32] = c/(w**6); U[17, 33] = a

    # 18 non-zero entries of 4th six rows:
    U[19, 5] = a/(w**3); U[20, 4] = a/(w**8); U[21, 4] = b*(w**5)
    U[22, 5] = b/w; U[18, 11] = b*(w**2); U[19, 10] = b/(w**4)
    U[22, 10] = a*(w**8); U[23, 11] = a*(w**8); U[19, 12] = c/w
    U[20, 13] = c/(w**2); U[18, 19] = c*(w**6); U[21, 18] = c/w
    U[23, 26] = c*(w**2); U[20, 27] = b/(w**8); U[21, 27] = a/(w**5)
    U[18, 33] = a*(w**2); U[22, 32] = c*(w**6); U[23, 33] = b/(w**2)

    # 18 non-zero entries of 5th six rows:
    U[24, 2] = a; U[26, 3] = a*w; U[27, 3] = b*(w**2)
    U[29, 2] = b; U[25, 9] = b*w; U[26, 8] = b*w**3
    U[27, 8] = a/(w**6); U[28, 9] = a*w; U[24, 16] = b/(w**4)
    U[26, 17] = c/(w**5); U[29, 16] = a*(w**6); U[24, 23] = c/(w**7)
    U[25, 22] = a/(w**2); U[28, 22] = b*(w**8); U[25, 25] = c/(w**6)
    U[29, 24] = c; U[27, 31] = c/(w**2); U[28, 30] = c/(w**6)

    # 18 non-zero entries of last six rows:
    U[30, 2] = b/(w**9); U[32, 3] = b; U[33, 3] = a/(w**9)
    U[35, 2] = a*w; U[31, 9] = a/(w**6); U[32, 8] = a/(w**8)
    U[33, 8] = b/(w**7); U[34, 9] = b*(w**4); U[30, 16] = a/(w**3)
    U[33, 17] = c/(w**5); U[35, 16] = b/(w**3); U[31, 22] = b*w
    U[34, 22] = a*w; U[35, 23] = c*(w**4); U[30, 24] = c*w
    U[34, 25] = c*(w**7); U[31, 30] = c/(w**3); U[32, 31] = c*(w**6)

    return U

# ==============================================================================
# 2. THE PHYSICS ENGINE (Golden AME Toolkit)
# ==============================================================================
class GoldenAMEBuilder:
    def __init__(self, dim=6):
        self.d = dim
        self.hilbert_dim = self.d ** 2 
        self.omega_20 = np.exp(1j * np.pi / 10) 
        self.omega_6 = np.exp(2j * np.pi / self.d) 

    def get_X_gate(self):
        X = np.zeros((self.d, self.d), dtype=np.complex128)
        for i in range(self.d):
            X[(i + 1) % self.d, i] = 1.0
        return X

    def get_Z_gate(self):
        Z = np.zeros((self.d, self.d), dtype=np.complex128)
        for i in range(self.d):
            Z[i, i] = self.omega_6 ** i
        return Z

    def get_F_gate(self):
        F = np.zeros((self.d, self.d), dtype=np.complex128)
        for i in range(self.d):
            for j in range(self.d):
                F[i, j] = (self.omega_6 ** (i * j)) / np.sqrt(self.d)
        return F

    def get_SUM_gate(self, control_idx, target_idx):
        SUM = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=np.complex128)
        for basis_idx in range(self.hilbert_dim):
            digits = [basis_idx // self.d, basis_idx % self.d]
            c_val = digits[control_idx]
            t_val = digits[target_idx]
            digits[target_idx] = (t_val + c_val) % self.d
            new_idx = digits[0] * self.d + digits[1]
            SUM[new_idx, basis_idx] = 1.0
        return SUM

    def get_Phase_gate(self, k_multiplier):
        P = np.eye(self.d, dtype=np.complex128)
        for level in range(self.d):
            P[level, level] = self.omega_20 ** (k_multiplier * level)
        return P

    def get_Givens_Rotation(self, m, n, phi):
        R = np.eye(self.d, dtype=np.complex128)
        R[m, m] = np.cos(phi)
        R[m, n] = -np.sin(phi)
        R[n, m] = np.sin(phi)
        R[n, n] = np.cos(phi)
        return R

    def expand_to_36(self, single_qudit_matrix, target_qudit):
        eye_6 = np.eye(self.d, dtype=np.complex128)
        if target_qudit == 0:
            return np.kron(single_qudit_matrix, eye_6)
        else:
            return np.kron(eye_6, single_qudit_matrix)

# ==============================================================================
# 3. HYBRID GYM ENVIRONMENT
# ==============================================================================
class HybridUnitaryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.builder = GoldenAMEBuilder(dim=6)
        self.hilbert_dim = 36
        
        self.obs_dim = self.hilbert_dim * self.hilbert_dim * 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float64)
        
        self.gate_dictionary, self.gate_names = self._build_action_dictionary()
        self.num_discrete_actions = len(self.gate_dictionary)
        
        self.target_U = generate_AME46_target()
        assert self.target_U.shape == (36, 36)
        
        self.current_U = None
        self.previous_fidelity = 0.0

    def _build_action_dictionary(self):
        gates, names = [], []
        
        for q in [0, 1]:
            for gate_func, name in [(self.builder.get_X_gate, "X"), 
                                    (self.builder.get_Z_gate, "Z"), 
                                    (self.builder.get_F_gate, "F")]:
                full_mat = self.builder.expand_to_36(gate_func(), q)
                gates.append((full_mat, q, 'discrete'))
                names.append(f"{name}_{q}")
                
        gates.append((self.builder.get_SUM_gate(0, 1), 'both', 'discrete'))
        names.append("SUM_0_1")
        gates.append((self.builder.get_SUM_gate(1, 0), 'both', 'discrete')) 
        names.append("SUM_1_0")
        
        for q in [0, 1]:
            for k in range(20):
                full_mat = self.builder.expand_to_36(self.builder.get_Phase_gate(k), q)
                gates.append((full_mat, q, 'discrete'))
                names.append(f"Phase_{q}_k{k}")
                
        for q in [0, 1]:
            for level in range(5):
                gates.append(((level, level+1), q, 'continuous'))
                names.append(f"Givens_{q}_{level}_{level+1}")
                
        return gates, names

    def reset(self):
        self.current_U = np.eye(self.hilbert_dim, dtype=np.complex128)
        self.previous_fidelity = self._calc_fidelity()
        return self._get_obs()

    def step(self, discrete_action, continuous_phi):
        gate_info = self.gate_dictionary[discrete_action]
        action_name = self.gate_names[discrete_action]
        is_continuous_used = 0.0 # <--- NEW: Flag to track if phi was used
        
        if gate_info[2] == 'discrete':
            matrix = gate_info[0] 
            self.current_U = matrix @ self.current_U
        else:
            is_continuous_used = 1.0 # <--- NEW: Flag updated
            m, n = gate_info[0]
            target_q = gate_info[1]
            base_matrix_6x6 = self.builder.get_Givens_Rotation(m, n, continuous_phi)
            
            U_tensor = self.current_U.reshape(6, 6, 36)
            if target_q == 0:
                out = np.tensordot(base_matrix_6x6, U_tensor, axes=([1], [0]))
            else:
                out = np.tensordot(base_matrix_6x6, U_tensor, axes=([1], [1]))
                out = np.moveaxis(out, 0, 1)
            
            self.current_U = out.reshape(36, 36)
            action_name += f"({continuous_phi:.2f}rad)"

        current_fidelity = self._calc_fidelity()
        delta = current_fidelity - self.previous_fidelity
        reward = delta * 100.0 
        
        self.previous_fidelity = current_fidelity
        done = current_fidelity >= 0.999
        if done: reward += 100.0
        
        # Note: We return is_continuous_used in the info dict so the agent can mask gradients
        return self._get_obs(), reward, done, {"fidelity": current_fidelity, "gate": action_name, "cont_used": is_continuous_used}

    def _calc_fidelity(self):
        trace_val = np.trace(self.target_U.conj().T @ self.current_U)
        return (np.abs(trace_val) ** 2) / (self.hilbert_dim ** 2)

    def _get_obs(self):
        return np.concatenate([self.current_U.real.flatten(), self.current_U.imag.flatten()])

# ==============================================================================
# 4. MULTI-HEADED HYBRID PPO AGENT
# ==============================================================================
class HybridPPOAgent:
    def __init__(self, num_discrete, obs_dim, initial_lr=1e-4, decay_steps=100000):
        self.num_discrete = num_discrete
        self.obs_dim = obs_dim
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_epochs = 10
        self.batch_size = 64
        self.cont_std = 1.0 
        
        # <--- BUG 2 FIX: Actually define the learning rate schedule --->
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr, decay_steps=decay_steps,
            end_learning_rate=1e-6, power=1.0)
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.clear_memory()

    def _build_actor(self):
        inputs = layers.Input(shape=(self.obs_dim,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        
        discrete_probs = layers.Dense(self.num_discrete, activation='softmax', name='discrete_head')(x)
        cont_mean_raw = layers.Dense(1, activation='tanh')(x)
        cont_mean = layers.Lambda(lambda tensor: tensor * np.pi, name='continuous_head')(cont_mean_raw)
        
        return tf.keras.Model(inputs, [discrete_probs, cont_mean])

    def _build_critic(self):
        inputs = layers.Input(shape=(self.obs_dim,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        value = layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inputs, value)

    def choose_action(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        discrete_probs, cont_mean = self.actor(state_tensor)
        
        discrete_action = tf.random.categorical(tf.math.log(discrete_probs + 1e-10), 1).numpy().item()
        discrete_log_prob = tf.math.log(discrete_probs[0, discrete_action] + 1e-10).numpy()
        
        mean_val = cont_mean[0, 0].numpy()
        cont_action = np.random.normal(mean_val, self.cont_std)
        cont_action = np.clip(cont_action, -np.pi, np.pi) 
        
        var = self.cont_std ** 2
        cont_log_prob = -0.5 * (((cont_action - mean_val) ** 2) / var + np.log(2 * np.pi * var))
        
        return discrete_action, cont_action, discrete_log_prob, cont_log_prob

    # <--- BUG 3 FIX: Memory now stores separate log probs and continuous usage masks --->
    def store_transition(self, state, d_act, c_act, reward, next_state, done, d_log_prob, c_log_prob, cont_used):
        self.states.append(state)
        self.d_actions.append(d_act)
        self.c_actions.append(c_act)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.d_log_probs.append(d_log_prob)
        self.c_log_probs.append(c_log_prob)
        self.cont_masks.append(cont_used)

    def clear_memory(self):
        self.states, self.d_actions, self.c_actions = [], [], []
        self.rewards, self.next_states, self.dones = [], [], []
        self.d_log_probs, self.c_log_probs, self.cont_masks = [], [], []

    @tf.function
    def train_step(self, states, d_actions, c_actions, returns, advantages, old_d_lp, old_c_lp, cont_masks, current_std):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values = tf.squeeze(self.critic(states, training=True))
            
            # <--- BUG 1 FIX: Critic Loss properly formulated using Returns --->
            critic_loss = tf.reduce_mean((returns - values)**2) 
            
            discrete_probs, cont_means = self.actor(states, training=True)
            indices = tf.stack([tf.range(tf.shape(d_actions)[0]), d_actions], axis=1)
            new_d_log_probs = tf.math.log(tf.gather_nd(discrete_probs, indices) + 1e-10)
            
            c_actions_tensor = tf.expand_dims(c_actions, 1)
            var = current_std ** 2
            new_c_log_probs = -0.5 * (((c_actions_tensor - cont_means) ** 2) / var + tf.math.log(2 * np.pi * var))
            new_c_log_probs = tf.squeeze(new_c_log_probs)
            
            # <--- BUG 3 FIX: Apply mask to continuous log probs so non-continuous actions aren't polluted --->
            old_total_lp = old_d_lp + (old_c_lp * cont_masks)
            new_total_lp = new_d_log_probs + (new_c_log_probs * cont_masks)
            
            ratio = tf.exp(new_total_lp - old_total_lp)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def learn(self):
        if not self.states: return
        
        states_arr = np.array(self.states, dtype=np.float32)
        d_act_arr = np.array(self.d_actions, dtype=np.int32)
        c_act_arr = np.array(self.c_actions, dtype=np.float32)
        rewards_arr = np.array(self.rewards, dtype=np.float32)
        next_states_arr = np.array(self.next_states, dtype=np.float32)
        dones_arr = np.array(self.dones, dtype=np.float32)
        old_d_lp_arr = np.array(self.d_log_probs, dtype=np.float32)
        old_c_lp_arr = np.array(self.c_log_probs, dtype=np.float32)
        cont_masks_arr = np.array(self.cont_masks, dtype=np.float32)

        values = self.critic(states_arr).numpy().flatten()
        next_values = self.critic(next_states_arr).numpy().flatten()
        deltas = rewards_arr + self.gamma * next_values * (1 - dones_arr) - values
        
        advantages = np.zeros_like(rewards_arr)
        last_adv = 0
        for t in reversed(range(len(rewards_arr))):
            advantages[t] = deltas[t] + self.gamma * 0.95 * last_adv * (1 - dones_arr[t])
            last_adv = advantages[t]
            
        # Target returns for the Critic to actually learn
        returns_arr = advantages + values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        num_samples = len(self.states)
        indices = np.arange(num_samples)
        for _ in range(self.policy_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                self.train_step(
                    tf.convert_to_tensor(states_arr[batch_idx]),
                    tf.convert_to_tensor(d_act_arr[batch_idx]),
                    tf.convert_to_tensor(c_act_arr[batch_idx]),
                    tf.convert_to_tensor(returns_arr[batch_idx]),
                    tf.convert_to_tensor(advantages[batch_idx]),
                    tf.convert_to_tensor(old_d_lp_arr[batch_idx]),
                    tf.convert_to_tensor(old_c_lp_arr[batch_idx]),
                    tf.convert_to_tensor(cont_masks_arr[batch_idx]),
                    tf.constant(self.cont_std, dtype=tf.float32)
                )
        self.clear_memory()
        self.cont_std = max(0.05, self.cont_std * 0.995)

# ==============================================================================
# 5. SERVER SAFE SAVING UTILS
# ==============================================================================
def save_results(all_fidelities, best_sequences):
    plt.figure(figsize=(12, 5))
    plt.plot(all_fidelities, alpha=0.3, color='blue', label='Episode Max Fidelity')
    if len(all_fidelities) >= 100:
        ma = np.convolve(all_fidelities, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_fidelities)), ma, color='red', label='100-Ep Moving Avg')
    plt.axhline(y=1.0, color='green', linestyle='--', label='Perfect Reconstruction (1.0)')
    plt.title('Hybrid PPO Matrix Decomposition Progress: AME(4,6)')
    plt.xlabel('Episode')
    plt.ylabel('Hilbert-Schmidt Fidelity')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, "hybrid_training_curve.png")
    plt.savefig(plot_path)
    plt.close()
    
    csv_path = os.path.join(OUTPUT_DIR, "hybrid_best_sequences.csv")
    best_sequences.sort(key=lambda x: x[1], reverse=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode_Fidelity', 'Sequence_Length', 'Gate_Sequence'])
        for seq_list, fid in best_sequences:
            writer.writerow([f"{fid:.6f}", len(seq_list), " -> ".join(seq_list)])

# ==============================================================================
# 6. MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == '__main__':
    EPISODES = 20000
    STEPS_PER_EPISODE = 50 # Increased slightly to allow sufficient sequence depth
    UPDATE_TIMESTEP = 2048 

    env = HybridUnitaryEnv()
    agent = HybridPPOAgent(num_discrete=env.num_discrete_actions, obs_dim=env.obs_dim, decay_steps=EPISODES)
    
    print(f"--- Starting Hybrid PPO Matrix Decomposition Search ---")
    print(f"Targeting Unitary Matrix of shape: {env.target_U.shape}")
    print(f"Action Space: {env.num_discrete_actions} Discrete Gates + 1 Continuous Angle (phi)\n")
    
    timestep_counter = 0
    all_fidelities = []
    best_sequences = []
    global_max_fidelity = 0.0
    
    start_time = time.time()

    for episode in range(EPISODES):
        observation = env.reset()
        episode_max_fid = 0.0
        episode_sequence = []
        
        for step in range(STEPS_PER_EPISODE):
            timestep_counter += 1
            
            d_act, c_act, d_lp, c_lp = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(d_act, c_act)
            
            agent.store_transition(observation, d_act, c_act, reward, next_observation, done, d_lp, c_lp, info['cont_used'])
            observation = next_observation
            
            episode_sequence.append(info['gate'])
            episode_max_fid = max(episode_max_fid, info['fidelity'])
            
            if timestep_counter % UPDATE_TIMESTEP == 0:
                agent.learn()
                
            if done: break
            
        all_fidelities.append(episode_max_fid)
        
        if episode_max_fid > global_max_fidelity:
            global_max_fidelity = episode_max_fid
            if global_max_fidelity > 0.05: 
                print(f"!!! NEW RECORD FIDELITY: {global_max_fidelity:.6f} at Ep {episode+1} !!!")
                
        if episode_max_fid > 0.90:
            best_sequences.append((episode_sequence.copy(), episode_max_fid))

        if (episode + 1) % 100 == 0:
            avg_fid = np.mean(all_fidelities[-100:])
            curr_lr = float(agent.lr_schedule(agent.actor_optimizer.iterations).numpy())
            elapsed = time.time() - start_time
            print(f'Ep: {episode+1}/{EPISODES} | Avg Fid: {avg_fid:.4f} | Best: {global_max_fidelity:.4f} | Cont. STD: {agent.cont_std:.3f} | LR: {curr_lr:.2e} | Time: {elapsed:.1f}s')

    print("\nTraining Complete. Saving Results...")
    save_results(all_fidelities, best_sequences)
    print(f"Results saved to {OUTPUT_DIR}/")