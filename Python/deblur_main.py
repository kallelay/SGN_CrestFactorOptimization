import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.linalg import inv

# Crest Factor Optimization for Multisine Signals
# Based on the Deblur algorithm by Ahmed Yahia Kallel, MST TU Chemnitz

# %% Signal Configuration
signal_config = {
    'min_frequency': 1,    # Minimum frequency in Hz
    'max_frequency': 1000, # Maximum frequency in Hz
    'num_frequencies': 10, # Number of frequency components
    'num_periods': 1,      # Number of periods of the lowest frequency
    'sampling_rate': 10e3, # Sampling rate in Hz
    'use_frequency_bin_optimization': True # Whether to use frequency bin optimization
}

# %% Algorithm Configuration
algo_config = {
    'max_iterations': 5000,  # Maximum number of iterations for optimization
    'sigmoid_factor': -1,    # Steepness factor for sigmoid transform
    'lambda': 1e-6,          # Regularization parameter for Gauss-Newton optimization
    'p_norm': 256,           # p-norm value for optimization (should be even)
    'animate': True,         # Whether to show animation of optimization process
    'max_iter_per_mode_stag': 300, # Maximum iterations to run before recovering from stagnation
    'max_iter_per_mode': 500 # Maximum iterations before switching modes
}

# %% Helper functions
def generate_optimized_frequencies(fmin, fmax, num_freqs, time_vector, sampling_rate, use_optimization):
    if use_optimization:
        log_freqs = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)
        freq_bins = np.fft.fftfreq(len(time_vector), 1/sampling_rate)
        freq_bin_indices = np.argmin(np.abs(freq_bins[:, np.newaxis] - log_freqs), axis=0)
        frequencies = freq_bins[freq_bin_indices]
    else:
        frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)
        freq_bin_indices = np.arange(num_freqs)
    return frequencies, freq_bin_indices

def generate_multisine_signal(amplitudes, frequencies, phases, time_vector):
    # Ensure all inputs are numpy arrays
    amplitudes = np.array(amplitudes).reshape(-1, 1)  # Nx1
    frequencies = np.array(frequencies).reshape(-1, 1)  # Nx1
    phases = np.array(phases).reshape(-1, 1)  # Nx1
    time_vector = np.array(time_vector).reshape(1, -1)  # 1xM

    # Generate the multisine signal
    # This will create an NxM matrix, which we then sum along axis 0 to get a 1xM vector
    return np.sum(amplitudes * np.cos(2*np.pi*frequencies@time_vector + phases), axis=0)



def calculate_crest_factor(signal):
    return np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))

def sigmoid_transform(signal, k):
    return 1 / (1 + np.exp(-k * signal)) - 0.5

# %% Generate time vector & frequency vector, optimize frequency vector
time_vector = np.arange(0, signal_config['num_periods']/signal_config['min_frequency'], 1/signal_config['sampling_rate'])
frequencies, freq_bin_indices = generate_optimized_frequencies(
    signal_config['min_frequency'], signal_config['max_frequency'],
    signal_config['num_frequencies'], time_vector, signal_config['sampling_rate'],
    signal_config['use_frequency_bin_optimization']
)

frequencies = np.unique(frequencies)
freq_bin_indices = np.unique(freq_bin_indices)

print('Optimized Frequencies:')
print(frequencies)

amplitude_vector = np.ones(len(frequencies)) / np.sqrt(len(frequencies))
initial_phases = np.random.rand(len(frequencies)) * 2 * np.pi

# %% Generate initial signal
initial_signal = generate_multisine_signal(amplitude_vector, frequencies, initial_phases, time_vector)
initial_crest_factor = calculate_crest_factor(initial_signal)

# %% Initialize optimization variables
optimization_results = {
    'current_phases': initial_phases,
    'crest_factor_history': np.zeros(algo_config['max_iterations']),
    'optimization_mode': np.zeros(algo_config['max_iterations']),
    'elapsed_time': np.zeros(algo_config['max_iterations']),
    'best_crest_factor': np.inf,
    'best_phases': initial_phases
}

# %% Main optimization loop
current_mode = 0  # 0: Sigmoid transform, 1: Gauss-Newton
num_rounds = 0
stagnation_counter = 0
mode_counter = 0
stagnation_threshold = 1e-6  # Threshold for detecting stagnation
previous_crest_factor = np.inf
ModeName = ['Sigmoid Transform', 'Gauss-Newton']

# Set up animation
if algo_config['animate']:
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    line_initial, = ax1.plot(time_vector, initial_signal, 'b')
    line_current, = ax1.plot(time_vector, initial_signal, 'r')
    ax1.set_title('Time Domain')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.set_ylim(-2*np.max(np.abs(initial_signal)), 2*np.max(np.abs(initial_signal)))
    ax1.legend(['Initial', 'Current'])

    line_cf, = ax2.plot(1, initial_crest_factor)
    ax2.set_title('Crest Factor History')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Crest Factor')
    ax2.grid(True)
    ax2.set_ylim(0, 2*initial_crest_factor)

    fig.suptitle('Crest Factor Optimization Progress')
    text = fig.text(0.1, 0.95, '', ha='center')

current_signal = initial_signal.copy()

for iteration in range(algo_config['max_iterations']):
    if current_mode == 0:
        # Sigmoid transform optimization
        transformed_signal = sigmoid_transform(current_signal, algo_config['sigmoid_factor'])
        fft_result = fft(transformed_signal)
        optimization_results['current_phases'] = np.angle(fft_result[freq_bin_indices])
    else:
        # Gauss-Newton optimization
        q = algo_config['p_norm'] / 2
        def phase_derivative(amplitudes, phases, signal, time):
            return -q * amplitudes[:, np.newaxis] * signal**(q-1) * np.sin(2*np.pi*frequencies[:, np.newaxis]*time + phases[:, np.newaxis])
        
        residual = (current_signal**q).T
        jacobian = phase_derivative(amplitude_vector, optimization_results['current_phases'], current_signal, time_vector).T
        optimization_results['current_phases'] -= np.dot(
            inv(np.dot(jacobian.T, jacobian) + np.eye(jacobian.shape[1]) * algo_config['lambda']),
            np.dot(jacobian.T, residual)
        ).flatten()

    # Generate new signal
    current_signal = generate_multisine_signal(amplitude_vector, frequencies, optimization_results['current_phases'], time_vector)

    # Calculate and store crest factor
    current_crest_factor = calculate_crest_factor(current_signal)
    relative_improvement = np.abs(previous_crest_factor - current_crest_factor) / previous_crest_factor
    optimization_results['crest_factor_history'][iteration] = current_crest_factor
    optimization_results['optimization_mode'][iteration] = current_mode

    mode_counter += 1

    # Update best crest factor
    if current_crest_factor < optimization_results['best_crest_factor']:
        optimization_results['best_crest_factor'] = current_crest_factor
        optimization_results['best_phases'] = optimization_results['current_phases']
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    # Check for stagnation and switch modes
    if (stagnation_counter >= algo_config['max_iter_per_mode_stag']) or (mode_counter > algo_config['max_iter_per_mode']):
        current_mode = 1 - current_mode
        stagnation_counter = 0
        mode_counter = 0
        num_rounds += 1
        print(f'End of Round {num_rounds}, best crest factor: {optimization_results["best_crest_factor"]:.6f}')
        previous_crest_factor = np.inf
    elif relative_improvement < stagnation_threshold and current_mode == 1:
        current_mode = 0  # Switch back to Sigmoid Transform
        stagnation_counter = 0
        mode_counter = 0
        num_rounds += 1
        print(f'Gauss-Newton stagnated, switching to Sigmoid Transform, Round {num_rounds}, best crest factor: {optimization_results["best_crest_factor"]:.6f}')
        previous_crest_factor = np.inf
    else:
        previous_crest_factor = current_crest_factor

    # Update animation if enabled
    if algo_config['animate'] and iteration % 10 == 0:
        line_current.set_ydata(current_signal)
        line_cf.set_xdata(range(1, iteration+2))
        line_cf.set_ydata(optimization_results['crest_factor_history'][:iteration+1])
        ax2.relim()
        ax2.autoscale_view()
        text.set_text(f'Iteration: {iteration}, Mode: {ModeName[current_mode]}, Current CF: {current_crest_factor:.4f}, Best CF: {optimization_results["best_crest_factor"]:.4f}')
        plt.draw()
        plt.pause(0.001)

# %% Generate final optimized signal
optimized_signal = generate_multisine_signal(amplitude_vector, frequencies, optimization_results['best_phases'], time_vector)
final_crest_factor = calculate_crest_factor(optimized_signal)

# %% Plot final results
plt.ioff()
plt.clf()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(time_vector, initial_signal, 'b', time_vector, optimized_signal, 'r')
ax1.set_title('Initial vs Optimized Signal')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend(['Initial', 'Optimized'])
ax1.grid(True)

ax2.plot(optimization_results['crest_factor_history'])
ax2.set_title('Crest Factor History')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Crest Factor')
ax2.grid(True)

fig.suptitle(f'Crest Factor Optimization Results\nInitial CF: {initial_crest_factor:.4f}, Final CF: {final_crest_factor:.4f}')

# %% Print results
print(f'Initial Crest Factor: {initial_crest_factor:.4f}')
print(f'Final Crest Factor: {final_crest_factor:.4f}')
print(f'Improvement: {(initial_crest_factor - final_crest_factor) / initial_crest_factor * 100:.2f}%')

# Print final phases
print('Final Optimized Phases:')
for freq, phase in zip(frequencies, optimization_results['best_phases']):
    print(f'Frequency: {freq:.2f} Hz, Phase: {phase:.4f} rad')

plt.show()