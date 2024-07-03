% Crest Factor Optimization for Multisine Signals
% Based on the Deblur algorithm by Ahmed Yahia Kallel, MST TU Chemnitz

%% Signal Configuration
signal_config = struct();
signal_config.min_frequency = 1;    % Minimum frequency in Hz
signal_config.max_frequency = 100; % Maximum frequency in Hz
signal_config.num_frequencies = 10; % Number of frequency components
signal_config.num_periods = 1;      % Number of periods of the lowest frequency
signal_config.sampling_rate = 1e3; % Sampling rate in Hz
signal_config.use_frequency_bin_optimization = true; % Whether to use frequency bin optimization

%% Algorithm Configuration
algo_config = struct();
algo_config.max_iterations = 5000;  % Maximum number of iterations for optimization
algo_config.sigmoid_factor = -1;    % Steepness factor for sigmoid transform
algo_config.lambda = 1e-6;          % Regularization parameter for Gauss-Newton optimization
algo_config.p_norm = 256;           % p-norm value for optimization (should be even)
algo_config.animate = true;         % Whether to show animation of optimization process
algo_config.use_gpu = false;        % Whether to use GPU acceleration
algo_config.max_iter_per_mode_stag = 300;% Maximum iterations to run before recovering from stagnation
algo_config.max_iter_per_mode = 500;% Maximum iterations before switching modes

%% Generate time vector & frequency vector, optimize frequency vector
time_vector = 0 : 1/signal_config.sampling_rate : signal_config.num_periods/signal_config.min_frequency - 1/signal_config.sampling_rate;
[frequencies, freq_bin_indices] = generate_optimized_frequencies(signal_config.min_frequency, signal_config.max_frequency, ...
    signal_config.num_frequencies, time_vector, signal_config.sampling_rate, signal_config.use_frequency_bin_optimization);

frequencies = unique(frequencies);
freq_bin_indices = unique(freq_bin_indices);

% Print optimized frequencies
disp('Optimized Frequencies:');
disp(array2table(frequencies, 'VariableNames', {'Frequency_Hz'}));

amplitude_vector = ones(length(frequencies), 1) / sqrt(length(frequencies));
initial_phases = rand(length(frequencies), 1) * 2 * pi;

% Move data to GPU if GPU mode is enabled
if algo_config.use_gpu
    time_vector = gpuArray(time_vector);
    frequencies = gpuArray(frequencies);
    amplitude_vector = gpuArray(amplitude_vector);
    initial_phases = gpuArray(initial_phases);
end

%% Generate initial signal
initial_signal = generate_multisine_signal(amplitude_vector, frequencies, initial_phases, time_vector);
initial_crest_factor = calculate_crest_factor(initial_signal);

%% Initialize optimization variables
optimization_results = struct();
optimization_results.current_phases = initial_phases;
optimization_results.crest_factor_history = zeros(algo_config.max_iterations, 1);
optimization_results.optimization_mode = zeros(algo_config.max_iterations, 1);
optimization_results.elapsed_time = zeros(algo_config.max_iterations, 1);
optimization_results.best_crest_factor = inf;
optimization_results.best_phases = initial_phases;

%% Main optimization loop
tic;
current_mode = 0;  % 0: Sigmoid transform, 1: Gauss-Newton
num_rounds = 0;
stagnation_counter = 0;
mode_counter = 0;
stagnation_threshold = 1e-6;  % Threshold for detecting stagnation
previous_crest_factor = inf;
ModeName = {'Sigmoid Transform', 'Gauss-Newton'};


% Set up animation
if algo_config.animate
    clf;
    subplot(2,1,1);
    plot(time_vector, initial_signal, 'b');
    hold on;
    time_plot = plot(time_vector, initial_signal, 'r');
    title('Time Domain'); xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    ylim([-2*max(abs(initial_signal)), 2*max(abs(initial_signal))]);
    legend('Initial', 'Current');
    
    subplot(2,1,2);
    cf_plot = plot(1, initial_crest_factor);
    title('Crest Factor History'); xlabel('Iteration'); ylabel('Crest Factor'); grid on;
    ylim([0, 2*initial_crest_factor]);
    
    sgtitle('Crest Factor Optimization Progress');
    
    text_handle = annotation('textbox', [0.1, 0.95, 0.8, 0.05], 'String', '', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end

current_signal = initial_signal;

for iteration = 1:algo_config.max_iterations
    if current_mode == 0
        % Sigmoid transform optimization
        transformed_signal = sigmoid_transform(current_signal, algo_config.sigmoid_factor);
        fft_result = fft(transformed_signal);
        optimization_results.current_phases = angle(fft_result(freq_bin_indices)).';

        
    else
        % Gauss-Newton optimization
        q = algo_config.p_norm / 2;
        phase_derivative = @(amplitudes, phases, signal, time) ...
            -q .* amplitudes .* signal.^(q-1) .* sin(2.*pi.*frequencies.*time + phases);
        residual = (current_signal.^q)';
        jacobian = [phase_derivative(amplitude_vector, optimization_results.current_phases, current_signal, time_vector)]';
        optimization_results.current_phases = optimization_results.current_phases - ...
            (inv(jacobian'*jacobian + eye(size(jacobian,2)) * algo_config.lambda)) * jacobian' * residual;
        % current_crest_factor = 

        previous_crest_factor = current_crest_factor;
    end
    
    % Generate new signal
    current_signal = generate_multisine_signal(amplitude_vector, frequencies, ...
        optimization_results.current_phases, time_vector);
    
    % Calculate and store crest factor
    current_crest_factor = calculate_crest_factor(current_signal);
    relative_improvement = abs(previous_crest_factor - current_crest_factor) / previous_crest_factor;
    optimization_results.crest_factor_history(iteration) = current_crest_factor;
    optimization_results.optimization_mode(iteration) = current_mode;
    optimization_results.elapsed_time(iteration) = toc;

    mode_counter= mode_counter + 1; %add mode counter + 1
    
    % Update best crest factor
    if current_crest_factor < optimization_results.best_crest_factor
        optimization_results.best_crest_factor = current_crest_factor;
        optimization_results.best_phases = optimization_results.current_phases;
        stagnation_counter = 0;
    else
        stagnation_counter = stagnation_counter + 1;
    end
    
    % Check for stagnation and switch modes
    if (stagnation_counter >= algo_config.max_iter_per_mode_stag) || (mode_counter > algo_config.max_iter_per_mode)
        current_mode = 1 - current_mode;
        stagnation_counter = 0;
        mode_counter = 0;
        num_rounds = num_rounds + 1;
        fprintf('End of Round %d, best crest factor: %2.6f\n', num_rounds, optimization_results.best_crest_factor);
        
        % Reset previous_crest_factor for the new mode
        previous_crest_factor = inf;
    elseif relative_improvement < stagnation_threshold && current_mode == 1
            current_mode = 0;  % Switch back to Sigmoid Transform
            stagnation_counter = 0;
            mode_counter = 0;
            num_rounds = num_rounds + 1;
            fprintf('Gauss-Newton stagnated, switching to Sigmoid Transform, Round %d, best crest factor: %2.6f\n', num_rounds, optimization_results.best_crest_factor);
            previous_crest_factor = inf;
    
    else
        previous_crest_factor = current_crest_factor;
    end
    
    
    % Update animation if enabled
    if algo_config.animate && mod(iteration, 10) == 0
        % Time domain animation
        set(time_plot, 'YData', gather(current_signal));
        
        % Crest factor history animation
        set(cf_plot, 'XData', 1:iteration, 'YData', optimization_results.crest_factor_history(1:iteration));
        
        % Update text
        set(text_handle, 'String', sprintf('Iteration: %d, Mode: %s, Current CF: %.4f, Best CF: %.4f', ...
            iteration, ModeName{current_mode+1}, current_crest_factor, optimization_results.best_crest_factor));
        
        drawnow;
    end
end

%% Generate final optimized signal
optimized_signal = generate_multisine_signal(amplitude_vector, frequencies, ...
    optimization_results.best_phases, time_vector);
final_crest_factor = calculate_crest_factor(optimized_signal);

%% Plot final results
clf;
subplot(2,1,1);
plot(gather(time_vector), gather(initial_signal), 'b', gather(time_vector), gather(optimized_signal), 'r');
title('Initial vs Optimized Signal');
xlabel('Time (s)'); ylabel('Amplitude');
legend('Initial', 'Optimized');
grid on;

subplot(2,1,2);
plot(optimization_results.crest_factor_history);
title('Crest Factor History');
xlabel('Iteration'); ylabel('Crest Factor');
grid on;

sgtitle(sprintf('Crest Factor Optimization Results\nInitial CF: %.4f, Final CF: %.4f', initial_crest_factor, final_crest_factor));

%% Print results
fprintf('Initial Crest Factor: %.4f\n', initial_crest_factor);
fprintf('Final Crest Factor: %.4f\n', final_crest_factor);
fprintf('Improvement: %.2f%%\n', (initial_crest_factor - final_crest_factor) / initial_crest_factor * 100);

% Print final phases
disp('Final Optimized Phases:');
phase_table = table(frequencies, optimization_results.best_phases, 'VariableNames', {'Frequency_Hz', 'Phase_rad'});
disp(phase_table);

%% Helper functions

function [frequencies, freq_bin_indices] = generate_optimized_frequencies(fmin, fmax, num_freqs, time_vector, sampling_rate, use_optimization)
    % Generates optimized frequencies for the multisine signal
    %
    % Inputs:
    %   fmin: Minimum frequency (Hz)
    %   fmax: Maximum frequency (Hz)
    %   num_freqs: Number of frequency components
    %   time_vector: Time vector of the signal
    %   sampling_rate: Sampling rate (Hz)
    %   use_optimization: Boolean flag for frequency bin optimization
    %
    % Outputs:
    %   frequencies: Vector of optimized frequencies
    %   freq_bin_indices: Indices of frequency bins (for FFT)

    if use_optimization
        log_freqs = logspace(log10(fmin), log10(fmax), num_freqs);
        freq_bins = (0:length(time_vector)-1) * sampling_rate / length(time_vector);
        [~, freq_bin_indices] = min(abs(freq_bins' - log_freqs), [], 1);
        frequencies = freq_bins(freq_bin_indices);
    else
        frequencies = logspace(log10(fmin), log10(fmax), num_freqs);
        freq_bin_indices = 1:num_freqs;
    end
    frequencies = frequencies(:);
end

function signal = generate_multisine_signal(amplitudes, frequencies, phases, time_vector)
    % Generates a multisine signal
    %
    % Inputs:
    %   amplitudes: Vector of amplitudes for each frequency component
    %   frequencies: Vector of frequencies
    %   phases: Vector of phases for each frequency component
    %   time_vector: Time vector of the signal
    %
    % Output:
    %   signal: Generated multisine signal

    signal = sum(amplitudes .* cos(2*pi*frequencies.*time_vector + phases), 1);
end

function crest_factor = calculate_crest_factor(signal)
    % Calculates the crest factor of a signal
    %
    % Input:
    %   signal: Input signal
    %
    % Output:
    %   crest_factor: Calculated crest factor

    crest_factor = max(abs(signal)) / rms(signal);
end

function transformed = sigmoid_transform(signal, k)
    % Applies sigmoid transform to a signal
    %
    % Inputs:
    %   signal: Input signal
    %   k: Steepness factor for sigmoid function
    %
    % Output:
    %   transformed: Transformed signal

    transformed = 1 ./ (1 + exp(-k * signal)) - 0.5;
end