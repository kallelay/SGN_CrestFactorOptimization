# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:47:21 2024

@author: Kallel
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
from scipy.fft import fft
from scipy.linalg import inv

app = Flask(__name__)



# Crest Factor Optimization for Multisine Signals
# Based on the Deblur algorithm by Ahmed Yahia Kallel, MST TU Chemnitz
# Translated using Claude.AI

# %% Signal Configuration
signal_config = {
    'min_frequency': 1,    # Minimum frequency in Hz
    'max_frequency': 1000, # Maximum frequency in Hz
    'num_frequencies': 10, # Number of frequency components
    'num_periods': 1,      # Number of periods of the lowest frequency
    'sampling_rate': 10e3, # Sampling rate in Hz
    'use_frequency_bin_optimization': True # Whether to use frequency bin optimization
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

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crest Factor Optimization</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; }
            .form-group { margin-bottom: 20px; }
            .plot-container { width: 100%; height: 400px; }
            #made-with-love { position: fixed; bottom: 10px; right: 10px; font-style: italic; }
            #optimization-info { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">Crest Factor Optimization for Multisine Signals</h1>

            <div class="row">
                <div class="col-md-6">
                    <h2>Signal Configuration</h2>
                    <form id="signal-config">
                        <div class="form-group">
                            <label for="min-frequency">Minimum Frequency (Hz):</label>
                            <input type="number" class="form-control" id="min-frequency" value="1" step="0.1">
                        </div>
                        <div class="form-group">
                            <label for="max-frequency">Maximum Frequency (Hz):</label>
                            <input type="number" class="form-control" id="max-frequency" value="100" step="0.1">
                        </div>
                        <div class="form-group">
                            <label for="num-frequencies">Number of Frequencies: <span id="num-frequencies-value">10</span></label>
                            <input type="range" class="form-range" id="num-frequencies" min="2" max="200" value="10">
                        </div>
                        <div class="form-group">
                            <label for="num-periods">Number of Periods: <span id="num-periods-value">1</span></label>
                            <input type="range" class="form-range" id="num-periods" min="1" max="10" value="1">
                        </div>
                        <div class="form-group">
                            <label for="sampling-rate">Sampling Rate (Hz):</label>
                            <input type="number" class="form-control" id="sampling-rate" value="1000" step="1">
                        </div>
                        <div class="form-check">
                            <input type="checkbox" class="form-check-input" id="use-frequency-bin-optimization" checked>
                            <label class="form-check-label" for="use-frequency-bin-optimization">Use Frequency Bin Optimization</label>
                        </div>
                    </form>
                </div>

                <div class="col-md-6">
                    <h2>Algorithm Configuration</h2>
                    <form id="algo-config">
                        <div class="form-group">
                            <label for="max-iterations">Maximum Iterations: <span id="max-iterations-value">1000</span></label>
                            <input type="range" class="form-range" id="max-iterations" min="100" max="10000" step="100" value="1000">
                        </div>
                        <div class="form-group">
                            <label for="sigmoid-factor">Sigmoid Factor:</label>
                            <input type="number" class="form-control" id="sigmoid-factor" value="1" step="1">
                        </div>
                        <div class="form-group">
                            <label for="lambda">Lambda:</label>
                            <input type="number" class="form-control" id="lambda" value="1e-5" step="1e-6">
                        </div>
                        <div class="form-group">
                            <label for="p-norm">p-norm: <span id="p-norm-value">256</span></label>
                            <input type="range" class="form-range" id="p-norm" min="2" max="1000" step="2" value="256">
                        </div>
                        <div class="form-group">
                            <label for="max-iter-per-mode">Max Iterations per Mode: <span id="max-iter-per-mode-value">100</span></label>
                            <input type="range" class="form-range" id="max-iter-per-mode" min="10" max="1000" step="10" value="100">
                        </div>
                    </form>
                </div>
            </div>

            <div class="text-center mt-4">
                <button id="optimize-btn" class="btn btn-primary btn-lg">Optimize</button>
                <button id="stop-btn" class="btn btn-danger btn-lg" style="display: none;">Stop Optimization</button>
            </div>

            <div id="optimization-info" class="mt-4">
                <h2>Optimization Information</h2>
                <div class="mb-3">
                    <label>Initial Crest Factor:</label>
                    <span id="initial-crest-factor" class="badge bg-secondary"></span>
                </div>
                <div class="mb-3">
                    <label>Final Crest Factor:</label>
                    <span id="final-crest-factor" class="badge bg-success"></span>
                </div>
                <div class="mb-3">
                    <label>Improvement:</label>
                    <span id="improvement" class="badge bg-info"></span>
                </div>
            </div>

            <div class="mt-4">
                <h2>Results</h2>
                <div class="mb-3">
                    <label>Optimized Frequencies:</label>
                    <pre id="optimized-frequencies" class="bg-light p-2"></pre>
                </div>
                <div class="mb-3">
                    <label>Final Optimized Phases:</label>
                    <pre id="final-phases" class="bg-light p-2"></pre>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div id="time-domain-plot" class="plot-container"></div>
                </div>
                <div class="col-md-6">
                    <div id="crest-factor-plot" class="plot-container"></div>
                </div>
            </div>
        </div>

        <div id="made-with-love">Made with ❤️ using Claude, Python and Flask</div>

        <script>
            // Update slider values
            $('input[type="range"]').on('input', function() {
                $(`#${this.id}-value`).text(this.value);
            });

            let isOptimizing = false;

            function optimize() {
                if (isOptimizing) {
                    console.log("Optimization already in progress");
                    return;
                }

                isOptimizing = true;
                $('#optimize-btn').hide();
                $('#stop-btn').show();

                const signalConfig = {
                    minFrequency: parseFloat($('#min-frequency').val()),
                    maxFrequency: parseFloat($('#max-frequency').val()),
                    numFrequencies: parseInt($('#num-frequencies').val()),
                    numPeriods: parseInt($('#num-periods').val()),
                    samplingRate: parseFloat($('#sampling-rate').val()),
                    useFrequencyBinOptimization: $('#use-frequency-bin-optimization').is(':checked')
                };

                const algoConfig = {
                    maxIterations: parseInt($('#max-iterations').val()),
                    sigmoidFactor: parseFloat($('#sigmoid-factor').val()),
                    lambda: parseFloat($('#lambda').val()),
                    pNorm: parseInt($('#p-norm').val()),
                    maxIterPerMode: parseInt($('#max-iter-per-mode').val())
                };

                fetch('/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ signalConfig, algoConfig }),
                })
                .then(response => response.json())
                .then(data => {
                    $('#initial-crest-factor').text(data.initialCrestFactor.toFixed(4));
                    $('#final-crest-factor').text(data.finalCrestFactor.toFixed(4));
                    $('#improvement').text(`${data.improvement.toFixed(2)}%`);

                    $('#optimized-frequencies').text(data.optimizedFrequencies.join(', '));

                    let phasesText = '';
                    for (let i = 0; i < data.optimizedFrequencies.length; i++) {
                        phasesText += `Frequency: ${data.optimizedFrequencies[i].toFixed(2)} Hz, Phase: ${data.optimizedPhases[i].toFixed(4)} rad\n`;
                    }
                    $('#final-phases').text(phasesText);

                    Plotly.newPlot('time-domain-plot', [
                        {y: data.initialSignal, name: 'Initial Signal'},
                        {y: data.optimizedSignal, name: 'Optimized Signal'}
                    ], {
                        title: 'Time Domain',
                        xaxis: {title: 'Time'},
                        yaxis: {title: 'Amplitude'}
                    });

                    Plotly.newPlot('crest-factor-plot', [
                        {y: data.crestFactorHistory, name: 'Crest Factor History'}
                    ], {
                        title: 'Crest Factor History',
                        xaxis: {title: 'Iteration'},
                        yaxis: {title: 'Crest Factor'}
                    });

                    finishOptimization();
                })
                .catch(error => {
                    console.error('Error during optimization:', error);
                    finishOptimization();
                });
            }

            function finishOptimization() {
                $('#stop-btn').hide();
                $('#optimize-btn').show();
                isOptimizing = false;
            }

            $('#optimize-btn').click(optimize);
            $('#stop-btn').click(() => {
                // Implement stop functionality if needed
                finishOptimization();
            });
        </script>
    </body>
    </html>
    ''')
import sys

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    signal_config = data['signalConfig']
    algo_config = data['algoConfig']
    app.logger.info(signal_config)
    app.logger.info(algo_config)
    app.logger.info('a')
  
    # Generate time vector & frequency vector, optimize frequency vector
    time_vector = np.arange(0, signal_config['numPeriods']/signal_config['minFrequency'], 1/signal_config['samplingRate'])
    frequencies, freq_bin_indices = generate_optimized_frequencies(
        signal_config['minFrequency'], signal_config['maxFrequency'],
        signal_config['numFrequencies'], time_vector, signal_config['samplingRate'],
        signal_config['useFrequencyBinOptimization']
    )

    frequencies = np.unique(frequencies)
    freq_bin_indices = np.unique(freq_bin_indices)

    amplitude_vector = np.ones(len(frequencies)) / np.sqrt(len(frequencies))
    initial_phases = np.random.rand(len(frequencies)) * 2 * np.pi

    # Generate initial signal
    initial_signal = generate_multisine_signal(amplitude_vector, frequencies, initial_phases, time_vector)
    initial_crest_factor = calculate_crest_factor(initial_signal)
    
    signal = initial_signal.copy()

    # Initialize optimization variables
    current_phases = initial_phases.copy()
    best_crest_factor = np.inf
    best_phases = initial_phases.copy()
    crest_factor_history = []
    

    # Main optimization loop
    current_mode = 0  # 0: Sigmoid transform, 1: Gauss-Newton
    for iteration in range(algo_config['maxIterations']):
        if current_mode == 0:
            transformed_signal = sigmoid_transform(signal, algo_config['sigmoidFactor'])
            fft_result = fft(transformed_signal)
            current_phases = np.angle(fft_result[freq_bin_indices])
        else:
            # Gauss-Newton optimization
            q = np.int16(algo_config['pNorm'] / 2)
            def phase_derivative(amplitudes, phases, signal, time):
                 return -q * amplitudes[:, np.newaxis] * signal**(q-1) * np.sin(2*np.pi*frequencies[:, np.newaxis]*time + phases[:, np.newaxis])
     
            residual = (signal**q).T
            jacobian = phase_derivative(amplitude_vector, current_phases, signal, time_vector).T
            current_phases -= np.dot(
                inv(np.dot(jacobian.T, jacobian) + np.eye(jacobian.shape[1]) * algo_config['lambda']),
                np.dot(jacobian.T, residual)
            ).flatten()

        # Generate new signal
        signal = generate_multisine_signal(amplitude_vector, frequencies, current_phases, time_vector)
        current_crest_factor = calculate_crest_factor(signal)
        crest_factor_history.append(current_crest_factor)

        if current_crest_factor < best_crest_factor:
            best_crest_factor = current_crest_factor
            best_phases = current_phases.copy()

        # Switch modes (simplified for this example)
        if iteration % algo_config['maxIterPerMode'] == 0:
            current_mode = 1 - current_mode

    # Generate final optimized signal
    optimized_signal = generate_multisine_signal(amplitude_vector, frequencies, best_phases, time_vector)
    final_crest_factor = calculate_crest_factor(optimized_signal)

    return jsonify({
        'initialCrestFactor': float(initial_crest_factor),
        'finalCrestFactor': float(final_crest_factor),
        'improvement': float((initial_crest_factor - final_crest_factor) / initial_crest_factor * 100),
        'optimizedFrequencies': frequencies.tolist(),
        'optimizedPhases': best_phases.tolist(),
        'crestFactorHistory': crest_factor_history,
        'initialSignal': initial_signal.tolist(),
        'optimizedSignal': optimized_signal.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)