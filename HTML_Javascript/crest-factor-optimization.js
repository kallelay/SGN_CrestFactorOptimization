$('input[type="range"]').on('input', function() {
  $(`#${this.id}-value`).text(this.value);
});


function disableConfigurations() {
  $('#signal-config input, #algo-config input').not('#animate, #update-every').prop('disabled', true);
  $('#use-frequency-bin-optimization').prop('disabled', true);
}

function enableConfigurations() {
  $('#signal-config input, #algo-config input').prop('disabled', false);
  $('#use-frequency-bin-optimization').prop('disabled', false);
}



// Helper functions
function linspace(start, end, num) {
  if (num < 2) return num === 1 ? [start] : [];
  const step = (end - start) / (num - 1);
  return Array.from({length: num}, (_, i) => start + step * i);
}

function logspace(a, b, n) {
  const logStart = Math.log10(a);
  const logEnd = Math.log10(b);
  const linArray = linspace(logStart, logEnd, n);
  return linArray.map(x => Math.pow(10, x));
}

/*function fftfreq(n, Fs) {
  return Array.from({length: n}, (_, i) => i * Fs / n);
}*/

	function fftfreq(n, Fs) {
		const result = [];
		const val = Fs / n;

		for (let i = 0; i < n; i++) {
			result.push(i * val);
		}

		return result;
	}


function findNearestIndices(array, fres) {
	return math.round(math.map(array, x=>x/fres));
	
 /* return values.map(value => {
    return array.reduce((closest, current, index) => 
      Math.abs(current - value) < Math.abs(array[closest] - value) ? index : closest, 0);
  });*/
}

function unique(array) {
  return Array.from(new Set(array));
}

function generateOptimizedFrequencies(fmin, fmax, numFreqs, timeVector, samplingRate, useOptimization) {
  if (useOptimization) {
    const logFreqs = logspace(fmin, fmax, numFreqs);
    const freqBins = fftfreq(timeVector.length, samplingRate);
	
	const N = samplingRate * timeVector[timeVector.length-1];
	const fres = samplingRate/N;
	
	
    const freqBinIndices = math.map(logFreqs, x=>math.round(x/fres));
    const frequencies = math.multiply(fres, freqBinIndices);
    return { frequencies, freqBinIndices: freqBinIndices.slice(0, frequencies.length) };
  } else {
    const frequencies = logspace(fmin, fmax, numFreqs);
    return { frequencies, freqBinIndices: Array.from({ length: numFreqs }, (_, i) => i) };
  }
}

function generateMultisineSignal(amplitudes, frequencies, phases, timevec) {
	
	 let angle = math.add(math.multiply(2 * Math.PI * frequencies[0], timevec), phases[0]);
	 let sum = math.multiply(amplitudes[0], math.map(angle, x=>math.cos(x)));
	 
	 N = amplitudes.length;

        for (let i = 1; i < N; i++) {
             angle = math.add(math.multiply(2 * Math.PI * frequencies[i], timevec), phases[i]);
            sum = math.add(sum, math.multiply(amplitudes[i], math.map(angle, x=>math.cos(x))));
        }
	return sum;
	
	
  /*  return timevec.map(t => 
        amplitudes.reduce((sum, amp, i) => 
            math.add(sum, 
                amplitudes[i] * math.cos(math.add(math.multiply(2 * Math.PI, frequencies[i], t), phases[i]))), 0)
    );*/
}

function calculateCrestFactor(signal) {
  const maxAbs = Math.max(...signal.map(Math.abs));
  const rms = Math.sqrt(signal.reduce((sum, x) => sum + x * x, 0) / signal.length);
  return maxAbs / rms;
}

function sigmoidTransform(signal, k) {
  return signal.map(x => 1 / (1 + Math.exp(-k * x)) - 0.5);
}

/*function phaseDerivative(amplitudes, phases, signal, time, q, frequencies) {
    const N = amplitudes.length;
    const M = time.length;
    const result = math.zeros(M, N);

    for (let j = 0; j < M; j++) {
        for (let i = 0; i < N; i++) {
            const angle = math.add(math.multiply(2 * Math.PI, frequencies[i], time[j]), phases[i]);
            result.set([j, i], math.multiply(-q, amplitudes[i], 
                math.pow(math.abs(signal[j]), math.subtract(q, 1)), 
                math.sin(angle)));
        }
    }

    return result;
}*/

function phaseDerivative(amplitudes, phases, signal, time, q, frequencies) {
    const N = amplitudes.length;
    const M = time.length;
    const result = [];

    for (let j = 0; j < M; j++) {
        const row = [];
        for (let i = 0; i < N; i++) {
            const angle = 2 * Math.PI * frequencies[i] * time[j] + phases[i];
            row.push(-q * amplitudes[i] * Math.pow(signal[j], q - 1) * Math.sin(angle));
        }
        result.push(row);
    }

    return math.matrix(result);
}



// Optimization variables
let isOptimizing = false;
let stopOptimization = false;
let optimizationTimeout = null;
let algoConfig = null;

$('#update-every').on('change', function() { if(algoConfig==null) return; algoConfig.updateEvery = parseInt($('#update-every').val());});
$('#animate').on('change', function() { if(algoConfig==null) return; algoConfig.animate =  $('#animate').is(':checked');});

function optimize() {
  if (isOptimizing) {
    console.log("Optimization already in progress");
    return;
  }

  isOptimizing = true;
  stopOptimization = false;
  $('#optimize-btn').hide();
  $('#stop-btn').show();
  disableConfigurations();

  const signalConfig = {
    minFrequency: parseFloat($('#min-frequency').val()),
    maxFrequency: parseFloat($('#max-frequency').val()),
    numFrequencies: parseInt($('#num-frequencies').val()),
    numPeriods: parseInt($('#num-periods').val()),
    samplingRate: parseFloat($('#sampling-rate').val()),
    useFrequencyBinOptimization: $('#use-frequency-bin-optimization').is(':checked')
  };

   algoConfig = {
    maxIterations: parseInt($('#max-iterations').val()),
    sigmoidFactor: parseFloat($('#sigmoid-factor').val()),
    lambda: parseFloat($('#lambda').val()),
    pNorm: parseInt($('#p-norm').val()),
    animate: $('#animate').is(':checked'),
    maxIterPerModeStag: parseInt($('#max-iter-per-mode-stag').val()),
    maxIterPerMode: parseInt($('#max-iter-per-mode').val()),
	updateEvery: parseInt($('#update-every').val())
  };

  const timeVector = linspace(0, signalConfig.numPeriods / signalConfig.minFrequency, Math.ceil(signalConfig.numPeriods * signalConfig.samplingRate / signalConfig.minFrequency));

  const {frequencies, freqBinIndices} = generateOptimizedFrequencies(
    signalConfig.minFrequency,
    signalConfig.maxFrequency,
    signalConfig.numFrequencies,
    timeVector,
    signalConfig.samplingRate,
    signalConfig.useFrequencyBinOptimization
  );

  const uniqueFrequencies = unique(frequencies);
  const uniqueFreqBinIndices = unique(freqBinIndices);

  $('#optimized-frequencies').text(uniqueFrequencies.join(', '));
  console.log('Optimized Frequencies:', uniqueFrequencies);

  const amplitudeVector = new Array(uniqueFrequencies.length).fill(1 / Math.sqrt(uniqueFrequencies.length));
  const initialPhases = new Array(uniqueFrequencies.length).fill().map(() => Math.random() * 2 * Math.PI);

  const initialSignal = generateMultisineSignal(amplitudeVector, uniqueFrequencies, initialPhases, timeVector);
  const initialCrestFactor = calculateCrestFactor(initialSignal);

  $('#initial-crest-factor').text(initialCrestFactor.toFixed(4));

  let currentPhases = initialPhases.slice();
  let bestCrestFactor = Infinity;
  let bestPhases = initialPhases.slice();
  let currentMode = 0;
  let stagnationCounter = 0;
  let modeCounter = 0;
  let previousCrestFactor = Infinity;

 let currentSignal = initialSignal.slice();

  Plotly.newPlot('time-domain-plot', [
    {y: initialSignal, name: 'Initial Signal'},
    {y: currentSignal, name: 'Current Signal'}
  ], {
    title: 'Time Domain',
    xaxis: {title: 'Time'},
    yaxis: {title: 'Amplitude'}
  });

  Plotly.newPlot('crest-factor-plot', [
    {x: [], y: [], name: 'Crest Factor History'}
  ], {
    title: 'Crest Factor History',
    xaxis: {title: 'Iteration'},
    yaxis: {title: 'Crest Factor'}
  });


function numericalJacobian(f, x, h = 1e-8) {
    const n = x.length;
    const fx = f(x);
    const m = fx.length;
    const J = [];

    for (let j = 0; j < n; j++) {
        const xh = [...x];
        xh[j] += h;
        const fxh = f(xh);
        
        const Jcol = [];
        for (let i = 0; i < m; i++) {
            Jcol.push((fxh[i] - fx[i]) / h);
        }
        J.push(Jcol);
    }

    return math.transpose(J);
}

function calculateGradient(amplitudes, frequencies, phases, signal, timeVector, q) {
    const gradient = new Array(phases.length).fill(0);
    const N = timeVector.length;
    
    for (let i = 0; i < phases.length; i++) {
        for (let t = 0; t < N; t++) {
            const angle = 2 * Math.PI * frequencies[i] * timeVector[t] + phases[i];
            gradient[i] -= q * amplitudes[i] * Math.pow(Math.abs(signal[t]), q - 1) * 
                           Math.sin(angle) * signal[t] / N;
        }
    }
    
    return gradient;
}

  let fftResult = 0;
curiter = -1;
  function optimizationStep(iteration) {
    currentSignal = generateMultisineSignal(amplitudeVector, uniqueFrequencies, currentPhases, timeVector);
	  if (curiter == iteration) return;
	  curiter++;
    if (stopOptimization || iteration >= algoConfig.maxIterations) {
      finishOptimization();
      return;
    }

    if (currentMode === 0) {
      $('#current-mode').text('Sigmoid Transform');
      const transformedSignal = sigmoidTransform(currentSignal, algoConfig.sigmoidFactor);
      fftResult = math.fft(transformedSignal);
      currentPhases = math.map(uniqueFreqBinIndices, i => Math.atan2(fftResult[i].im, fftResult[i].re));
	 // for(i =0; i < uniqueFreqBinIndices.length; i++) currentPhases[i] = Math.atan2(fftResult[uniqueFreqBinIndices[i]].im, fftResult[uniqueFreqBinIndices[i]].re);
    } else {
		
		$('#current-mode').text('Gauss-Newton');
		
		
		
		
		
		
		const q = math.divide(algoConfig.pNorm, 2);
		const residual =  math.map(currentSignal,x=>math.pow(x,q));
		//const residual = currentSignal.map(x => math.map(x, x1 => math.pow(x1,q)));
		const jacobian = phaseDerivative(amplitudeVector, currentPhases, currentSignal, timeVector, q, uniqueFrequencies);
		//const jacobian = numericalJacobian(x=>math.map(x,x1=>math.pow(x1,q)), currentSignal);
		
		 
    try {
        const JT = math.transpose(jacobian);
        const JTJ = math.multiply(JT, jacobian);
        const JTr = math.multiply(JT, residual);
        const update = math.multiply(math.pinv(JTJ), JTr); //math.lusolve(JTJ, JTr);
        
		
        currentPhases = currentPhases.map((phase, i) => phase - update._data[i]);
    } catch (error) {
        console.error('Error in Gauss-Newton step:', error);
        currentMode = 0;
    }
         
        }
   // currentSignal = generateMultisineSignal(amplitudeVector, uniqueFrequencies, currentPhases, timeVector);

    const currentCrestFactor = calculateCrestFactor(currentSignal);
    $('#current-crest-factor').text(currentCrestFactor.toFixed(4));
    $('#best-crest-factor').text(bestCrestFactor.toFixed(4));
    $('#current-iter').text((1+iteration) + "/" + algoConfig.maxIterations);

    let phasesText = '';
    for (let i = 0; i < uniqueFrequencies.length; i++) {
      phasesText += `Frequency: ${uniqueFrequencies[i].toFixed(2)} Hz, Phase: ${currentPhases[i].toFixed(4)} rad\n`;
    }
    $('#current-phases').text(phasesText);

    const relativeImprovement = Math.abs(previousCrestFactor - currentCrestFactor) / previousCrestFactor;

    modeCounter++;

    if (currentCrestFactor < bestCrestFactor) {
      bestCrestFactor = currentCrestFactor;
      bestPhases = currentPhases.slice();
      stagnationCounter = 0;
    } else {
      stagnationCounter++;
    }

	//HACK: Gauss Newton only 10% iterations stagnation
	if ((stagnationCounter >= algoConfig.maxIterPerModeStag*0.1) && (currentMode == 0)) {
	  currentMode = 1 - currentMode;
      stagnationCounter = 0;
      modeCounter = 0;
      previousCrestFactor = Infinity;
	}


    if (stagnationCounter >= algoConfig.maxIterPerModeStag || modeCounter > algoConfig.maxIterPerMode) {
      currentMode = 1 - currentMode;
      stagnationCounter = 0;
      modeCounter = 0;
      previousCrestFactor = Infinity;
    } else if (relativeImprovement < 1e-6 && currentMode === 1) {
      currentMode = 0;
      stagnationCounter = 0;
      modeCounter = 0;
      previousCrestFactor = Infinity;
    } else {
      previousCrestFactor = currentCrestFactor;
    }


    if (algoConfig.animate && iteration % algoConfig.updateEvery === 0) {
		
    Plotly.extendTraces('crest-factor-plot', {x : [[iteration]]   ,y: [[currentCrestFactor]]}, [0]);
	
	
      Plotly.animate('time-domain-plot', {
        data: [
          {y: initialSignal},
          {y: currentSignal}
        ],
        traces: [0, 1],
        layout: {}
      }, {
        transition: {duration: 0},
        frame: {duration: 0, redraw: false}
      });
    }

    optimizationTimeout = setTimeout(() => optimizationStep(iteration + 1), 0);
	
  }

  function finishOptimization() {
    const optimizedSignal = generateMultisineSignal(amplitudeVector, uniqueFrequencies, bestPhases, timeVector);
    const finalCrestFactor = calculateCrestFactor(optimizedSignal);

    $('#final-crest-factor').text(finalCrestFactor.toFixed(4));
    $('#improvement').text(`${((initialCrestFactor - finalCrestFactor) / initialCrestFactor * 100).toFixed(2)}%`);

    let phasesText = '';
    for (let i = 0; i < uniqueFrequencies.length; i++) {
      phasesText += `Frequency: ${uniqueFrequencies[i].toFixed(2)} Hz, Phase: ${bestPhases[i].toFixed(4)} rad\n`;
    }
    $('#final-phases').text(phasesText);

    Plotly.animate('time-domain-plot', {
      data: [
        {y: initialSignal},
        {y: optimizedSignal}
      ],
      traces: [0, 1],
      layout: {}
    }, {
      transition: {duration: 0},
      frame: {duration: 0, redraw: false}
    });

    $('#stop-btn').hide();
    $('#optimize-btn').show();
	
    enableConfigurations();
    isOptimizing = false;
  }

  optimizationStep(0);
}

function stopOptimizationProcess() {
	stopOptimization = true;
	finishOptimization();
 /* stopOptimization = true;
  if (optimizationTimeout) {
    clearTimeout(optimizationTimeout);
  }
  isOptimizing = false;
  $('#stop-btn').hide();
  $('#optimize-btn').show();*/
}

$('#optimize-btn').click(optimize);
$('#stop-btn').click(stopOptimizationProcess);