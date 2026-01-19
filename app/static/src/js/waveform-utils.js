/**
 * EHR-style Waveform Generators
 * Shared utility functions for generating realistic vital sign waveforms
 * Used by vital.html and vital-prediction.html
 */

/**
 * Generate heart rate waveform with realistic oscillation
 * @param {number} t - Time variable
 * @param {number} baseline - Baseline heart rate from MongoDB
 * @returns {number} - Generated heart rate value
 */
function generateHeartRateWaveform(t, baseline) {
    const amplitude = 5;
    const frequency = 0.1;
    const smoothFactor = 0.02;
    const noise = Math.random() * 0.5 - 0.25;

    const heartRate = baseline + amplitude * Math.sin(frequency * t) + noise;
    const smoothHeartRate = heartRate * (1 - smoothFactor) + baseline * smoothFactor;
    return Math.min(Math.max(smoothHeartRate, 50), 120);
}

/**
 * Generate oxygen saturation waveform with realistic variation
 * @param {number} t - Time variable
 * @param {number} baseline - Baseline SpO2 from MongoDB
 * @returns {number} - Generated SpO2 value
 */
function generateOxygenSaturationWaveform(t, baseline) {
    const amplitude = 2;
    const frequency = 0.2;
    const noise = Math.random() * 0.5 - 0.25;
    return baseline + amplitude * Math.sin(frequency * t + Math.PI / 2) + noise;
}

/**
 * Generate respiratory rate waveform with realistic breathing pattern
 * @param {number} t - Time variable
 * @param {number} baseline - Baseline RR from MongoDB
 * @returns {number} - Generated respiratory rate value
 */
function generateRespiratoryRateWaveform(t, baseline) {
    const amplitude = 2;
    const frequency = 0.1;
    const noise = Math.random() * 1 - 0.5;
    return baseline + amplitude * Math.sin(frequency * t) + noise;
}

// Export for module systems (if used)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        generateHeartRateWaveform,
        generateOxygenSaturationWaveform,
        generateRespiratoryRateWaveform
    };
}
