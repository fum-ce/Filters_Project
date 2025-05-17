## Loading and Processing Voice Signals

To add voice signal processing, you'll need to:

1. Load an audio file
2. Apply your existing filters to the audio data
3. Play back or save the filtered audio

Here's a step-by-step implementation approach:

### Step 1: Install and Import Required Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import IPython.display as ipd  # For playing audio in notebooks
```

### Step 2: Function to Load Voice Signals

```python
def load_voice_signal(file_path):
    """Load a voice signal from a WAV file."""
    sample_rate, signal = wavfile.read(file_path)
    
    # If stereo, convert to mono by averaging channels
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    # Normalize signal to range [-1, 1]
    signal = signal.astype(float)
    if signal.max() > 0:
        signal /= np.max(np.abs(signal))
        
    return sample_rate, signal
```

### Step 3: Function to Save Filtered Voice Signals

```python
def save_filtered_audio(signal, sample_rate, file_path):
    """Save a filtered audio signal to a WAV file."""
    # Ensure the signal is in the right range for WAV files
    signal = np.clip(signal, -1, 1)
    signal = (signal * 32767).astype(np.int16)
    
    wavfile.write(file_path, sample_rate, signal)
```

### Step 4: Function to Apply a Filter to a Voice Signal and Compare

```python
def apply_filter_to_voice(signal, sample_rate, filter_func, **filter_kwargs):
    """Apply a filter to a voice signal and return the filtered signal."""
    filtered_signal = filter_func(signal, **filter_kwargs)
    
    # Ensure the filtered signal has the same length as the original
    if len(filtered_signal) != len(signal):
        # If needed, pad or truncate
        if len(filtered_signal) < len(signal):
            filtered_signal = np.pad(filtered_signal, 
                                    (0, len(signal) - len(filtered_signal)), 
                                    'constant')
        else:
            filtered_signal = filtered_signal[:len(signal)]
    
    return filtered_signal
```

### Step 5: Function to Plot and Compare Original vs Filtered Voice

```python
def plot_voice_comparison(original_signal, filtered_signal, sample_rate, title="Voice Signal Comparison"):
    """Plot original and filtered voice signals for comparison."""
    duration = len(original_signal) / sample_rate
    time = np.linspace(0, duration, len(original_signal))
    
    plt.figure(figsize=(14, 8))
    
    # Original signal plot
    plt.subplot(2, 1, 1)
    plt.plot(time, original_signal)
    plt.title('Original Voice Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Filtered signal plot
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_signal)
    plt.title(f'Filtered Voice Signal - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Also plot spectrograms for frequency domain comparison
    plt.figure(figsize=(14, 8))
    
    # Original signal spectrogram
    plt.subplot(2, 1, 1)
    plt.specgram(original_signal, Fs=sample_rate, cmap='viridis')
    plt.title('Original Voice Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Filtered signal spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(filtered_signal, Fs=sample_rate, cmap='viridis')
    plt.title(f'Filtered Voice Spectrogram - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()
```

### Step 6: Example Usage with the Existing Filters

Here's how you can use your previously implemented filters with voice data:

```python
# Assuming you have a WAV file with voice data
file_path = 'voice_sample.wav'  # Change to your file path
sample_rate, voice_signal = load_voice_signal(file_path)

# Let's try a few of the filters
# 1. Moving Average Filter
filtered_voice_ma = apply_filter_to_voice(voice_signal, sample_rate, 
                                          moving_average_filter, window_size=101)
plot_voice_comparison(voice_signal, filtered_voice_ma, sample_rate, 
                      title="Moving Average Filter")

# Play the original and filtered signals
print("Playing original voice signal:")
ipd.display(ipd.Audio(voice_signal, rate=sample_rate))
print("Playing filtered voice signal:")
ipd.display(ipd.Audio(filtered_voice_ma, rate=sample_rate))

# 2. Low-Pass Filter (Good for removing high-frequency noise)
filtered_voice_lp = apply_filter_to_voice(voice_signal, sample_rate, 
                                         fir_low_pass_filter, 
                                         cutoff_frequency=1000, 
                                         sample_rate=sample_rate, 
                                         filter_length=101)
plot_voice_comparison(voice_signal, filtered_voice_lp, sample_rate, 
                      title="Low-Pass Filter")

# 3. Gaussian Filter
filtered_voice_gauss = apply_filter_to_voice(voice_signal, sample_rate, 
                                            gaussian_filter, 
                                            sigma=5, window_size=51)
plot_voice_comparison(voice_signal, filtered_voice_gauss, sample_rate, 
                      title="Gaussian Filter")

# Save the filtered audio
save_filtered_audio(filtered_voice_lp, sample_rate, 'voice_filtered_lowpass.wav')
```

### Step 7: Creating a Voice Filter Comparison Dashboard

You can create a function to compare all your filters on a single voice sample:

```python
def compare_all_filters_on_voice(voice_signal, sample_rate):
    """Apply all filters to a voice signal and compare results."""
    # Dictionary to store all filtered signals
    filtered_signals = {}
    
    # Apply each filter
    filtered_signals['Moving Average'] = apply_filter_to_voice(
        voice_signal, sample_rate, moving_average_filter, window_size=101)
    
    filtered_signals['Low-Pass'] = apply_filter_to_voice(
        voice_signal, sample_rate, fir_low_pass_filter, 
        cutoff_frequency=1000, sample_rate=sample_rate, filter_length=101)
    
    filtered_signals['High-Pass'] = apply_filter_to_voice(
        voice_signal, sample_rate, fir_high_pass_filter,
        cutoff_frequency=500, sample_rate=sample_rate, filter_length=101)
    
    filtered_signals['Band-Pass'] = apply_filter_to_voice(
        voice_signal, sample_rate, fir_band_pass_filter,
        low_cutoff=300, high_cutoff=3000, sample_rate=sample_rate, filter_length=101)
    
    filtered_signals['Gaussian'] = apply_filter_to_voice(
        voice_signal, sample_rate, gaussian_filter,
        sigma=5, window_size=51)
    
    filtered_signals['Median'] = apply_filter_to_voice(
        voice_signal, sample_rate, median_filter,
        window_size=31)
    
    filtered_signals['Box'] = apply_filter_to_voice(
        voice_signal, sample_rate, box_filter,
        window_size=51)
    
    filtered_signals['Savitzky-Golay'] = apply_filter_to_voice(
        voice_signal, sample_rate, savitzky_golay_filter,
        window_size=51, poly_order=3)
    
    # Plot time domain comparison
    plt.figure(figsize=(15, 12))
    duration = len(voice_signal) / sample_rate
    time = np.linspace(0, duration, len(voice_signal))
    
    plt.subplot(len(filtered_signals) + 1, 1, 1)
    plt.plot(time, voice_signal)
    plt.title('Original Voice Signal')
    plt.ylabel('Amplitude')
    
    for i, (filter_name, filtered_signal) in enumerate(filtered_signals.items(), 2):
        plt.subplot(len(filtered_signals) + 1, 1, i)
        plt.plot(time, filtered_signal)
        plt.title(f'{filter_name} Filter')
        plt.ylabel('Amplitude')
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    return filtered_signals
```

### Getting Sample Voice Files

If you don't already have voice samples, you can:

1. Record your own using a microphone and software like Audacity
2. Download free samples from online libraries like Freesound (https://freesound.org/)
3. Use synthetic speech generation for testing

### Additional Voice Processing Features

You might also want to:

1. Implement real-time voice processing if needed
2. Add voice-specific metrics like Signal-to-Noise Ratio (SNR) for comparing filter effectiveness
3. Add voice-specific filters like noise gates or de-essers

This implementation should integrate well with your existing project structure and allow you to extend your filter applications to voice signals. Let me know if you need any clarification or additional features!