import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy import signal


class AdvancedAudioDataLoader:
    """
    Advanced audio data loader with augmentations and event-background mixing.
    Supports separate event and background CSV files.
    """
    
    def __init__(self, event_csv_path, background_csv_path, duration=3.0, sample_rate=16000, 
                 classification_mode='sigmoid', augmentation_config=None, pairing_mode='random'):
        """
        Args:
            event_csv_path: Path to CSV with columns: event_path, label (snoring=1, other=0)
            background_csv_path: Path to CSV with column: background_path
            duration: Target audio duration in seconds
            sample_rate: Audio sampling rate
            classification_mode: 'sigmoid' or 'softmax'
            augmentation_config: Dict with augmentation parameters
            pairing_mode: 'random' (pair randomly) or 'sequential' (pair in order)
        """
        self.event_csv_path = event_csv_path
        self.background_csv_path = background_csv_path
        self.duration = duration
        self.sample_rate = sample_rate
        self.target_length = int(duration * sample_rate)
        self.classification_mode = classification_mode
        self.pairing_mode = pairing_mode
        
        # Load event and background data
        self.event_df = pd.read_csv(event_csv_path)
        self.background_df = pd.read_csv(background_csv_path)
        
        # Validate data
        if 'event_path' not in self.event_df.columns or 'label' not in self.event_df.columns:
            raise ValueError("Event CSV must have 'event_path' and 'label' columns")
        if 'background_path' not in self.background_df.columns:
            raise ValueError("Background CSV must have 'background_path' column")
        
        # Get number of samples (based on events)
        self.num_samples = len(self.event_df)
        self.num_backgrounds = len(self.background_df)
        
        print(f"Loaded {self.num_samples} events and {self.num_backgrounds} backgrounds")
        print(f"Class distribution: Snoring={sum(self.event_df['label'] == 1)}, Other={sum(self.event_df['label'] == 0)}")
        
        # Set up augmentation config
        self.aug_config = augmentation_config or self._get_default_aug_config()
        
        # Initialize augmentation functions
        self._init_augmentations()
    
    def _get_default_aug_config(self):
        """Default augmentation configuration."""
        return {
            'gain': {'prob': 0.5, 'min_gain': 0.5, 'max_gain': 2.0},
            'polarity_inversion': {'prob': 0.3},
            'impulse_response': {'prob': 0.4, 'ir_paths': []},
            'colored_noise': {'prob': 0.3, 'noise_types': ['white', 'pink'], 'snr_db': [10, 30]},
            'high_pass_filter': {'prob': 0.3, 'cutoff_freq': [20, 500]},
            'peak_normalization': {'prob': 0.4, 'target_peak': 0.9},
            'pitch_shift': {'prob': 0.3, 'semitones': [-2, 2]},
            'time_shift': {'prob': 0.4, 'max_shift_ratio': 0.1},
            'mixing': {'snr_range_db': [-5, 20]}
        }
    
    def _init_augmentations(self):
        """Initialize augmentation functions."""
        self.augmentations = {
            'gain': self._apply_gain,
            'polarity_inversion': self._apply_polarity_inversion,
            'impulse_response': self._apply_impulse_response,
            'colored_noise': self._apply_colored_noise,
            'high_pass_filter': self._apply_high_pass_filter,
            'peak_normalization': self._apply_peak_normalization,
            'pitch_shift': self._apply_pitch_shift,
            'time_shift': self._apply_time_shift
        }
    
    def _load_audio(self, file_path):
        """Load audio file and convert to target length."""
        # Handle tensor input
        if isinstance(file_path, tf.Tensor):
            file_path = file_path.numpy().decode('utf-8')
        
        # Load audio
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        audio = tf.squeeze(audio, axis=-1)
        
        # Pad or trim
        current_length = tf.shape(audio)[0]
        if current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            audio = tf.pad(audio, [[0, padding]])
        elif current_length > self.target_length:
            # Random crop only if audio is longer than target
            max_start = current_length - self.target_length
            if max_start > 0:
                start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
                audio = audio[start:start + self.target_length]
            else:
                audio = audio[:self.target_length]
        # If current_length == self.target_length, return as is
        
        return audio
    
    # Augmentation functions (same as before)
    def _apply_gain(self, audio, config):
        """Apply random gain."""
        if tf.random.uniform([]) < config['prob']:
            gain = tf.random.uniform([], config['min_gain'], config['max_gain'])
            audio = audio * gain
        return audio
    
    def _apply_polarity_inversion(self, audio, config):
        """Invert polarity randomly."""
        if tf.random.uniform([]) < config['prob']:
            audio = -audio
        return audio
    
    def _apply_impulse_response(self, audio, config):
        """Convolve with impulse response."""
        if tf.random.uniform([]) < config['prob'] and config['ir_paths']:
            # Load random IR
            ir_path = np.random.choice(config['ir_paths'])
            ir = self._load_audio(ir_path)
            
            # Convolve using FFT
            audio_fft = tf.signal.fft(tf.cast(audio, tf.complex64))
            ir_fft = tf.signal.fft(tf.cast(ir, tf.complex64))
            convolved_fft = audio_fft * ir_fft
            convolved = tf.real(tf.signal.ifft(convolved_fft))
            
            # Normalize
            max_val = tf.reduce_max(tf.abs(convolved))
            if max_val > 1e-8:
                audio = convolved / max_val
            else:
                audio = convolved
        return audio
    
    def _apply_colored_noise(self, audio, config):
        """Add colored noise."""
        if tf.random.uniform([]) < config['prob']:
            noise_type = np.random.choice(config['noise_types'])
            snr_db = tf.random.uniform([], config['snr_db'][0], config['snr_db'][1])
            
            # Generate noise
            if noise_type == 'white':
                noise = tf.random.normal(tf.shape(audio))
            elif noise_type == 'pink':
                # Simplified pink noise generation
                white = tf.random.normal(tf.shape(audio))
                # Apply 1/f filter in frequency domain
                fft = tf.signal.rfft(white)
                freqs = tf.range(tf.shape(fft)[0], dtype=tf.float32) + 1.0
                pink_filter = 1.0 / tf.sqrt(freqs)
                pink_fft = fft * tf.cast(pink_filter, tf.complex64)
                noise = tf.signal.irfft(pink_fft, fft_length=[self.target_length])
            else:
                noise = tf.random.normal(tf.shape(audio))
            
            # Apply SNR
            signal_power = tf.reduce_mean(tf.square(audio))
            noise_power = tf.reduce_mean(tf.square(noise))
            if noise_power > 1e-8:
                noise_scale = tf.sqrt(signal_power / (noise_power * tf.pow(10.0, snr_db / 10.0)))
                audio = audio + noise * noise_scale
        
        return audio
    
    def _apply_high_pass_filter(self, audio, config):
        """Apply high-pass filter."""
        if tf.random.uniform([]) < config['prob']:
            cutoff = tf.random.uniform([], config['cutoff_freq'][0], config['cutoff_freq'][1])
            
            # Simple butterworth filter implementation
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            # Create filter coefficients (simplified)
            # In practice, you'd use scipy.signal.butter
            b = tf.constant([0.95, -0.95], dtype=tf.float32)
            a = tf.constant([1.0, -0.9], dtype=tf.float32)
            
            # Apply filter
            audio = tf.nn.conv1d(
                tf.expand_dims(tf.expand_dims(audio, 0), -1),
                tf.expand_dims(tf.expand_dims(b, -1), -1),
                stride=1,
                padding='SAME'
            )
            audio = tf.squeeze(audio)
        
        return audio
    
    def _apply_peak_normalization(self, audio, config):
        """Normalize to target peak."""
        if tf.random.uniform([]) < config['prob']:
            current_peak = tf.reduce_max(tf.abs(audio))
            if current_peak > 1e-8:
                audio = audio * (config['target_peak'] / current_peak)
        return audio
    
    def _apply_pitch_shift(self, audio, config):
        """Apply pitch shift (simplified version)."""
        if tf.random.uniform([]) < config['prob']:
            semitones = tf.random.uniform([], config['semitones'][0], config['semitones'][1])
            shift_factor = tf.pow(2.0, semitones / 12.0)
            
            # Resample (simplified - in practice use librosa or similar)
            new_length = tf.cast(tf.cast(self.target_length, tf.float32) / shift_factor, tf.int32)
            new_length = tf.maximum(new_length, 1)  # Ensure at least 1 sample
            
            # Create indices safely
            if new_length > 1:
                indices = tf.linspace(0.0, tf.cast(self.target_length - 1, tf.float32), new_length)
            else:
                indices = tf.constant([0.0])
            
            indices = tf.clip_by_value(tf.cast(indices, tf.int32), 0, self.target_length - 1)
            audio = tf.gather(audio, indices)
            
            # Pad or trim back to original length
            current_length = tf.shape(audio)[0]
            if current_length < self.target_length:
                audio = tf.pad(audio, [[0, self.target_length - current_length]])
            else:
                audio = audio[:self.target_length]
        
        return audio
    
    def _apply_time_shift(self, audio, config):
        """Apply time shift."""
        if tf.random.uniform([]) < config['prob']:
            max_shift = int(self.target_length * config['max_shift_ratio'])
            if max_shift > 0:
                shift = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
                audio = tf.roll(audio, shift, axis=0)
        return audio
    
    def _apply_augmentations(self, audio, aug_names):
        """Apply selected augmentations."""
        for aug_name in aug_names:
            if aug_name in self.augmentations and aug_name in self.aug_config:
                audio = self.augmentations[aug_name](audio, self.aug_config[aug_name])
        return audio
    
    def _mix_audio(self, event_audio, background_audio):
        """Mix event and background with random SNR."""
        snr_range = self.aug_config['mixing']['snr_range_db']
        snr_db = tf.random.uniform([], snr_range[0], snr_range[1])
        
        # Calculate mixing weights
        event_power = tf.reduce_mean(tf.square(event_audio))
        background_power = tf.reduce_mean(tf.square(background_audio))
        
        # Avoid division by zero
        background_power = tf.maximum(background_power, 1e-8)
        event_power = tf.maximum(event_power, 1e-8)
        
        # Calculate scale factor for background
        snr_linear = tf.pow(10.0, snr_db / 10.0)
        background_scale = tf.sqrt(event_power / (background_power * snr_linear))
        
        # Mix
        mixed = event_audio + background_audio * background_scale
        
        # Prevent clipping
        max_val = tf.reduce_max(tf.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / (max_val + 1e-8)
        
        return mixed
    
    def _get_background_index(self, event_index):
        """Get background index based on pairing mode."""
        if self.pairing_mode == 'random':
            # Random pairing
            return np.random.randint(0, self.num_backgrounds)
        else:
            # Sequential pairing with wraparound
            return event_index % self.num_backgrounds
    
    def _process_sample(self, event_index):
        """Process a single sample."""
        # Convert tensor input
        if isinstance(event_index, tf.Tensor):
            event_index = event_index.numpy()
        
        # Get event info
        event_row = self.event_df.iloc[event_index]
        event_path = event_row['event_path']
        label = event_row['label']
        
        # Get background path (random or sequential pairing)
        bg_index = self._get_background_index(event_index)
        background_path = self.background_df.iloc[bg_index]['background_path']
        
        # Load audio files
        event_audio = self._load_audio(event_path)
        background_audio = self._load_audio(background_path)
        
        # Apply augmentations
        aug_names = list(self.augmentations.keys())
        event_audio = self._apply_augmentations(event_audio, aug_names)
        background_audio = self._apply_augmentations(background_audio, aug_names)
        
        # Mix audio
        mixed_audio = self._mix_audio(event_audio, background_audio)
        
        # Handle label based on classification mode
        if self.classification_mode == 'sigmoid':
            # Binary classification: snoring (1) vs other (0)
            output_label = tf.cast(label, tf.float32)
        else:
            # Multi-class (even though we only have 2 classes)
            output_label = tf.cast(label, tf.float32)
        
        return mixed_audio, output_label
    
    def create_dataset(self, batch_size=32, shuffle=True):
        """Create TensorFlow dataset."""
        # Create dataset from event indices
        indices = np.arange(self.num_samples)
        dataset = tf.data.Dataset.from_tensor_slices(indices)
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(1000)
        
        # Map processing function
        dataset = dataset.map(
            lambda idx: tf.py_function(
                self._process_sample,
                [idx],
                [tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes
        if self.classification_mode == 'sigmoid':
            dataset = dataset.map(
                lambda audio, label: (
                    tf.ensure_shape(audio, [self.target_length]),
                    tf.ensure_shape(label, [])
                )
            )
        else:
            dataset = dataset.map(
                lambda audio, label: (
                    tf.ensure_shape(audio, [self.target_length]),
                    tf.cast(label, tf.int32)  # Ensure int32 for softmax
                )
            )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_info(self):
        """Get dataset information."""
        return {
            'num_events': self.num_samples,
            'num_backgrounds': self.num_backgrounds,
            'num_snoring': sum(self.event_df['label'] == 1),
            'num_other': sum(self.event_df['label'] == 0),
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'pairing_mode': self.pairing_mode
        }


# Usage example
def create_train_val_loaders(train_event_csv, train_bg_csv, val_event_csv, val_bg_csv, 
                            batch_size=32, pairing_mode='random', **kwargs):
    """Create training and validation data loaders with separate event and background CSVs."""
    
    # Training loader with augmentation
    train_loader = AdvancedAudioDataLoader(
        train_event_csv,
        train_bg_csv,
        pairing_mode=pairing_mode,
        **kwargs
    )
    train_dataset = train_loader.create_dataset(batch_size=batch_size, shuffle=True)
    
    # Validation loader without augmentation
    val_aug_config = {
        'gain': {'prob': 0.0},
        'polarity_inversion': {'prob': 0.0},
        'impulse_response': {'prob': 0.0},
        'colored_noise': {'prob': 0.0},
        'high_pass_filter': {'prob': 0.0},
        'peak_normalization': {'prob': 0.0},
        'pitch_shift': {'prob': 0.0},
        'time_shift': {'prob': 0.0},
        'mixing': {'snr_range_db': [10, 10]}  # Fixed SNR for validation
    }
    
    val_loader = AdvancedAudioDataLoader(
        val_event_csv,
        val_bg_csv,
        augmentation_config=val_aug_config,
        pairing_mode='sequential',  # Use sequential pairing for validation for consistency
        **kwargs
    )
    val_dataset = val_loader.create_dataset(batch_size=batch_size, shuffle=False)
    
    return {
        'train': {'loader': train_loader, 'dataset': train_dataset},
        'val': {'loader': val_loader, 'dataset': val_dataset}
    }

if __name__ == "__main__":
  # Create data loaders
  loaders = create_train_val_loaders(
      train_event_csv='train_event.csv',
      train_bg_csv='background_train.csv', 
      val_event_csv='val_event.csv',
      val_bg_csv='background_val.csv',
      batch_size=32,
      pairing_mode='random',  # or 'sequential'
      duration=3.0,
      sample_rate=16000,
      classification_mode='sigmoid'  # for binary classification
  )
  
  # Access datasets
  train_dataset = loaders['train']['dataset']
  val_dataset = loaders['val']['dataset']
