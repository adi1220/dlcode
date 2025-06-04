import tensorflow as tf
import numpy as np
import os


class AudioLoader:
    """
    Simplified audio data loader for classification tasks.
    """
    
    def __init__(self, data_file_path, duration=3.0, sample_rate=16000, 
                 augment=True, standardize=True):
        """
        Args:
            data_file_path: Path to text file with "audio_path,label" format
            duration: Target audio duration in seconds
            sample_rate: Audio sampling rate
            augment: Whether to apply data augmentation
            standardize: Whether to standardize audio data
        """
        self.data_file_path = data_file_path
        self.duration = duration
        self.sample_rate = sample_rate
        self.target_length = int(duration * sample_rate)
        self.augment = augment
        self.standardize = standardize
        
        # Load data and create class mapping
        self.audio_paths, self.labels, self.class_mapping = self._load_data()
        
        # Calculate statistics for standardization
        if self.standardize:
            self._calculate_stats()
    
    def _load_data(self):
        """Load audio paths and labels, create class mapping."""
        audio_paths = []
        label_strings = []
        
        # Read file
        with open(self.data_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    path, label = line.split(',', 1)
                    audio_paths.append(path.strip())
                    label_strings.append(label.strip())
        
        # Create class mapping
        unique_labels = sorted(set(label_strings))
        class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert to numeric labels
        labels = [class_mapping[label] for label in label_strings]
        
        print(f"Loaded {len(audio_paths)} samples with {len(unique_labels)} classes")
        print(f"Class mapping: {class_mapping}")
        
        return audio_paths, labels, class_mapping
    
    def _calculate_stats(self):
        """Calculate mean and std for standardization."""
        # Sample a subset for efficiency
        sample_size = min(50, len(self.audio_paths))
        sample_indices = np.random.choice(len(self.audio_paths), sample_size, replace=False)
        
        all_audio = []
        for idx in sample_indices:
            try:
                audio = self._load_audio(self.audio_paths[idx])
                all_audio.append(audio.numpy())
            except:
                continue
        
        if all_audio:
            all_audio = np.concatenate(all_audio)
            self.mean = np.mean(all_audio)
            self.std = np.std(all_audio) + 1e-8
        else:
            self.mean = 0.0
            self.std = 1.0
        
        print(f"Audio stats - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
    
    def _load_audio(self, file_path):
        """Load and preprocess audio file."""
        if isinstance(file_path, tf.Tensor):
            file_path = file_path.numpy().decode('utf-8')
        
        # Load audio file
        if file_path.endswith('.wav'):
            audio_binary = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1)
        elif file_path.endswith('.pcm'):
            audio_binary = tf.io.read_file(file_path)
            audio = tf.audio.decode_raw(audio_binary, tf.int16)
            audio = tf.cast(audio, tf.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported format: {file_path}")
        
        # Pad or trim to target length
        current_length = tf.shape(audio)[0]
        if current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            audio = tf.pad(audio, [[0, padding]])
        else:
            # Trim to target length
            audio = audio[:self.target_length]
        
        return audio
    
    def _augment_audio(self, audio):
        """Apply simple data augmentation."""
        if not self.augment:
            return audio
        
        # Random volume scaling (0.8 to 1.2)
        volume_scale = tf.random.uniform([], 0.8, 1.2)
        audio = audio * volume_scale
        
        # Add small amount of noise
        noise = tf.random.normal(tf.shape(audio), stddev=0.005)
        audio = audio + noise
        
        return audio
    
    def _process_sample(self, file_path, label):
        """Process a single audio sample."""
        # Load audio
        audio = self._load_audio(file_path)
        
        # Apply augmentation
        audio = self._augment_audio(audio)
        
        # Standardize
        if self.standardize:
            audio = (audio - self.mean) / self.std
        
        return audio, label
    
    def create_dataset(self, batch_size=32, shuffle=True):
        """Create TensorFlow dataset."""
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((self.audio_paths, self.labels))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(1000)
        
        # Map processing function
        dataset = dataset.map(
            lambda path, label: tf.py_function(
                self._process_sample, [path, label], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes
        dataset = dataset.map(
            lambda audio, label: (
                tf.ensure_shape(audio, [self.target_length]),
                tf.ensure_shape(label, [])
            )
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_info(self):
        """Get dataset information."""
        return {
            'num_samples': len(self.audio_paths),
            'num_classes': len(self.class_mapping),
            'class_mapping': self.class_mapping,
            'duration': self.duration,
            'sample_rate': self.sample_rate
        }


def create_loaders(train_file, val_file=None, test_file=None, **kwargs):
    """Create train/val/test loaders."""
    loaders = {}
    
    # Training loader (with augmentation)
    train_loader = AudioLoader(train_file, augment=True, **kwargs)
    loaders['train'] = {
        'loader': train_loader,
        'dataset': train_loader.create_dataset(shuffle=True),
        'info': train_loader.get_info()
    }
    
    # Validation loader (no augmentation)
    if val_file:
        val_loader = AudioLoader(val_file, augment=False, **kwargs)
        # Use training stats for consistency
        if hasattr(train_loader, 'mean'):
            val_loader.mean = train_loader.mean
            val_loader.std = train_loader.std
        
        loaders['val'] = {
            'loader': val_loader,
            'dataset': val_loader.create_dataset(shuffle=False),
            'info': val_loader.get_info()
        }
    
    # Test loader (no augmentation)
    if test_file:
        test_loader = AudioLoader(test_file, augment=False, **kwargs)
        # Use training stats for consistency
        if hasattr(train_loader, 'mean'):
            test_loader.mean = train_loader.mean
            test_loader.std = train_loader.std
        
        loaders['test'] = {
            'loader': test_loader,
            'dataset': test_loader.create_dataset(shuffle=False),
            'info': test_loader.get_info()
        }
    
    return loaders


# Example usage
if __name__ == '__main__':
    # Create dummy test data
    import tempfile
    import wave
    
    def create_dummy_wav(filepath, duration=2.0, sample_rate=16000):
        """Create a dummy WAV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        samples = int(duration * sample_rate)
        audio_data = np.random.uniform(-0.5, 0.5, samples).astype(np.float32)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    # Create test files
    temp_dir = tempfile.mkdtemp()
    test_files = [
        (os.path.join(temp_dir, 'audio1.wav'), 'snoring'),
        (os.path.join(temp_dir, 'audio2.wav'), 'normal'),
        (os.path.join(temp_dir, 'audio3.wav'), 'snoring'),
        (os.path.join(temp_dir, 'audio4.wav'), 'normal')
    ]
    
    # Create audio files and data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for filepath, label in test_files:
            create_dummy_wav(filepath)
            f.write(f"{filepath},{label}\n")
        data_file = f.name
    
    try:
        # Test the loader
        print("Testing AudioLoader...")
        loader = AudioLoader(data_file, duration=2.0, sample_rate=16000)
        
        # Print info
        info = loader.get_info()
        print(f"Samples: {info['num_samples']}")
        print(f"Classes: {info['class_mapping']}")
        
        # Create dataset
        dataset = loader.create_dataset(batch_size=2)
        
        # Test one batch
        for batch_audio, batch_labels in dataset.take(1):
            print(f"Batch audio shape: {batch_audio.shape}")
            print(f"Batch labels: {batch_labels.numpy()}")
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Cleanup
        try:
            os.unlink(data_file)
            for filepath, _ in test_files:
                if os.path.exists(filepath):
                    os.unlink(filepath)
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
  # Simple usage
  loader = AudioLoader('data.txt', duration=3.0, sample_rate=16000)
  dataset = loader.create_dataset(batch_size=32)
  
  # Multiple datasets
  loaders = create_loaders(
      train_file='train.txt',
      val_file='val.txt', 
      duration=2.0,
      sample_rate=16000
  )
  train_dataset = loaders['train']['dataset']
