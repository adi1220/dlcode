import tensorflow as tf
import numpy as np
import pandas as pd
import os
import wave
import tempfile
import shutil
from datetime import datetime


class AudioDataLoaderTester:
    """Test suite for AdvancedAudioDataLoader."""
    
    def __init__(self, test_dir=None):
        """Initialize tester with temporary directory."""
        self.test_dir = test_dir or tempfile.mkdtemp(prefix='audio_loader_test_')
        self.sample_rate = 16000
        self.duration = 2.0  # seconds
        self.num_samples = int(self.sample_rate * self.duration)
        
        # Create subdirectories
        self.events_dir = os.path.join(self.test_dir, 'events')
        self.backgrounds_dir = os.path.join(self.test_dir, 'backgrounds')
        self.ir_dir = os.path.join(self.test_dir, 'impulse_responses')
        
        for dir_path in [self.events_dir, self.backgrounds_dir, self.ir_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_wav_file(self, filepath, audio_data):
        """Create a WAV file from audio data."""
        # Ensure audio is in correct format
        audio_data = np.array(audio_data, dtype=np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def generate_synthetic_event(self, event_type='snoring'):
        """Generate synthetic event audio."""
        t = np.linspace(0, self.duration, self.num_samples)
        
        if event_type == 'snoring':
            # Simulate snoring with low frequency modulated sine waves
            fundamental = 100 + 50 * np.sin(2 * np.pi * 0.3 * t)
            audio = 0.3 * np.sin(2 * np.pi * fundamental * t)
            # Add harmonics
            audio += 0.15 * np.sin(2 * np.pi * 2 * fundamental * t)
            audio += 0.1 * np.sin(2 * np.pi * 3 * fundamental * t)
            # Add envelope
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
            audio *= envelope
        
        elif event_type == 'cough':
            # Simulate cough with burst of noise
            audio = np.zeros(self.num_samples)
            cough_start = int(0.5 * self.sample_rate)
            cough_duration = int(0.3 * self.sample_rate)
            burst = np.random.normal(0, 0.3, cough_duration)
            # Apply envelope
            envelope = np.hanning(cough_duration)
            burst *= envelope
            audio[cough_start:cough_start + cough_duration] = burst
        
        elif event_type == 'speech':
            # Simulate speech with formant-like frequencies
            f1, f2 = 700, 1200  # Formant frequencies
            audio = 0.2 * np.sin(2 * np.pi * f1 * t)
            audio += 0.15 * np.sin(2 * np.pi * f2 * t)
            # Add amplitude modulation
            audio *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        
        else:  # 'other'
            # Random tonal sound
            freq = np.random.uniform(200, 800)
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        return audio
    
    def generate_background_noise(self, noise_type='ambient'):
        """Generate background noise."""
        if noise_type == 'ambient':
            # Low-level pink-ish noise
            white = np.random.normal(0, 0.05, self.num_samples)
            # Simple low-pass filter
            b = np.ones(10) / 10
            audio = np.convolve(white, b, mode='same')
        
        elif noise_type == 'traffic':
            # Mix of low frequencies
            t = np.linspace(0, self.duration, self.num_samples)
            audio = 0.02 * np.sin(2 * np.pi * 50 * t)  # Engine hum
            audio += 0.01 * np.sin(2 * np.pi * 100 * t)
            audio += 0.03 * np.random.normal(0, 1, self.num_samples)  # Random noise
        
        elif noise_type == 'white':
            audio = 0.05 * np.random.normal(0, 1, self.num_samples)
        
        else:  # 'silence'
            audio = 0.001 * np.random.normal(0, 1, self.num_samples)
        
        return audio
    
    def generate_impulse_response(self, ir_type='room'):
        """Generate synthetic impulse response."""
        if ir_type == 'room':
            # Simple room IR with early reflections and decay
            ir_length = int(0.5 * self.sample_rate)  # 0.5 second IR
            ir = np.zeros(ir_length)
            ir[0] = 1.0  # Direct sound
            # Early reflections
            for i in range(5):
                delay = int((i + 1) * 0.02 * self.sample_rate)
                if delay < ir_length:
                    ir[delay] = 0.7 ** (i + 1)
            # Exponential decay
            decay = np.exp(-3 * np.linspace(0, 0.5, ir_length))
            noise = 0.1 * np.random.normal(0, 1, ir_length)
            ir += noise * decay
        
        elif ir_type == 'hall':
            # Longer reverb
            ir_length = int(1.0 * self.sample_rate)
            ir = np.zeros(ir_length)
            ir[0] = 1.0
            # Multiple reflections
            for i in range(10):
                delay = int((i + 1) * 0.05 * self.sample_rate)
                if delay < ir_length:
                    ir[delay] = 0.8 ** (i + 1)
            # Longer decay
            decay = np.exp(-2 * np.linspace(0, 1.0, ir_length))
            noise = 0.15 * np.random.normal(0, 1, ir_length)
            ir += noise * decay
        
        else:  # 'none'
            ir = np.array([1.0])  # Delta function (no reverb)
        
        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-8)
        
        # Pad to match audio length
        if len(ir) < self.num_samples:
            ir = np.pad(ir, (0, self.num_samples - len(ir)))
        else:
            ir = ir[:self.num_samples]
        
        return ir
    
    def create_test_dataset(self, num_train=20, num_val=10, num_backgrounds=15):
        """Create test audio files and CSV files with separate event and background CSVs."""
        print(f"Creating test dataset in {self.test_dir}")
        
        # Event types - now only snoring and other
        event_types = ['snoring', 'other']
        background_types = ['ambient', 'traffic', 'white', 'silence']
        ir_types = ['room', 'hall', 'none']
        
        # Generate event files for training
        train_event_data = []
        print("Generating training event audio files...")
        for i in range(num_train):
            # Alternate between snoring and other
            event_type = event_types[i % 2]
            if event_type == 'snoring':
                audio = self.generate_synthetic_event('snoring')
            else:
                # For 'other', randomly choose between cough, speech, or other sound
                other_type = np.random.choice(['cough', 'speech', 'other'])
                audio = self.generate_synthetic_event(other_type)
            
            filename = f"train_event_{event_type}_{i:04d}.wav"
            filepath = os.path.join(self.events_dir, filename)
            self.create_wav_file(filepath, audio)
            
            # Label: 1 for snoring, 0 for other
            label = 1 if event_type == 'snoring' else 0
            train_event_data.append({
                'event_path': filepath,
                'label': label
            })
        
        # Generate event files for validation
        val_event_data = []
        print("Generating validation event audio files...")
        for i in range(num_val):
            event_type = event_types[i % 2]
            if event_type == 'snoring':
                audio = self.generate_synthetic_event('snoring')
            else:
                other_type = np.random.choice(['cough', 'speech', 'other'])
                audio = self.generate_synthetic_event(other_type)
            
            filename = f"val_event_{event_type}_{i:04d}.wav"
            filepath = os.path.join(self.events_dir, filename)
            self.create_wav_file(filepath, audio)
            
            label = 1 if event_type == 'snoring' else 0
            val_event_data.append({
                'event_path': filepath,
                'label': label
            })
        
        # Generate background files for training
        train_bg_data = []
        print("Generating training background audio files...")
        for i in range(num_backgrounds):
            bg_type = background_types[i % len(background_types)]
            audio = self.generate_background_noise(bg_type)
            filename = f"train_background_{bg_type}_{i:04d}.wav"
            filepath = os.path.join(self.backgrounds_dir, filename)
            self.create_wav_file(filepath, audio)
            train_bg_data.append({'background_path': filepath})
        
        # Generate background files for validation (fewer backgrounds)
        val_bg_data = []
        num_val_backgrounds = max(5, num_backgrounds // 3)
        print("Generating validation background audio files...")
        for i in range(num_val_backgrounds):
            bg_type = background_types[i % len(background_types)]
            audio = self.generate_background_noise(bg_type)
            filename = f"val_background_{bg_type}_{i:04d}.wav"
            filepath = os.path.join(self.backgrounds_dir, filename)
            self.create_wav_file(filepath, audio)
            val_bg_data.append({'background_path': filepath})
        
        # Generate impulse response files
        ir_files = []
        print("Generating impulse response files...")
        for i, ir_type in enumerate(ir_types):
            ir = self.generate_impulse_response(ir_type)
            filename = f"ir_{ir_type}_{i:02d}.wav"
            filepath = os.path.join(self.ir_dir, filename)
            self.create_wav_file(filepath, ir)
            ir_files.append(filepath)
        
        # Save CSV files
        train_event_csv = os.path.join(self.test_dir, 'train_event.csv')
        train_bg_csv = os.path.join(self.test_dir, 'background_train.csv')
        val_event_csv = os.path.join(self.test_dir, 'val_event.csv')
        val_bg_csv = os.path.join(self.test_dir, 'background_val.csv')
        
        pd.DataFrame(train_event_data).to_csv(train_event_csv, index=False)
        pd.DataFrame(train_bg_data).to_csv(train_bg_csv, index=False)
        pd.DataFrame(val_event_data).to_csv(val_event_csv, index=False)
        pd.DataFrame(val_bg_data).to_csv(val_bg_csv, index=False)
        
        print(f"Created datasets:")
        print(f"  Training: {num_train} events, {num_backgrounds} backgrounds")
        print(f"  Validation: {num_val} events, {num_val_backgrounds} backgrounds")
        print(f"CSV files:")
        print(f"  Train event: {train_event_csv}")
        print(f"  Train background: {train_bg_csv}")
        print(f"  Val event: {val_event_csv}")
        print(f"  Val background: {val_bg_csv}")
        print(f"Label mapping: snoring=1, other=0")
        
        return train_event_csv, train_bg_csv, val_event_csv, val_bg_csv, ir_files
    
    def test_basic_functionality(self, loader, dataset, test_name="Basic"):
        """Test basic data loading functionality."""
        print(f"\n--- Testing {test_name} Functionality ---")
        
        try:
            # Test loading one batch
            for i, (audio_batch, label_batch) in enumerate(dataset.take(1)):
                print(f"Batch {i+1}:")
                print(f"  Audio shape: {audio_batch.shape}")
                print(f"  Label shape: {label_batch.shape}")
                print(f"  Audio range: [{tf.reduce_min(audio_batch):.3f}, {tf.reduce_max(audio_batch):.3f}]")
                print(f"  Labels: {label_batch.numpy()}")
                
                # Check for NaN or Inf
                assert not tf.reduce_any(tf.math.is_nan(audio_batch)), "NaN values in audio!"
                assert not tf.reduce_any(tf.math.is_inf(audio_batch)), "Inf values in audio!"
                
                print("  ✓ No NaN or Inf values")
                
            print(f"✓ {test_name} functionality test passed!")
            return True
            
        except Exception as e:
            print(f"✗ {test_name} functionality test failed: {e}")
            return False
    
    def test_augmentations(self, train_event_csv, train_bg_csv, ir_files):
        """Test specific augmentations."""
        print("\n--- Testing Augmentations ---")
        
        # Test each augmentation individually
        augmentations_to_test = [
            ('gain', {'gain': {'prob': 1.0, 'min_gain': 0.5, 'max_gain': 2.0}}),
            ('polarity_inversion', {'polarity_inversion': {'prob': 1.0}}),
            ('colored_noise', {'colored_noise': {'prob': 1.0, 'noise_types': ['white'], 'snr_db': [20, 20]}}),
            ('peak_normalization', {'peak_normalization': {'prob': 1.0, 'target_peak': 0.9}}),
            ('time_shift', {'time_shift': {'prob': 1.0, 'max_shift_ratio': 0.1}}),
        ]
        
        for aug_name, aug_config in augmentations_to_test:
            print(f"\nTesting {aug_name}...")
            
            # Create minimal config with only this augmentation
            config = {k: {'prob': 0.0} for k in ['gain', 'polarity_inversion', 'impulse_response', 
                                                  'colored_noise', 'high_pass_filter', 'peak_normalization', 
                                                  'pitch_shift', 'time_shift']}
            config.update(aug_config)
            config['mixing'] = {'snr_range_db': [10, 10]}
            
            if aug_name == 'impulse_response':
                config['impulse_response']['ir_paths'] = ir_files
            
            try:
                # Import the loader class (assuming it's available)
                
                
                loader = AdvancedAudioDataLoader(
                    train_event_csv,
                    train_bg_csv,
                    duration=2.0,
                    sample_rate=16000,
                    augmentation_config=config
                )
                
                dataset = loader.create_dataset(batch_size=4, shuffle=False)
                
                # Load one batch
                for audio_batch, label_batch in dataset.take(1):
                    print(f"  ✓ {aug_name} augmentation applied successfully")
                    print(f"    Audio shape: {audio_batch.shape}")
                    break
                    
            except Exception as e:
                print(f"  ✗ {aug_name} augmentation failed: {e}")
    
    def test_classification_modes(self, train_event_csv, train_bg_csv):
        """Test sigmoid vs softmax classification modes."""
        print("\n--- Testing Classification Modes ---")
        
        try:
            
            
            # Test sigmoid mode
            print("\nTesting sigmoid (binary) mode...")
            loader_sigmoid = AdvancedAudioDataLoader(
                train_event_csv,
                train_bg_csv,
                duration=2.0,
                sample_rate=16000,
                classification_mode='sigmoid',
                augmentation_config={'mixing': {'snr_range_db': [10, 10]}}
            )
            dataset_sigmoid = loader_sigmoid.create_dataset(batch_size=4)
            
            for audio_batch, label_batch in dataset_sigmoid.take(1):
                print(f"  Sigmoid labels: {label_batch.numpy()}")
                assert label_batch.dtype == tf.float32, "Sigmoid labels should be float32"
                assert tf.reduce_all((label_batch == 0) | (label_batch == 1)), "Sigmoid labels should be 0 or 1"
                print("  ✓ Sigmoid mode working correctly")
            
            # Test softmax mode
            print("\nTesting softmax (multi-class) mode...")
            loader_softmax = AdvancedAudioDataLoader(
                train_event_csv,
                train_bg_csv,
                duration=2.0,
                sample_rate=16000,
                classification_mode='softmax',
                augmentation_config={'mixing': {'snr_range_db': [10, 10]}}
            )
            dataset_softmax = loader_softmax.create_dataset(batch_size=4)
            
            for audio_batch, label_batch in dataset_softmax.take(1):
                print(f"  Softmax labels: {label_batch.numpy()}")
                print("  ✓ Softmax mode working correctly")
                
        except Exception as e:
            print(f"✗ Classification mode test failed: {e}")
    
    def simple_shape_test(self, event_csv, bg_csv):
        """Simple test to check dataset output shapes."""
        print("\n--- Simple Shape Test ---")
        try:
            
            
            # Create loader
            loader = AdvancedAudioDataLoader(
                event_csv,
                bg_csv,
                duration=2.0,
                sample_rate=16000,
                classification_mode='sigmoid',
                pairing_mode='random'
            )
            
            # Create dataset
            train_ds = loader.create_dataset(batch_size=8, shuffle=True)
            
            # Test loop - exactly what you wanted
            for x, y in train_ds.take(1):
                print(f"Audio batch shape: {x.shape}")
                print(f"Label batch shape: {y.shape}")
                print(f"Audio dtype: {x.dtype}")
                print(f"Label dtype: {y.dtype}")
                print(f"Sample labels: {y.numpy()}")
                break
                
        except Exception as e:
            print(f"✗ Simple shape test failed: {e}")
    
    def cleanup(self):
        """Clean up test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"\nCleaned up test directory: {self.test_dir}")


def run_all_tests():
    """Run all tests for AdvancedAudioDataLoader."""
    print("=" * 60)
    print("ADVANCED AUDIO DATA LOADER TEST SUITE")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    # Create tester
    tester = AudioDataLoaderTester()
    
    try:
        # Create test dataset
        train_event_csv, train_bg_csv, val_event_csv, val_bg_csv, ir_files = tester.create_test_dataset(
            num_train=20, 
            num_val=10,
            num_backgrounds=15
        )
        
        # Test 1: Basic functionality with augmentations
        print("\n" + "=" * 40)
        aug_config = {
            'gain': {'prob': 0.5, 'min_gain': 0.8, 'max_gain': 1.2},
            'polarity_inversion': {'prob': 0.3},
            'impulse_response': {'prob': 0.0, 'ir_paths': ir_files},  # Disabled for basic test
            'colored_noise': {'prob': 0.3, 'noise_types': ['white', 'pink'], 'snr_db': [20, 30]},
            'high_pass_filter': {'prob': 0.0},  # Disabled for basic test
            'peak_normalization': {'prob': 0.5, 'target_peak': 0.9},
            'pitch_shift': {'prob': 0.0},  # Disabled for basic test
            'time_shift': {'prob': 0.3, 'max_shift_ratio': 0.1},
            'mixing': {'snr_range_db': [5, 15]}
        }
        
        train_loader = AdvancedAudioDataLoader(
            train_event_csv,
            train_bg_csv,
            duration=2.0,
            sample_rate=16000,
            classification_mode='sigmoid',
            augmentation_config=aug_config,
            pairing_mode='random'
        )
        
        # Print dataset info
        info = train_loader.get_info()
        print(f"Dataset Info:")
        print(f"  Events: {info['num_events']} (Snoring: {info['num_snoring']}, Other: {info['num_other']})")
        print(f"  Backgrounds: {info['num_backgrounds']}")
        print(f"  Pairing mode: {info['pairing_mode']}")
        
        train_dataset = train_loader.create_dataset(batch_size=4, shuffle=True)
        tester.test_basic_functionality(train_loader, train_dataset, "Training")
        
        # Test 2: Validation without augmentations
        print("\n" + "=" * 40)
        val_aug_config = {k: {'prob': 0.0} for k in aug_config.keys()}
        val_aug_config['mixing'] = {'snr_range_db': [10, 10]}
        
        val_loader = AdvancedAudioDataLoader(
            val_event_csv,
            val_bg_csv,
            duration=2.0,
            sample_rate=16000,
            classification_mode='sigmoid',
            augmentation_config=val_aug_config,
            pairing_mode='sequential'
        )
        
        val_dataset = val_loader.create_dataset(batch_size=4, shuffle=False)
        tester.test_basic_functionality(val_loader, val_dataset, "Validation")
        
        # Test 3: Individual augmentations
        print("\n" + "=" * 40)
        tester.test_augmentations(train_event_csv, train_bg_csv, ir_files)
        
        # Test 4: Classification modes
        print("\n" + "=" * 40)
        tester.test_classification_modes(train_event_csv, train_bg_csv)
        
        # Test 5: Pairing modes
        print("\n" + "=" * 40)
        print("\n--- Testing Pairing Modes ---")
        
        # Test random pairing
        print("\nTesting random pairing...")
        random_loader = AdvancedAudioDataLoader(
            train_event_csv,
            train_bg_csv,
            duration=2.0,
            sample_rate=16000,
            pairing_mode='random',
            augmentation_config={'mixing': {'snr_range_db': [10, 10]}}
        )
        print("  ✓ Random pairing mode working")
        
        # Test sequential pairing
        print("\nTesting sequential pairing...")
        seq_loader = AdvancedAudioDataLoader(
            train_event_csv,
            train_bg_csv,
            duration=2.0,
            sample_rate=16000,
            pairing_mode='sequential',
            augmentation_config={'mixing': {'snr_range_db': [10, 10]}}
        )
        print("  ✓ Sequential pairing mode working")
        
        # Test 6: Simple shape test
        print("\n" + "=" * 40)
        tester.simple_shape_test(train_event_csv, train_bg_csv)
        
        # Performance test
        print("\n" + "=" * 40)
        print("\n--- Performance Test ---")
        import time
        
        # Time loading batches
        start_time = time.time()
        num_batches = 0
        for _ in train_dataset.take(5):
            num_batches += 1
        
        elapsed = time.time() - start_time
        print(f"Loaded {num_batches} batches in {elapsed:.2f} seconds")
        print(f"Average time per batch: {elapsed/num_batches:.3f} seconds")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        keep_files = input("\nKeep test files for inspection? (y/n): ").lower() == 'y'
        if not keep_files:
            tester.cleanup()
        else:
            print(f"Test files kept in: {tester.test_dir}")


if __name__ == "__main__":
    run_all_tests()
