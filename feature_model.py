import tensorflow as tf
import numpy as np
import os


class AudioFeatureExtractor:
    """Extract audio features from raw audio using only TensorFlow operations."""
    
    def __init__(self, sampling_rate=16000, duration=2.0, hop_length=256, window_size=512, roll_percent=0.85):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.hop_length = hop_length
        self.window_size = window_size
        self.roll_percent = roll_percent
        self.target_length = int(sampling_rate * duration)
        
    def _pad_audio(self, audio: tf.Tensor) -> tf.Tensor:
        """Pad audio to the specified duration if necessary."""
        current_length = tf.shape(audio)[0]
        padding_needed = self.target_length - current_length
        
        padded_audio = tf.cond(
            padding_needed > 0,
            lambda: tf.pad(audio, [[0, padding_needed]]),
            lambda: audio[:self.target_length]
        )
        
        return padded_audio
    
    def _get_frames(self, audio: tf.Tensor) -> tf.Tensor:
        """Extract overlapping frames from audio."""
        frames = tf.signal.frame(
            audio,
            frame_length=self.window_size,
            frame_step=self.hop_length,
            pad_end=False
        )
        return frames
    
    def _compute_zcr(self, frames: tf.Tensor) -> tf.Tensor:
        """Compute Zero Crossing Rate for each frame."""
        signs = tf.sign(frames)
        padded_signs = tf.pad(signs, [[0, 0], [1, 0]], constant_values=0)
        sign_changes = tf.abs(padded_signs[:, 1:] - padded_signs[:, :-1])
        zcr = tf.reduce_sum(sign_changes, axis=1) / 2.0
        zcr = zcr / tf.cast(self.window_size, tf.float32)
        return zcr
    
    def _compute_rms(self, frames: tf.Tensor) -> tf.Tensor:
        """Compute Root Mean Square energy for each frame."""
        squared = tf.square(frames)
        mean_squared = tf.reduce_mean(squared, axis=1)
        rms = tf.sqrt(mean_squared)
        return rms
    
    def _compute_stft(self, audio: tf.Tensor) -> tf.Tensor:
        """Compute magnitude spectrogram using TensorFlow's STFT."""
        stft = tf.signal.stft(
            audio,
            frame_length=self.window_size,
            frame_step=self.hop_length,
            fft_length=self.window_size,
            pad_end=False
        )
        magnitude = tf.abs(stft)
        return magnitude
    
    def _compute_spectral_centroid(self, magnitude_spectrogram: tf.Tensor) -> tf.Tensor:
        """Compute Spectral Centroid for each frame."""
        num_freq_bins = tf.shape(magnitude_spectrogram)[1]
        freq_bins = tf.range(num_freq_bins, dtype=tf.float32)
        freq_bins = freq_bins * self.sampling_rate / (2.0 * tf.cast(num_freq_bins - 1, tf.float32))
        
        weighted_sum = tf.reduce_sum(magnitude_spectrogram * freq_bins[tf.newaxis, :], axis=1)
        magnitude_sum = tf.reduce_sum(magnitude_spectrogram, axis=1)
        epsilon = tf.keras.backend.epsilon()
        spectral_centroid = weighted_sum / (magnitude_sum + epsilon)
        
        return spectral_centroid
    
    def _compute_spectral_rolloff(self, magnitude_spectrogram: tf.Tensor) -> tf.Tensor:
        """Compute Spectral Rolloff for each frame."""
        total_energy = tf.reduce_sum(magnitude_spectrogram, axis=1, keepdims=True)
        epsilon = tf.keras.backend.epsilon()
        total_energy = total_energy + epsilon
        
        cumsum = tf.cumsum(magnitude_spectrogram / total_energy, axis=1)
        mask = cumsum >= self.roll_percent
        mask_float = tf.cast(mask, tf.float32)
        rolloff_bins = tf.argmax(mask_float, axis=1, output_type=tf.int32)
        
        rolloff_bins_float = tf.cast(rolloff_bins, tf.float32)
        num_freq_bins = tf.cast(tf.shape(magnitude_spectrogram)[1], tf.float32)
        spectral_rolloff = rolloff_bins_float * self.sampling_rate / (2.0 * (num_freq_bins - 1.0))
        
        return spectral_rolloff
    
    def extract_features(self, audio: tf.Tensor) -> dict:
        """Extract all audio features from the input audio tensor."""
        padded_audio = self._pad_audio(audio)
        frames = self._get_frames(padded_audio)
        
        # Time-domain features
        zcr = self._compute_zcr(frames)
        rms = self._compute_rms(frames)
        
        # Frequency-domain features
        magnitude_spectrogram = self._compute_stft(padded_audio)
        spectral_centroid = self._compute_spectral_centroid(magnitude_spectrogram)
        spectral_rolloff = self._compute_spectral_rolloff(magnitude_spectrogram)
        
        features = {
            'zcr': zcr,
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff
        }
        
        return features


class FeatureExtractorBlock(tf.keras.layers.Layer):
    """Feature extractor block for processing audio features."""
    
    def __init__(self, output_dim=64, name_suffix="", **kwargs):
        super().__init__(name=f"feature_extractor_block_{name_suffix}", **kwargs)
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name=f'dense1_{name_suffix}')
        self.dense2 = tf.keras.layers.Dense(96, activation='relu', name=f'dense2_{name_suffix}')
        self.dense3 = tf.keras.layers.Dense(output_dim, activation='relu', name=f'dense3_{name_suffix}')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class SingleHeadAttention(tf.keras.layers.Layer):
    """Single head attention layer for feature fusion."""
    
    def __init__(self, d_model, **kwargs):
        super().__init__(name="single_head_attention", **kwargs)
        self.d_model = d_model
        self.wq = tf.keras.layers.Dense(d_model, name="attention_query")
        self.wk = tf.keras.layers.Dense(d_model, name="attention_key")
        self.wv = tf.keras.layers.Dense(d_model, name="attention_value")
    
    def call(self, inputs):
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(attention_weights, v)
        output = tf.reduce_mean(attended, axis=1)
        
        return output


class AudioFeatureModel(tf.keras.Model):
    """
    Model that processes audio features through 4 separate branches.
    Each audio feature (ZCR, RMS, Spectral Centroid, Spectral Rolloff) has its own branch.
    """
    
    def __init__(self, 
                 sampling_rate=16000,
                 audio_duration=2.0,
                 fusion_method="average", 
                 feature_dim=64, 
                 num_classes=1,
                 target_windows=186,
                 **kwargs):
        super().__init__(name="AudioFeatureModel", **kwargs)
        
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.fusion_method = fusion_method
        self.feature_dim = feature_dim
        self.target_windows = target_windows
        
        # Audio feature extractor
        self.audio_extractor = AudioFeatureExtractor(
            sampling_rate=sampling_rate,
            duration=audio_duration,
            hop_length=256,
            window_size=512,
            roll_percent=0.85
        )
        
        # Feature processing layers - one for each audio feature
        # These expand single feature values to 32-dim representations
        self.zcr_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ], name='zcr_processor')
        
        self.rms_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ], name='rms_processor')
        
        self.centroid_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ], name='centroid_processor')
        
        self.rolloff_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ], name='rolloff_processor')
        
        # Feature extractor blocks - one for each audio feature branch
        self.extractor_block_zcr = FeatureExtractorBlock(feature_dim, name_suffix="zcr")
        self.extractor_block_rms = FeatureExtractorBlock(feature_dim, name_suffix="rms")
        self.extractor_block_centroid = FeatureExtractorBlock(feature_dim, name_suffix="centroid")
        self.extractor_block_rolloff = FeatureExtractorBlock(feature_dim, name_suffix="rolloff")
        
        # Batch normalization layers for each branch
        self.norm_zcr = tf.keras.layers.BatchNormalization(name='norm_zcr')
        self.norm_rms = tf.keras.layers.BatchNormalization(name='norm_rms')
        self.norm_centroid = tf.keras.layers.BatchNormalization(name='norm_centroid')
        self.norm_rolloff = tf.keras.layers.BatchNormalization(name='norm_rolloff')
        
        # Fusion layer
        if fusion_method == "attention":
            self.fusion_layer = SingleHeadAttention(feature_dim)
        
        # Classification head
        self.classifier_dense1 = tf.keras.layers.Dense(128, activation='relu', name='classifier_dense1')
        self.classifier_bn1 = tf.keras.layers.BatchNormalization(name='classifier_bn1')
        self.classifier_dropout1 = tf.keras.layers.Dropout(0.3, name='classifier_dropout1')
        
        self.classifier_dense2 = tf.keras.layers.Dense(64, activation='relu', name='classifier_dense2')
        self.classifier_bn2 = tf.keras.layers.BatchNormalization(name='classifier_bn2')
        self.classifier_dropout2 = tf.keras.layers.Dropout(0.2, name='classifier_dropout2')
        
        self.classifier_output = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='classifier_output')
    
    @tf.function
    def process_audio_features(self, audio):
        """Extract and process audio features."""
        # Extract features
        features = self.audio_extractor.extract_features(audio)
        
        # Each feature has shape (num_frames,)
        # We need to process them to match the expected input format
        
        # Get the number of frames
        num_frames = tf.shape(features['zcr'])[0]
        
        # Pad or truncate to target windows using tf.cond
        def adjust_frames(feature_tensor):
            return tf.cond(
                num_frames < self.target_windows,
                lambda: tf.pad(feature_tensor, [[0, self.target_windows - num_frames]]),
                lambda: feature_tensor[:self.target_windows]
            )
        
        # Adjust each feature to target windows
        zcr = adjust_frames(features['zcr'])
        rms = adjust_frames(features['rms'])
        spectral_centroid = adjust_frames(features['spectral_centroid'])
        spectral_rolloff = adjust_frames(features['spectral_rolloff'])
        
        return zcr, rms, spectral_centroid, spectral_rolloff
    
    def process_feature_branch(self, feature_values, feature_processor, extractor_block):
        """
        Process a single audio feature through its branch.
        
        Args:
            feature_values: Tensor of shape (num_windows,)
            feature_processor: Neural network to expand feature to 32-dim
            extractor_block: FeatureExtractorBlock for this branch
            
        Returns:
            Processed features of shape (feature_dim,)
        """
        # Expand feature values to higher dimension
        # Shape: (num_windows,) -> (num_windows, 1)
        feature_values = tf.expand_dims(feature_values, axis=-1)
        
        # Process through feature-specific processor
        # Shape: (num_windows, 1) -> (num_windows, 32)
        processed = feature_processor(feature_values)
        
        # Apply feature extractor block
        # Shape: (num_windows, 32) -> (num_windows, feature_dim)
        extracted = extractor_block(processed)
        
        # Aggregate across windows (mean pooling)
        # Shape: (num_windows, feature_dim) -> (feature_dim,)
        aggregated = tf.reduce_mean(extracted, axis=0)
        
        return aggregated
    
    def call(self, audio_batch, training=None):
        """
        Process raw audio through the model.
        
        Args:
            audio_batch: Tensor of shape (batch_size, audio_length)
            
        Returns:
            Classification predictions of shape (batch_size, num_classes)
        """
        batch_size = tf.shape(audio_batch)[0]
        
        # Process each audio sample in the batch
        def process_single_audio(audio):
            # Extract audio features
            zcr, rms, spectral_centroid, spectral_rolloff = self.process_audio_features(audio)
            
            # Process each feature through its own branch
            features_zcr = self.process_feature_branch(zcr, self.zcr_processor, self.extractor_block_zcr)
            features_rms = self.process_feature_branch(rms, self.rms_processor, self.extractor_block_rms)
            features_centroid = self.process_feature_branch(spectral_centroid, self.centroid_processor, self.extractor_block_centroid)
            features_rolloff = self.process_feature_branch(spectral_rolloff, self.rolloff_processor, self.extractor_block_rolloff)
            
            # Apply normalization to each feature branch
            features_zcr = self.norm_zcr(tf.expand_dims(features_zcr, 0), training=training)
            features_rms = self.norm_rms(tf.expand_dims(features_rms, 0), training=training)
            features_centroid = self.norm_centroid(tf.expand_dims(features_centroid, 0), training=training)
            features_rolloff = self.norm_rolloff(tf.expand_dims(features_rolloff, 0), training=training)
            
            # Remove the extra dimension added for batch norm
            features_zcr = tf.squeeze(features_zcr, 0)
            features_rms = tf.squeeze(features_rms, 0)
            features_centroid = tf.squeeze(features_centroid, 0)
            features_rolloff = tf.squeeze(features_rolloff, 0)
            
            # Stack all features for fusion
            all_features = tf.stack([features_zcr, features_rms, features_centroid, features_rolloff], axis=0)
            
            return all_features
        
        # Process batch
        all_batch_features = tf.map_fn(
            process_single_audio,
            audio_batch,
            fn_output_signature=tf.TensorSpec(shape=[4, self.feature_dim], dtype=tf.float32)
        )
        
        # Now all_batch_features has shape (batch_size, 4, feature_dim)
        
        # Fusion
        if self.fusion_method == "attention":
            fused_features = self.fusion_layer(all_batch_features)
        else:  # average fusion
            fused_features = tf.reduce_mean(all_batch_features, axis=1)
        
        # Classification head
        x = self.classifier_dense1(fused_features)
        x = self.classifier_bn1(x, training=training)
        x = self.classifier_dropout1(x, training=training)
        
        x = self.classifier_dense2(x)
        x = self.classifier_bn2(x, training=training)
        x = self.classifier_dropout2(x, training=training)
        
        output = self.classifier_output(x)
        
        return output


def create_functional_audio_model(sampling_rate=16000,
                                 audio_duration=2.0,
                                 fusion_method="average",
                                 feature_dim=64,
                                 target_windows=186):
    """
    Create a functional API version of the audio feature model for better visualization.
    """
    # Input layer for raw audio
    audio_input = tf.keras.layers.Input(shape=(None,), name='raw_audio_input')
    
    # Feature extraction layer (using Lambda for the audio feature extraction)
    extractor = AudioFeatureExtractor(
        sampling_rate=sampling_rate,
        duration=audio_duration,
        hop_length=256,
        window_size=512,
        roll_percent=0.85
    )
    
    def extract_features_wrapper(audio):
        features = extractor.extract_features(audio)
        # Stack features
        return tf.stack([
            features['zcr'],
            features['rms'],
            features['spectral_centroid'],
            features['spectral_rolloff']
        ], axis=0)
    
    # Extract features for each audio in batch
    features = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(
            extract_features_wrapper,
            x,
            fn_output_signature=tf.TensorSpec(shape=[4, None], dtype=tf.float32)
        ),
        name='audio_feature_extraction'
    )(audio_input)
    
    # Transpose to (batch, time, features)
    features = tf.keras.layers.Lambda(
        lambda x: tf.transpose(x, [0, 2, 1]),
        name='transpose_features'
    )(features)
    
    # Pad/truncate to target windows
    def adjust_to_target_windows(x):
        current_frames = tf.shape(x)[1]
        return tf.cond(
            current_frames < target_windows,
            lambda: tf.pad(x, [[0, 0], [0, target_windows - current_frames], [0, 0]]),
            lambda: x[:, :target_windows, :]
        )
    
    features = tf.keras.layers.Lambda(
        lambda x: adjust_to_target_windows(x),
        name='adjust_windows'
    )(features)
    
    # Split features into separate branches
    zcr_branch = tf.keras.layers.Lambda(lambda x: x[:, :, 0], name='extract_zcr')(features)
    rms_branch = tf.keras.layers.Lambda(lambda x: x[:, :, 1], name='extract_rms')(features)
    centroid_branch = tf.keras.layers.Lambda(lambda x: x[:, :, 2], name='extract_centroid')(features)
    rolloff_branch = tf.keras.layers.Lambda(lambda x: x[:, :, 3], name='extract_rolloff')(features)
    
    # Process each branch
    def create_branch(feature_input, name_suffix):
        # Expand dimension
        x = tf.keras.layers.Reshape((target_windows, 1), name=f'reshape_{name_suffix}')(feature_input)
        
        # Feature processor
        x = tf.keras.layers.Dense(16, activation='relu', name=f'process_dense1_{name_suffix}')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name=f'process_dense2_{name_suffix}')(x)
        
        # Feature extractor block
        x = tf.keras.layers.Dense(128, activation='relu', name=f'extract_dense1_{name_suffix}')(x)
        x = tf.keras.layers.Dense(96, activation='relu', name=f'extract_dense2_{name_suffix}')(x)
        x = tf.keras.layers.Dense(feature_dim, activation='relu', name=f'extract_dense3_{name_suffix}')(x)
        
        # Aggregate
        x = tf.keras.layers.GlobalAveragePooling1D(name=f'aggregate_{name_suffix}')(x)
        
        return x
    
    # Create all branches
    zcr_processed = create_branch(zcr_branch, 'zcr')
    rms_processed = create_branch(rms_branch, 'rms')
    centroid_processed = create_branch(centroid_branch, 'centroid')
    rolloff_processed = create_branch(rolloff_branch, 'rolloff')
    
    # Add normalization layers
    zcr_normalized = tf.keras.layers.BatchNormalization(name='norm_zcr')(zcr_processed)
    rms_normalized = tf.keras.layers.BatchNormalization(name='norm_rms')(rms_processed)
    centroid_normalized = tf.keras.layers.BatchNormalization(name='norm_centroid')(centroid_processed)
    rolloff_normalized = tf.keras.layers.BatchNormalization(name='norm_rolloff')(rolloff_processed)
    
    # Stack for fusion
    stacked = tf.keras.layers.Lambda(
        lambda x: tf.stack(x, axis=1),
        name='stack_branches'
    )([zcr_normalized, rms_normalized, centroid_normalized, rolloff_normalized])
    
    # Fusion
    if fusion_method == "attention":
        # Attention mechanism
        q = tf.keras.layers.Dense(feature_dim, name='attention_query')(stacked)
        k = tf.keras.layers.Dense(feature_dim, name='attention_key')(stacked)
        v = tf.keras.layers.Dense(feature_dim, name='attention_value')(stacked)
        
        attention_scores = tf.keras.layers.Lambda(
            lambda x: tf.nn.softmax(tf.matmul(x[0], x[1], transpose_b=True) / tf.sqrt(float(feature_dim))),
            name='compute_attention'
        )([q, k])
        
        attended = tf.keras.layers.Lambda(
            lambda x: tf.matmul(x[0], x[1]),
            name='apply_attention'
        )([attention_scores, v])
        
        fused = tf.keras.layers.GlobalAveragePooling1D(name='attention_pooling')(attended)
    else:
        fused = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1),
            name='average_fusion'
        )(stacked)
    
    # Classification head
    x = tf.keras.layers.Dense(128, activation='relu', name='classifier_dense1')(fused)
    x = tf.keras.layers.BatchNormalization(name='classifier_bn1')(x)
    x = tf.keras.layers.Dropout(0.3, name='classifier_dropout1')(x)
    
    x = tf.keras.layers.Dense(64, activation='relu', name='classifier_dense2')(x)
    x = tf.keras.layers.BatchNormalization(name='classifier_bn2')(x)
    x = tf.keras.layers.Dropout(0.2, name='classifier_dropout2')(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier_output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=audio_input, outputs=output, name=f'AudioFeatureModel_{fusion_method}')
    
    return model


def visualize_audio_model(model, model_name, save_dir="audio_model_visualizations"):
    """
    Create comprehensive visualizations of the audio feature model.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VISUALIZING AUDIO MODEL: {model_name}")
    print(f"{'='*60}")
    
    # 1. Model Summary
    print("\n1. MODEL SUMMARY:")
    print("-" * 40)
    model.summary()
    
    # 2. Detailed layer information
    print(f"\n2. LAYER DETAILS:")
    print("-" * 40)
    total_params = 0
    trainable_params = 0
    
    for i, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        layer_trainable = sum([tf.size(w).numpy() for w in layer.trainable_weights])
        total_params += layer_params
        trainable_params += layer_trainable
        
        # Get output shape safely
        try:
            if hasattr(layer, 'output_shape'):
                output_shape = str(layer.output_shape)
            elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
                output_shape = str(layer.output.shape)
            else:
                output_shape = "Unknown"
        except:
            output_shape = "Unknown"
        
        print(f"Layer {i:2d}: {layer.name:30s} | "
              f"Type: {type(layer).__name__:20s} | "
              f"Output: {output_shape:30s} | "
              f"Params: {layer_params:8,d}")
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 3. Generate model plots
    print(f"\n3. GENERATING VISUALIZATION PLOTS:")
    print("-" * 40)
    
    try:
        # Standard plot
        plot_path = os.path.join(save_dir, f"{model_name}_architecture.png")
        tf.keras.utils.plot_model(
            model,
            to_file=plot_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=200
        )
        print(f"✓ Architecture diagram saved: {plot_path}")
        
        # Detailed plot with layer activations
        plot_path_detailed = os.path.join(save_dir, f"{model_name}_detailed.png")
        tf.keras.utils.plot_model(
            model,
            to_file=plot_path_detailed,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            expand_nested=True,
            rankdir='TB',
            dpi=200
        )
        print(f"✓ Detailed diagram saved: {plot_path_detailed}")
        
        # Horizontal layout
        plot_path_horizontal = os.path.join(save_dir, f"{model_name}_horizontal.png")
        tf.keras.utils.plot_model(
            model,
            to_file=plot_path_horizontal,
            show_shapes=True,
            show_layer_names=True,
            rankdir='LR',  # Left to Right
            expand_nested=True,
            dpi=200
        )
        print(f"✓ Horizontal diagram saved: {plot_path_horizontal}")
        
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
        print("  Note: Install graphviz and pydot for visualization:")
        print("  pip install graphviz pydot")
        print("  sudo apt-get install graphviz")
    
    # 4. Save model for Netron
    print(f"\n4. SAVING MODEL FOR NETRON:")
    print("-" * 40)
    
    try:
        # Save in SavedModel format for TensorFlow Serving
        saved_model_path = os.path.join(save_dir, f"{model_name}_savedmodel")
        tf.saved_model.save(model, saved_model_path)
        print(f"✓ SavedModel saved: {saved_model_path}")
        print(f"  Open with Netron: https://netron.app")
        
    except Exception as e:
        print(f"⚠ Could not export SavedModel: {e}")
    
    try:
        # Save as .keras
        keras_path = os.path.join(save_dir, f"{model_name}.keras")
        model.save(keras_path)
        print(f"✓ Keras model saved: {keras_path}")
        
        # Save as .h5 for compatibility
        h5_path = os.path.join(save_dir, f"{model_name}.h5")
        model.save(h5_path, save_format='h5')
        print(f"✓ H5 model saved: {h5_path}")
        
        # Save model weights
        weights_path = os.path.join(save_dir, f"{model_name}_weights.h5")
        model.save_weights(weights_path)
        print(f"✓ Weights saved: {weights_path}")
        
    except Exception as e:
        print(f"⚠ Could not save model files: {e}")
    
    # 5. Create model configuration JSON
    print(f"\n5. MODEL CONFIGURATION:")
    print("-" * 40)
    
    import json
    config = model.get_config()
    config_path = os.path.join(save_dir, f"{model_name}_config.json")
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved: {config_path}")
    except:
        pass
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Dynamic'}")
    print(f"Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Dynamic'}")
    print(f"Number of layers: {len(model.layers)}")
    
    # 6. Create a simple architecture diagram using matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define positions
        y_positions = {
            'input': 0.9,
            'features': 0.75,
            'branches': 0.5,
            'normalization': 0.35,
            'fusion': 0.2,
            'classifier': 0.05
        }
        
        # Draw boxes
        boxes = [
            ('Raw Audio Input\n(batch, audio_length)', 0.5, y_positions['input'], 'lightblue'),
            ('Audio Feature Extraction\nZCR, RMS, Centroid, Rolloff', 0.5, y_positions['features'], 'lightyellow'),
            ('ZCR\nBranch', 0.15, y_positions['branches'], 'lightgreen'),
            ('RMS\nBranch', 0.35, y_positions['branches'], 'lightgreen'),
            ('Centroid\nBranch', 0.55, y_positions['branches'], 'lightgreen'),
            ('Rolloff\nBranch', 0.75, y_positions['branches'], 'lightgreen'),
            ('BatchNorm\nZCR', 0.15, y_positions['normalization'], 'lightcyan'),
            ('BatchNorm\nRMS', 0.35, y_positions['normalization'], 'lightcyan'),
            ('BatchNorm\nCentroid', 0.55, y_positions['normalization'], 'lightcyan'),
            ('BatchNorm\nRolloff', 0.75, y_positions['normalization'], 'lightcyan'),
            ('Feature Fusion\n(Attention/Average)', 0.5, y_positions['fusion'], 'lightcoral'),
            ('Classification Head\n(Dense → BatchNorm → Dropout)', 0.5, y_positions['classifier'], 'plum')
        ]
        
        for text, x, y, color in boxes:
            rect = patches.FancyBboxPatch((x-0.08, y-0.03), 0.16, 0.06,
                                        boxstyle="round,pad=0.01",
                                        facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw arrows
        arrows = [
            (0.5, y_positions['input']-0.03, 0.5, y_positions['features']+0.03),
            (0.5, y_positions['features']-0.03, 0.15, y_positions['branches']+0.03),
            (0.5, y_positions['features']-0.03, 0.35, y_positions['branches']+0.03),
            (0.5, y_positions['features']-0.03, 0.55, y_positions['branches']+0.03),
            (0.5, y_positions['features']-0.03, 0.75, y_positions['branches']+0.03),
            (0.15, y_positions['branches']-0.03, 0.15, y_positions['normalization']+0.03),
            (0.35, y_positions['branches']-0.03, 0.35, y_positions['normalization']+0.03),
            (0.55, y_positions['branches']-0.03, 0.55, y_positions['normalization']+0.03),
            (0.75, y_positions['branches']-0.03, 0.75, y_positions['normalization']+0.03),
            (0.15, y_positions['normalization']-0.03, 0.5, y_positions['fusion']+0.03),
            (0.35, y_positions['normalization']-0.03, 0.5, y_positions['fusion']+0.03),
            (0.55, y_positions['normalization']-0.03, 0.5, y_positions['fusion']+0.03),
            (0.75, y_positions['normalization']-0.03, 0.5, y_positions['fusion']+0.03),
            (0.5, y_positions['fusion']-0.03, 0.5, y_positions['classifier']+0.03)
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Audio Feature Model Architecture with Batch Normalization - {model_name}', 
                     fontsize=16, weight='bold', pad=20)
        
        diagram_path = os.path.join(save_dir, f"{model_name}_custom_diagram.png")
        plt.tight_layout()
        plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Custom architecture diagram saved: {diagram_path}")
        
    except ImportError:
        print("⚠ Matplotlib not available for custom diagram")
    
    return save_dir


# Example usage and testing
def test_audio_feature_model():
    """Test the audio feature model with dummy data."""
    print("Testing Audio Feature Model with Batch Normalization...")
    print("="*60)
    
    # Model parameters
    batch_size = 8
    audio_length = int(1.5 * 16000)  # 1.5 seconds of audio
    
    # Create dummy audio batch
    audio_batch = tf.random.uniform(
        shape=(batch_size, audio_length),
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32
    )
    
    # Test with attention fusion
    print("\n1. Testing with ATTENTION fusion:")
    model_attention = AudioFeatureModel(
        sampling_rate=16000,
        audio_duration=2.0,
        fusion_method="attention",
        feature_dim=64,
        num_classes=1,
        target_windows=186
    )
    
    # Forward pass
    predictions_attention = model_attention(audio_batch)
    print(f"   Input shape: {audio_batch.shape}")
    print(f"   Output shape: {predictions_attention.shape}")
    
    # Test with average fusion
    print("\n2. Testing with AVERAGE fusion:")
    model_average = AudioFeatureModel(
        sampling_rate=16000,
        audio_duration=2.0,
        fusion_method="average",
        feature_dim=64,
        num_classes=1,
        target_windows=186
    )
    
    predictions_average = model_average(audio_batch)
    print(f"   Input shape: {audio_batch.shape}")
    print(f"   Output shape: {predictions_average.shape}")
    
    # Compile and test training
    print("\n3. Testing model compilation and training:")
    model_attention.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy labels
    dummy_labels = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=2, dtype=tf.int32)
    dummy_labels = tf.cast(dummy_labels, tf.float32)
    
    # Test one training step
    loss = model_attention.train_on_batch(audio_batch, dummy_labels)
    print(f"   Training loss: {loss}")
    
    # Test functional model and visualization
    print("\n4. Creating functional model for visualization:")
    functional_model = create_functional_audio_model(
        sampling_rate=16000,
        audio_duration=2.0,
        fusion_method="attention",
        feature_dim=64,
        target_windows=186
    )
    
    functional_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Visualize all models
    print("\n5. Generating visualizations:")
    dummy_input = tf.random.uniform((1, 16000), dtype=tf.float32)  # 1 second of audio
    _ = model_attention(dummy_input)
    # Visualize attention model
    save_dir_att = visualize_audio_model(model_attention, "AudioModel_Attention_BN")
    
    # Visualize average model  
    save_dir_avg = visualize_audio_model(model_average, "AudioModel_Average_BN")
    
    # Visualize functional model
    save_dir_func = visualize_audio_model(functional_model, "AudioModel_Functional_BN")
    
    return model_attention, model_average, functional_model


if __name__ == '__main__':
    # Test the models
    model_att, model_avg, func_model = test_audio_feature_model()
    
    print("\n" + "="*60)
    print("✅ All tests passed successfully!")
    print("\nThe model now includes:")
    print("1. Raw audio input processing")
    print("2. Four audio feature extraction branches (ZCR, RMS, Spectral Centroid, Spectral Rolloff)")
    print("3. Batch normalization after each feature branch")
    print("4. Feature fusion using attention or average pooling")
    print("5. Classification head with batch normalization and dropout")
    print("\nBatch normalization benefits:")
    print("- Normalizes features across different scales")
    print("- Improves training stability")
    print("- Faster convergence")
    print("- Acts as regularization")
    print("\nVisualization files created:")
    print("- Architecture diagrams (PNG)")
    print("- SavedModel format (for TensorFlow Serving)")
    print("- Keras/H5 models (for Netron visualization)")
    print("- Model configuration (JSON)")
    print("\nTo view in Netron:")
    print("1. Go to https://netron.app")
    print("2. Upload the .keras or .h5 file")
