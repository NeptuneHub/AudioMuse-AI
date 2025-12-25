# 1. Create a virtual environment:
#      python3 -m venv test/.venv
#
# 2. Activate the virtual environment:
#      source test/.venv/bin/activate
#
# 3. Install requirements:
#      pip install -r test/requirements.txt
# 4. Run this script:
#      python -m pytest tests/unit/test_analysis.py --tb=short

"""Unit tests for tasks/analysis.py"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from tasks.analysis import run_inference, _find_onnx_name, sigmoid, robust_load_audio_with_fallback, analyze_track


class TestFindOnnxName:
    """Tests for the _find_onnx_name helper function"""

    def test_direct_match(self):
        """Test when candidate name matches directly"""
        names = ['model/Placeholder', 'model/dense/BiasAdd']
        result = _find_onnx_name('model/Placeholder', names)
        assert result == 'model/Placeholder'

    def test_strip_colon_suffix(self):
        """Test stripping :0 suffix from TensorFlow-style names"""
        names = ['model/Placeholder', 'model/dense/BiasAdd']
        result = _find_onnx_name('model/Placeholder:0', names)
        assert result == 'model/Placeholder'

    def test_extract_last_part_after_slash(self):
        """Test extracting last part after '/' when full path doesn't match"""
        names = ['Placeholder', 'BiasAdd']
        result = _find_onnx_name('model/dense/Placeholder:0', names)
        assert result == 'Placeholder'

    def test_replace_slash_with_underscore(self):
        """Test replacing '/' with '_' for ONNX naming convention"""
        names = ['model_Placeholder', 'model_dense_BiasAdd']
        result = _find_onnx_name('model/Placeholder:0', names)
        assert result == 'model_Placeholder'

    def test_fallback_to_first_name(self):
        """Test fallback to first available name when no match found"""
        names = ['first_input', 'second_input']
        result = _find_onnx_name('completely_unknown_name', names)
        assert result == 'first_input'

    def test_empty_names_list(self):
        """Test with empty names list returns None"""
        names = []
        result = _find_onnx_name('any_name', names)
        assert result is None

    def test_complex_tensorflow_name(self):
        """Test with complex TensorFlow-style tensor name"""
        names = ['serving_default_model_Placeholder']
        result = _find_onnx_name('serving_default_model_Placeholder:0', names)
        assert result == 'serving_default_model_Placeholder'

    def test_nested_path_extraction(self):
        """Test extracting from deeply nested path"""
        names = ['BiasAdd']
        result = _find_onnx_name('model/layer1/layer2/BiasAdd:0', names)
        assert result == 'BiasAdd'


class TestRunInference:
    """Tests for the run_inference function"""

    def test_successful_inference_direct_match(self):
        """Test successful inference with direct name match"""
        # Create mock ONNX session
        mock_session = Mock()
        
        # Mock inputs
        mock_input = Mock()
        mock_input.name = 'model/Placeholder'
        mock_session.get_inputs.return_value = [mock_input]
        
        # Mock outputs
        mock_output = Mock()
        mock_output.name = 'model/dense/BiasAdd'
        mock_session.get_outputs.return_value = [mock_output]
        
        # Mock inference result
        expected_result = np.array([[0.1, 0.2, 0.3]])
        mock_session.run.return_value = [expected_result]
        
        # Run inference
        feed_dict = {'model/Placeholder': np.random.rand(1, 10)}
        result = run_inference(mock_session, feed_dict, 'model/dense/BiasAdd')
        
        # Verify
        assert result is not None
        np.testing.assert_array_equal(result, expected_result)
        mock_session.run.assert_called_once()

    def test_inference_with_tensorflow_style_names(self):
        """Test inference with TensorFlow-style names (:0 suffix)"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'model_Placeholder'  # ONNX style
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]
        
        expected_result = np.array([[0.5]])
        mock_session.run.return_value = [expected_result]
        
        # Feed dict with TF-style name
        feed_dict = {'model/Placeholder:0': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)
        
        assert result is not None
        np.testing.assert_array_equal(result, expected_result)

    def test_inference_without_output_tensor_name(self):
        """Test inference uses first output when output_tensor_name is None"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output1 = Mock()
        mock_output1.name = 'first_output'
        mock_output2 = Mock()
        mock_output2.name = 'second_output'
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]
        
        expected_result = np.array([[1.0, 2.0]])
        mock_session.run.return_value = [expected_result]
        
        feed_dict = {'input': np.random.rand(1, 3)}
        result = run_inference(mock_session, feed_dict, output_tensor_name=None)
        
        # Should use first_output
        assert result is not None
        mock_session.run.assert_called_with(['first_output'], {'input': feed_dict['input']})

    def test_inference_with_multiple_inputs(self):
        """Test inference with multiple input tensors"""
        mock_session = Mock()
        
        mock_input1 = Mock()
        mock_input1.name = 'input1'
        mock_input2 = Mock()
        mock_input2.name = 'input2'
        mock_session.get_inputs.return_value = [mock_input1, mock_input2]
        
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]
        
        expected_result = np.array([[0.7]])
        mock_session.run.return_value = [expected_result]
        
        feed_dict = {
            'input1': np.random.rand(1, 5),
            'input2': np.random.rand(1, 3)
        }
        result = run_inference(mock_session, feed_dict)
        
        assert result is not None
        # Verify both inputs were mapped
        call_args = mock_session.run.call_args
        assert 'input1' in call_args[0][1]
        assert 'input2' in call_args[0][1]

    def test_inference_returns_none_when_input_mapping_fails(self):
        """Test returns None when input name cannot be mapped"""
        mock_session = Mock()
        
        # No inputs available
        mock_session.get_inputs.return_value = []
        
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]
        
        feed_dict = {'unknown_input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)
        
        assert result is None

    def test_inference_returns_none_when_no_outputs(self):
        """Test returns None when ONNX session has no outputs"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        # No outputs
        mock_session.get_outputs.return_value = []
        
        feed_dict = {'input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)
        
        assert result is None

    def test_inference_with_path_based_name_mapping(self):
        """Test inference with path-based name extraction"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'Placeholder'  # Just the last part
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output = Mock()
        mock_output.name = 'BiasAdd'
        mock_session.get_outputs.return_value = [mock_output]
        
        expected_result = np.array([[0.3, 0.4]])
        mock_session.run.return_value = [expected_result]
        
        # Full TF-style path
        feed_dict = {'model/dense/Placeholder:0': np.random.rand(1, 8)}
        result = run_inference(mock_session, feed_dict, 'model/dense/BiasAdd:0')
        
        assert result is not None
        np.testing.assert_array_equal(result, expected_result)

    def test_inference_with_underscore_conversion(self):
        """Test inference with slash to underscore conversion"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'model_Placeholder'  # Underscores
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output = Mock()
        mock_output.name = 'model_output'
        mock_session.get_outputs.return_value = [mock_output]
        
        expected_result = np.array([[0.6]])
        mock_session.run.return_value = [expected_result]
        
        # Slashes in feed dict
        feed_dict = {'model/Placeholder': np.random.rand(1, 4)}
        result = run_inference(mock_session, feed_dict, 'model/output')
        
        assert result is not None
        np.testing.assert_array_equal(result, expected_result)

    def test_inference_result_unwrapping(self):
        """Test that result is properly unwrapped from list"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]
        
        # ONNX runtime returns list
        expected_array = np.array([[1.0, 2.0, 3.0]])
        mock_session.run.return_value = [expected_array]
        
        feed_dict = {'input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)
        
        # Should unwrap the array from the list
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected_array)

    def test_inference_with_empty_result_list(self):
        """Test handling of empty result list from ONNX runtime"""
        mock_session = Mock()
        
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]
        
        # Empty list returned
        mock_session.run.return_value = []
        
        feed_dict = {'input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)
        
        # Should return the empty list itself
        assert result == []


class TestSigmoid:
    """Tests for the sigmoid utility function"""

    def test_sigmoid_basic(self):
        """Test sigmoid with basic values"""
        result = sigmoid(0)
        assert np.isclose(result, 0.5)

    def test_sigmoid_positive(self):
        """Test sigmoid with positive value"""
        result = sigmoid(2.0)
        assert result > 0.5
        assert result < 1.0

    def test_sigmoid_negative(self):
        """Test sigmoid with negative value"""
        result = sigmoid(-2.0)
        assert result > 0.0
        assert result < 0.5

    def test_sigmoid_array(self):
        """Test sigmoid with numpy array"""
        x = np.array([0, 1, -1, 2, -2])
        result = sigmoid(x)
        
        assert len(result) == 5
        assert np.all(result > 0)
        assert np.all(result < 1)
        assert np.isclose(result[0], 0.5)

    def test_sigmoid_numerical_stability_large_positive(self):
        """Test sigmoid doesn't overflow with large positive values"""
        result = sigmoid(100)
        assert np.isfinite(result)
        assert np.isclose(result, 1.0)

    def test_sigmoid_numerical_stability_large_negative(self):
        """Test sigmoid doesn't underflow with large negative values"""
        result = sigmoid(-100)
        assert np.isfinite(result)
        assert np.isclose(result, 0.0)

    def test_sigmoid_symmetry(self):
        """Test sigmoid symmetry: sigmoid(x) + sigmoid(-x) = 1"""
        x = 1.5
        assert np.isclose(sigmoid(x) + sigmoid(-x), 1.0)


class TestRobustLoadAudioWithFallback:
    """Tests for the robust_load_audio_with_fallback function"""

    @patch('tasks.analysis.librosa.load')
    def test_successful_direct_load(self, mock_librosa_load):
        """Test successful direct loading with librosa"""
        # Mock successful librosa load
        expected_audio = np.random.rand(16000)
        expected_sr = 16000
        mock_librosa_load.return_value = (expected_audio, expected_sr)
        
        audio, sr = robust_load_audio_with_fallback('test.mp3', target_sr=16000)
        
        assert audio is not None
        assert sr == expected_sr
        np.testing.assert_array_equal(audio, expected_audio)
        mock_librosa_load.assert_called_once()

    @patch('tasks.analysis.librosa.load')
    def test_direct_load_with_custom_sample_rate(self, mock_librosa_load):
        """Test direct loading with custom target sample rate"""
        expected_audio = np.random.rand(22050)
        expected_sr = 22050
        mock_librosa_load.return_value = (expected_audio, expected_sr)
        
        audio, sr = robust_load_audio_with_fallback('test.wav', target_sr=22050)
        
        assert sr == 22050
        mock_librosa_load.assert_called_once_with('test.wav', sr=22050, mono=True, duration=600)

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_on_librosa_failure(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback to pydub when librosa fails"""
        # First librosa call fails
        mock_librosa_load.side_effect = [
            Exception("Librosa failed"),
            (np.random.rand(16000), 16000)  # Second call succeeds (loading temp WAV)
        ]
        
        # Mock AudioSegment
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=1000)  # Non-zero duration
        mock_processed = Mock()
        mock_segment.set_frame_rate.return_value = mock_processed
        mock_processed.set_channels.return_value = mock_processed
        mock_audio_segment.return_value = mock_segment
        
        # Mock temp file
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.wav'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')
        
        assert audio is not None
        assert sr == 16000
        assert mock_librosa_load.call_count == 2
        mock_audio_segment.assert_called_once()

    @patch('tasks.analysis.librosa.load')
    def test_returns_none_on_empty_audio(self, mock_librosa_load):
        """Test returns None when librosa loads empty audio"""
        # Librosa returns empty array
        mock_librosa_load.return_value = (np.array([]), 16000)
        
        audio, sr = robust_load_audio_with_fallback('empty.mp3')
        
        assert audio is None
        assert sr is None

    @patch('tasks.analysis.librosa.load')
    def test_returns_none_on_none_audio(self, mock_librosa_load):
        """Test returns None when librosa returns None"""
        mock_librosa_load.return_value = (None, 16000)
        
        audio, sr = robust_load_audio_with_fallback('invalid.mp3')
        
        assert audio is None
        assert sr is None

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_with_zero_duration_segment(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback returns None when pydub loads zero-duration segment"""
        mock_librosa_load.side_effect = Exception("Librosa failed")
        
        # Mock AudioSegment with zero duration
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=0)
        mock_audio_segment.return_value = mock_segment
        
        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')
        
        assert audio is None
        assert sr is None

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_cleans_up_temp_file(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback cleans up temporary WAV file"""
        mock_librosa_load.side_effect = [
            Exception("Librosa failed"),
            (np.random.rand(16000), 16000)
        ]
        
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=1000)
        mock_processed = Mock()
        mock_segment.set_frame_rate.return_value = mock_processed
        mock_processed.set_channels.return_value = mock_processed
        mock_audio_segment.return_value = mock_segment
        
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.wav'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Mock os.path.exists to return True
        with patch('tasks.analysis.os.path.exists', return_value=True):
            audio, sr = robust_load_audio_with_fallback('test.mp3')
        
        # Verify temp file was removed
        mock_remove.assert_called_with('/tmp/test.wav')

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_handles_silent_audio(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback detects and rejects silent audio (all zeros)"""
        mock_librosa_load.side_effect = [
            Exception("Librosa failed"),
            (np.zeros(16000), 16000)  # Silent audio
        ]
        
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=1000)
        mock_processed = Mock()
        mock_segment.set_frame_rate.return_value = mock_processed
        mock_processed.set_channels.return_value = mock_processed
        mock_audio_segment.return_value = mock_segment
        
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.wav'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        audio, sr = robust_load_audio_with_fallback('silent.mp3')
        
        assert audio is None
        assert sr is None

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    def test_fallback_handles_pydub_failure(self, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test returns None when both librosa and pydub fail"""
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_audio_segment.side_effect = Exception("Pydub failed")
        
        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')
        
        assert audio is None
        assert sr is None

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_resamples_audio(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback resamples audio to target sample rate"""
        mock_librosa_load.side_effect = [
            Exception("Librosa failed"),
            (np.random.rand(22050), 22050)
        ]
        
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=1000)
        mock_processed = Mock()
        mock_segment.set_frame_rate.return_value = mock_processed
        mock_processed.set_channels.return_value = mock_processed
        mock_audio_segment.return_value = mock_segment
        
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.wav'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        audio, sr = robust_load_audio_with_fallback('test.mp3', target_sr=22050)
        
        # Verify set_frame_rate was called with target_sr
        mock_segment.set_frame_rate.assert_called_once_with(22050)

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_converts_to_mono(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback converts audio to mono"""
        mock_librosa_load.side_effect = [
            Exception("Librosa failed"),
            (np.random.rand(16000), 16000)
        ]
        
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=1000)
        mock_processed = Mock()
        mock_segment.set_frame_rate.return_value = mock_processed
        mock_processed.set_channels.return_value = mock_processed
        mock_audio_segment.return_value = mock_segment
        
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.wav'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        audio, sr = robust_load_audio_with_fallback('stereo.mp3')
        
        # Verify set_channels was called with 1 (mono)
        mock_processed.set_channels.assert_called_once_with(1)

    @patch('tasks.analysis.librosa.load')
    def test_uses_audio_load_timeout_config(self, mock_librosa_load):
        """Test that AUDIO_LOAD_TIMEOUT config is used"""
        mock_librosa_load.return_value = (np.random.rand(16000), 16000)
        
        robust_load_audio_with_fallback('test.mp3', target_sr=16000)
        
        # Verify duration parameter is passed from config
        call_args = mock_librosa_load.call_args
        assert 'duration' in call_args.kwargs
        assert call_args.kwargs['duration'] == 600  # AUDIO_LOAD_TIMEOUT from config

    @patch('tasks.analysis.os.remove')
    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis.AudioSegment.from_file')
    @patch('tasks.analysis.NamedTemporaryFile')
    def test_fallback_export_parameters(self, mock_temp_file, mock_audio_segment, mock_librosa_load, mock_remove):
        """Test fallback uses correct export parameters for WAV conversion"""
        mock_librosa_load.side_effect = [
            Exception("Librosa failed"),
            (np.random.rand(16000), 16000)
        ]
        
        mock_segment = Mock()
        mock_segment.__len__ = Mock(return_value=1000)
        mock_processed = Mock()
        mock_segment.set_frame_rate.return_value = mock_processed
        mock_processed.set_channels.return_value = mock_processed
        mock_audio_segment.return_value = mock_segment
        
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.wav'
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        robust_load_audio_with_fallback('test.mp3', target_sr=16000)
        
        # Verify export was called with correct parameters
        export_call = mock_processed.export.call_args
        assert export_call[1]['format'] == 'wav'
        assert '-codec:a' in export_call[1]['parameters']
        assert 'pcm_s16le' in export_call[1]['parameters']


class TestAnalyzeTrack:
    """Tests for the analyze_track function
    
    WHAT WE'RE TESTING:
    - Control flow and decision logic (does it handle None audio correctly?)
    - Error handling paths (what happens when models fail?)
    - Data transformations (are spectrograms created correctly?)
    - Return value structure (does it return the right dict format?)
    - Integration between components (does audio -> spectrogram -> model flow work?)
    
    WHAT WE'RE NOT TESTING (mocked):
    - Whether librosa actually extracts features correctly (trust librosa)
    - Whether ONNX models actually predict correctly (trust ONNX runtime)
    - Actual audio processing quality (not relevant for unit tests)
    - File I/O operations (mocked for speed and reliability)
    """

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_successful_track_analysis(self, mock_audio_load, mock_mel, mock_beat, mock_rms, 
                                       mock_chroma, mock_onnx_session):
        """Test complete successful analysis flow
        
        TESTS: End-to-end successful analysis with all components working
        MOCKS: Audio loading, librosa features, ONNX models
        """
        # Mock audio loading
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        # Mock librosa features
        mock_beat.return_value = (120.0, np.array([0, 100, 200]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        # Mock ONNX session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]  # Embeddings
        mock_onnx_session.return_value = mock_session
        
        mood_labels = ['happy', 'sad', 'energetic', 'calm', 'aggressive']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Function returns correct structure
        assert result is not None
        assert embeddings is not None
        assert 'tempo' in result
        assert 'key' in result
        assert 'scale' in result
        assert 'moods' in result
        assert 'energy' in result
        assert isinstance(result['moods'], dict)
        assert len(result['moods']) == len(mood_labels)

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_audio_load_failure(self, mock_audio_load):
        """Test returns None when audio loading fails
        
        TESTS: Error handling - does function correctly handle audio load failure?
        MOCKS: Audio loading to simulate failure
        """
        # Mock audio load failure
        mock_audio_load.return_value = (None, None)
        
        mood_labels = ['happy', 'sad']
        model_paths = {'embedding': '/path/to/model.onnx'}
        
        result, embeddings = analyze_track('bad_file.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Function returns None on failure
        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_empty_audio(self, mock_audio_load):
        """Test returns None when audio is empty array
        
        TESTS: Edge case handling - empty audio detection
        """
        # Mock empty audio
        mock_audio_load.return_value = (np.array([]), 16000)
        
        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}
        
        result, embeddings = analyze_track('empty.mp3', mood_labels, model_paths)
        
        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_silent_audio(self, mock_audio_load):
        """Test returns None when audio is all zeros (silent)
        
        TESTS: Edge case - silent audio detection
        """
        # Mock silent audio (all zeros)
        mock_audio_load.return_value = (np.zeros(16000), 16000)
        
        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}
        
        result, embeddings = analyze_track('silent.mp3', mood_labels, model_paths)
        
        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_short_audio(self, mock_audio_load, mock_beat, mock_rms, 
                                         mock_chroma, mock_mel):
        """Test returns None when audio is too short for spectrograms
        
        TESTS: Edge case - audio too short to create patches
        """
        # Mock very short audio
        mock_audio = np.random.rand(100)  # Very short
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 10)
        mock_mel.return_value = np.random.rand(96, 10)  # Too short for patches
        
        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}
        
        result, embeddings = analyze_track('short.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Function detects too-short audio
        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_spectrogram_dtype_conversion(self, mock_audio_load, mock_mel, mock_beat, 
                                          mock_rms, mock_chroma, mock_onnx_session):
        """Test spectrograms are converted to float32
        
        TESTS: Data type conversion - critical for CPU compatibility
        """
        mock_audio = np.random.rand(16000).astype(np.float64)  # Start with float64 to test conversion
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        # Use float64 to verify the code converts it to float32
        mock_mel.return_value = np.random.rand(96, 1000).astype(np.float64)
        
        # Track what dtype is passed to ONNX for the FIRST model (embedding model)
        captured_input = None
        call_count = [0]  # Use list to allow modification in nested function
        
        def capture_run(output_names, feed_dict):
            nonlocal captured_input
            call_count[0] += 1
            # Only capture the FIRST call (embedding model with spectrogram patches)
            if call_count[0] == 1:
                for key, val in feed_dict.items():
                    captured_input = val
            # Return float32 to simulate real model output
            return [np.random.rand(5, 200).astype(np.float32)]
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.side_effect = capture_run
        mock_onnx_session.return_value = mock_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Critical dtype conversion to float32
        # The code explicitly converts to float32 with .astype(np.float32)
        assert captured_input is not None
        assert captured_input.dtype == np.dtype('float32')

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_key_detection_logic(self, mock_audio_load, mock_mel, mock_beat, mock_rms, 
                                  mock_chroma, mock_onnx_session):
        """Test key detection algorithm executes without errors
        
        TESTS: Key detection algorithm runs and produces valid output
        NOTE: We test the algorithm RUNS correctly, not musical accuracy
        """
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        
        # Use any valid chroma data - we're testing the algorithm works,
        # not that it's musically accurate (that's an integration test concern)
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, _ = analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Key detection produces valid results
        assert result is not None
        assert 'key' in result
        assert 'scale' in result
        # Key should be one of the 12 notes
        assert result['key'] in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # Scale should be major or minor
        assert result['scale'] in ['major', 'minor']

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_model_inference_failure_handling(self, mock_audio_load, mock_mel, mock_beat, 
                                               mock_rms, mock_chroma, mock_onnx_session):
        """Test returns None when model inference fails
        
        TESTS: Error handling - model failure recovery
        """
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        # Mock ONNX to raise exception
        mock_onnx_session.side_effect = Exception("Model loading failed")
        
        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/embedding.onnx'}
        
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Graceful failure on model errors
        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_tempo_extraction(self, mock_audio_load, mock_mel, mock_beat, mock_rms, 
                               mock_chroma, mock_onnx_session):
        """Test tempo is extracted and returned as float
        
        TESTS: Tempo extraction and type conversion
        """
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        # Mock specific tempo
        expected_tempo = 128.5
        mock_beat.return_value = (expected_tempo, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, _ = analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Tempo value is correctly extracted and typed
        assert result is not None
        assert result['tempo'] == expected_tempo
        assert isinstance(result['tempo'], float)

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_energy_calculation(self, mock_audio_load, mock_mel, mock_beat, mock_rms, 
                                 mock_chroma, mock_onnx_session):
        """Test energy is calculated from RMS
        
        TESTS: Energy calculation and averaging
        """
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        
        # Mock specific RMS values
        rms_values = np.array([[0.1, 0.2, 0.3, 0.4]])
        expected_energy = np.mean(rms_values)
        mock_rms.return_value = rms_values
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, _ = analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Energy is calculated as mean of RMS
        assert result is not None
        assert np.isclose(result['energy'], expected_energy)
        assert isinstance(result['energy'], float)


class TestOOMFallback:
    """Tests for GPU OOM detection and CPU fallback"""

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.ort.get_available_providers')
    def test_embedding_oom_fallback_to_cpu(self, mock_providers, mock_audio_load, mock_mel, 
                                           mock_beat, mock_rms, mock_chroma, mock_onnx_session):
        """Test GPU OOM during embedding inference triggers CPU fallback
        
        TESTS: OOM detection and automatic CPU fallback for embedding model
        """
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        # Mock GPU session that fails with OOM on first run
        gpu_session_call_count = [0]
        cpu_session_call_count = [0]
        
        def gpu_run(output_names, feed_dict):
            gpu_session_call_count[0] += 1
            # First call (embedding) should OOM, subsequent calls succeed
            if gpu_session_call_count[0] == 1:
                import onnxruntime as ort
                raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                    "Failed to allocate memory for requested buffer of size 765249024"
                )
            return [np.random.rand(5, 200)]
        
        def cpu_run(output_names, feed_dict):
            cpu_session_call_count[0] += 1
            return [np.random.rand(5, 200)]
        
        # Track created sessions
        sessions_created = []
        
        def create_session(model_path, providers=None, provider_options=None):
            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            
            # Determine if this is a CPU or GPU session
            if isinstance(providers, list) and 'CPUExecutionProvider' in providers and len(providers) == 1:
                # CPU-only session
                mock_session.run.side_effect = cpu_run
                sessions_created.append('CPU')
            else:
                # GPU session
                mock_session.run.side_effect = gpu_run
                sessions_created.append('GPU')
            
            return mock_session
        
        mock_onnx_session.side_effect = create_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        # WHAT WE'RE TESTING: Analysis completes successfully after OOM
        assert result is not None
        assert embeddings is not None
        # Verify CPU fallback session was created
        assert 'CPU' in sessions_created
        assert cpu_session_call_count[0] > 0

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.ort.get_available_providers')
    def test_prediction_oom_fallback_to_cpu(self, mock_providers, mock_audio_load, mock_mel, 
                                            mock_beat, mock_rms, mock_chroma, mock_onnx_session):
        """Test GPU OOM during prediction inference triggers CPU fallback
        
        TESTS: OOM detection and automatic CPU fallback for prediction model
        """
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        gpu_session_call_count = [0]
        cpu_session_call_count = [0]
        
        def gpu_run(output_names, feed_dict):
            gpu_session_call_count[0] += 1
            # Second call (prediction) should OOM
            if gpu_session_call_count[0] == 2:
                import onnxruntime as ort
                raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                    "Failed to allocate memory for requested buffer"
                )
            return [np.random.rand(5, 200)]
        
        def cpu_run(output_names, feed_dict):
            cpu_session_call_count[0] += 1
            return [np.random.rand(5, 200)]
        
        sessions_created = []
        
        def create_session(model_path, providers=None, provider_options=None):
            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            
            if isinstance(providers, list) and 'CPUExecutionProvider' in providers and len(providers) == 1:
                mock_session.run.side_effect = cpu_run
                sessions_created.append('CPU')
            else:
                mock_session.run.side_effect = gpu_run
                sessions_created.append('GPU')
            
            return mock_session
        
        mock_onnx_session.side_effect = create_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        assert result is not None
        assert embeddings is not None
        assert 'CPU' in sessions_created
        assert cpu_session_call_count[0] > 0

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.ort.get_available_providers')
    def test_secondary_model_oom_fallback_to_cpu(self, mock_providers, mock_audio_load, mock_mel, 
                                                  mock_beat, mock_rms, mock_chroma, mock_onnx_session):
        """Test GPU OOM during secondary model inference triggers CPU fallback
        
        TESTS: OOM detection and automatic CPU fallback for secondary models
        """
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        gpu_session_call_count = [0]
        cpu_session_call_count = [0]
        
        def gpu_run(output_names, feed_dict):
            gpu_session_call_count[0] += 1
            # Make secondary models OOM (after embedding and prediction)
            if gpu_session_call_count[0] > 2:
                import onnxruntime as ort
                raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                    "Failed to allocate memory"
                )
            return [np.random.rand(5, 200) if gpu_session_call_count[0] <= 2 else np.random.rand(5, 2)]
        
        def cpu_run(output_names, feed_dict):
            cpu_session_call_count[0] += 1
            return [np.random.rand(5, 2)]
        
        sessions_created = []
        
        def create_session(model_path, providers=None, provider_options=None):
            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            
            if isinstance(providers, list) and 'CPUExecutionProvider' in providers and len(providers) == 1:
                mock_session.run.side_effect = cpu_run
                sessions_created.append('CPU')
            else:
                mock_session.run.side_effect = gpu_run
                sessions_created.append('GPU')
            
            return mock_session
        
        mock_onnx_session.side_effect = create_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        assert result is not None
        assert embeddings is not None
        # Verify CPU fallback sessions were created for secondary models
        assert 'CPU' in sessions_created
        assert cpu_session_call_count[0] > 0

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.ort.get_available_providers')
    def test_non_oom_exception_is_reraised(self, mock_providers, mock_audio_load, mock_mel, 
                                           mock_beat, mock_rms, mock_chroma, mock_onnx_session):
        """Test non-OOM exceptions are re-raised (not caught by OOM handler)
        
        TESTS: Only OOM errors trigger fallback, other errors propagate
        """
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        def gpu_run(output_names, feed_dict):
            import onnxruntime as ort
            # Raise a non-OOM ONNX exception
            raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                "Model execution error: Invalid input shape"
            )
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.side_effect = gpu_run
        mock_onnx_session.return_value = mock_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        # Non-OOM exception should propagate and result in None return
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        # Should return None due to exception (caught by outer try-except)
        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis.ort.get_available_providers')
    def test_successful_gpu_inference_no_fallback(self, mock_providers, mock_audio_load, mock_mel, 
                                                  mock_beat, mock_rms, mock_chroma, mock_onnx_session):
        """Test successful GPU inference doesn't trigger CPU fallback
        
        TESTS: Normal operation - no OOM, no fallback
        """
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)
        
        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)
        
        cpu_fallback_used = [False]
        
        def create_session(model_path, providers=None, provider_options=None):
            # Check if this is a CPU-only fallback session
            if isinstance(providers, list) and 'CPUExecutionProvider' in providers and len(providers) == 1:
                cpu_fallback_used[0] = True
            
            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            
            # Successful inference - no OOM
            call_count = [0]
            def successful_run(output_names, feed_dict):
                call_count[0] += 1
                if call_count[0] <= 2:
                    return [np.random.rand(5, 200)]
                else:
                    return [np.random.rand(5, 2)]
            
            mock_session.run.side_effect = successful_run
            return mock_session
        
        mock_onnx_session.side_effect = create_session
        
        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx'
        }
        
        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)
        
        # Analysis should succeed without CPU fallback
        assert result is not None
        assert embeddings is not None
        assert cpu_fallback_used[0] is False


