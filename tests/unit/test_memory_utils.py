"""
Unit tests for tasks/memory_utils.py
Tests the memory management and data sanitization utilities.
"""

import pytest
from unittest.mock import Mock, MagicMock

from tasks.memory_utils import (
    sanitize_string_for_db,
    sanitize_json_for_db,
    cleanup_cuda_memory,
    cleanup_onnx_session,
    comprehensive_memory_cleanup,
    handle_onnx_memory_error,
    SessionRecycler
)


class TestSanitizeStringForDB:
    """Test string sanitization for database writes."""
    
    def test_remove_null_bytes(self):
        """Test removal of NULL bytes (0x00)."""
        result = sanitize_string_for_db("hello\x00world")
        assert result == "helloworld"
    
    def test_remove_control_characters(self):
        """Test removal of control characters."""
        result = sanitize_string_for_db("test\x01\x02\x03text")
        assert result == "testtext"
    
    def test_preserve_valid_characters(self):
        """Test that valid characters are preserved."""
        text = "Normal Text 123 !@#$%^&*()"
        result = sanitize_string_for_db(text)
        assert result == text
    
    def test_preserve_unicode(self):
        """Test that Unicode characters are preserved."""
        text = "Hello 世界 Привет"
        result = sanitize_string_for_db(text)
        assert result == text
    
    def test_handle_none(self):
        """Test that None input returns None."""
        result = sanitize_string_for_db(None)
        assert result is None
    
    def test_handle_empty_string(self):
        """Test that empty string returns empty string."""
        result = sanitize_string_for_db("")
        assert result == ""
    
    def test_complex_corruption(self):
        """Test handling of multiple types of corruption."""
        text = "Artist\x00Name\x01With\x02Control\x03Chars"
        result = sanitize_string_for_db(text)
        assert result == "ArtistNameWithControlChars"
    
    def test_preserve_whitespace(self):
        """Test that valid whitespace (tab, newline, etc.) is preserved."""
        text = "Line 1\nLine 2\tTabbed"
        result = sanitize_string_for_db(text)
        assert result == text


class TestCleanupCudaMemory:
    """Test CUDA memory cleanup function."""
    
    def test_cleanup_returns_bool(self):
        """Test that cleanup function returns a boolean."""
        result = cleanup_cuda_memory(force=False)
        assert isinstance(result, bool)
    
    def test_cleanup_does_not_raise(self):
        """Test that cleanup does not raise exceptions."""
        cleanup_cuda_memory(force=True)
        cleanup_cuda_memory(force=False)


class TestCleanupOnnxSession:
    """Test ONNX session cleanup function."""
    
    def test_cleanup_none_session(self):
        """Test that cleanup handles None session gracefully."""
        cleanup_onnx_session(None, "test_session")
    
    def test_cleanup_mock_session(self):
        """Test cleanup with a mock session object."""
        class MockSession:
            pass
        
        session = MockSession()
        cleanup_onnx_session(session, "mock_session")


class TestSessionRecycler:
    """Test SessionRecycler class."""
    
    def test_initialization(self):
        """Test SessionRecycler initialization."""
        recycler = SessionRecycler(recycle_interval=10)
        assert recycler.recycle_interval == 10
        assert recycler.use_count == 0
    
    def test_default_interval(self):
        """Test default recycling interval."""
        recycler = SessionRecycler()
        assert recycler.recycle_interval == 20
    
    def test_increment(self):
        """Test usage counter increment."""
        recycler = SessionRecycler(recycle_interval=5)
        assert recycler.use_count == 0
        
        recycler.increment()
        assert recycler.use_count == 1
        
        recycler.increment()
        assert recycler.use_count == 2
    
    def test_should_recycle_before_interval(self):
        """Test that should_recycle returns False before interval."""
        recycler = SessionRecycler(recycle_interval=5)
        
        for i in range(4):
            recycler.increment()
            assert not recycler.should_recycle()
    
    def test_should_recycle_at_interval(self):
        """Test that should_recycle returns True at interval."""
        recycler = SessionRecycler(recycle_interval=5)
        
        for i in range(5):
            recycler.increment()
        
        assert recycler.should_recycle()
    
    def test_should_recycle_after_interval(self):
        """Test that should_recycle returns True after interval."""
        recycler = SessionRecycler(recycle_interval=5)
        
        for i in range(10):
            recycler.increment()
        
        assert recycler.should_recycle()
    
    def test_mark_recycled(self):
        """Test that mark_recycled resets counter."""
        recycler = SessionRecycler(recycle_interval=5)
        
        for i in range(5):
            recycler.increment()
        
        assert recycler.should_recycle()
        
        recycler.mark_recycled()
        assert recycler.use_count == 0
        assert not recycler.should_recycle()
    
    def test_get_use_count(self):
        """Test get_use_count method."""
        recycler = SessionRecycler(recycle_interval=5)
        
        assert recycler.get_use_count() == 0
        
        recycler.increment()
        assert recycler.get_use_count() == 1
        
        recycler.increment()
        recycler.increment()
        assert recycler.get_use_count() == 3
    
    def test_reset(self):
        """Test reset method."""
        recycler = SessionRecycler(recycle_interval=5)
        
        for i in range(3):
            recycler.increment()
        
        recycler.reset()
        assert recycler.use_count == 0
    
    def test_full_cycle(self):
        """Test a full recycling cycle."""
        recycler = SessionRecycler(recycle_interval=3)
        
        # First cycle
        for i in range(3):
            assert not recycler.should_recycle()
            recycler.increment()
        
        assert recycler.should_recycle()
        recycler.mark_recycled()
        
        # Second cycle
        for i in range(3):
            assert not recycler.should_recycle()
            recycler.increment()

        assert recycler.should_recycle()


class TestHandleOnnxMemoryError:
    """Test ONNX memory error handling with cleanup, retry and CPU fallback."""

    def test_non_memory_error_reraises_same_object(self):
        """A non-memory error is re-raised as the same exception object."""
        err = ValueError("boom")
        cleanup = Mock()
        retry = Mock()

        with pytest.raises(ValueError) as excinfo:
            handle_onnx_memory_error(
                err, "test context", cleanup_func=cleanup, retry_func=retry
            )

        assert excinfo.value is err
        cleanup.assert_not_called()
        retry.assert_not_called()

    def test_memory_error_triggers_cleanup_and_returns_retry_result(self):
        """A BFCArena error calls cleanup once and returns the retry result."""
        err = Exception("BFCArena failed")
        cleanup = Mock()
        retry = Mock(return_value="retried")

        result = handle_onnx_memory_error(
            err, "test context", cleanup_func=cleanup, retry_func=retry
        )

        assert result == "retried"
        cleanup.assert_called_once()
        retry.assert_called_once()

    def test_cpu_fallback_returns_result_session_provider_tuple(self):
        """CPU fallback returns (result, new_session, provider)."""
        err = Exception("BFCArena failed")
        session_mock = MagicMock()
        creator = Mock(return_value=(session_mock, "CPUExecutionProvider"))
        retry = Mock(return_value="r")

        result = handle_onnx_memory_error(
            err,
            "test context",
            retry_func=retry,
            fallback_to_cpu=True,
            session_creator=creator,
        )

        assert result == ("r", session_mock, "CPUExecutionProvider")
        creator.assert_called_once()
        retry.assert_called_once()

    def test_retry_failure_propagates_retry_exception(self):
        """If the retry itself fails, that exception propagates."""
        err = Exception("Failed to allocate memory for requested buffer")
        retry_err = RuntimeError("still failing")
        retry = Mock(side_effect=retry_err)

        with pytest.raises(RuntimeError) as excinfo:
            handle_onnx_memory_error(err, "test context", retry_func=retry)

        assert excinfo.value is retry_err

    def test_memory_error_without_retry_or_fallback_reraises_original(self):
        """A memory error with no retry and no fallback re-raises the original."""
        err = Exception("out of memory")

        with pytest.raises(Exception) as excinfo:
            handle_onnx_memory_error(err, "test context")

        assert excinfo.value is err


class TestComprehensiveMemoryCleanup:
    """Test combined memory cleanup result dictionary."""

    def test_returns_expected_keys_with_bool_values(self):
        """Result dict has exactly the four expected keys, all booleans."""
        results = comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)

        assert set(results.keys()) == {"cuda", "onnx_pool", "gc", "malloc_trim"}
        assert all(isinstance(v, bool) for v in results.values())

    def test_gc_is_always_true(self):
        """Garbage collection is always reported as successful."""
        results = comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
        assert results["gc"] is True

    def test_force_cuda_false_reports_cuda_false(self):
        """Disabling CUDA cleanup leaves the cuda flag False."""
        results = comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
        assert results["cuda"] is False

    def test_reset_onnx_pool_false_reports_onnx_pool_false(self):
        """Disabling pool reset leaves the onnx_pool flag False."""
        results = comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
        assert results["onnx_pool"] is False


class TestSanitizeJsonForDB:
    """Test recursive JSON sanitization for jsonb writes."""

    def test_nested_structures_are_cleaned_recursively(self):
        """Strings at every depth are cleaned; containers keep their types."""
        data = {
            "a": {"b": "Va\x00l"},
            "l": [1, "S\x01tr", {"n": "B\x00ad"}],
            "t": ("x\x00y",),
        }

        result = sanitize_json_for_db(data)

        assert result == {
            "a": {"b": "Val"},
            "l": [1, "Str", {"n": "Bad"}],
            "t": ("xy",),
        }
        assert isinstance(result["t"], tuple)
        assert isinstance(result["l"], list)
        assert result["l"][0] == 1

    def test_non_string_scalars_pass_through(self):
        """Non-string scalars are returned unchanged."""
        assert sanitize_json_for_db(123) == 123
        assert sanitize_json_for_db(None) is None

