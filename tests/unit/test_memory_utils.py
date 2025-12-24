"""
Unit tests for tasks/memory_utils.py
Tests the memory management and data sanitization utilities.
"""

import pytest
import gc
from tasks.memory_utils import (
    sanitize_string_for_db,
    cleanup_cuda_memory,
    cleanup_onnx_session,
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

