"""Tests for utility modules."""

import pytest
import logging
from pathlib import Path
from src.utils.logger import setup_logger, get_logger, ColoredFormatter
from src.utils.file_handler import FileHandler


class TestLogger:
    """Tests for logger utility."""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
    
    def test_setup_logger_with_level(self):
        """Test logger with custom level."""
        logger = setup_logger("test_debug", level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logger_no_duplicates(self):
        """Test that repeated calls don't add duplicate handlers."""
        logger1 = setup_logger("test_dup")
        handler_count1 = len(logger1.handlers)
        
        logger2 = setup_logger("test_dup")
        handler_count2 = len(logger2.handlers)
        
        assert handler_count1 == handler_count2
        assert logger1 is logger2
    
    def test_setup_logger_with_file(self, tmp_path):
        """Test logger with file output."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file", log_file=log_file)
        
        logger.info("Test message")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
    
    def test_get_logger(self):
        """Test getting existing logger."""
        setup_logger("test_get")
        logger = get_logger("test_get")
        
        assert logger.name == "test_get"
    
    def test_colored_formatter(self):
        """Test colored formatter."""
        formatter = ColoredFormatter(
            fmt='%(levelname)s - %(message)s'
        )
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "Test message" in formatted


class TestFileHandler:
    """Tests for file handler utility."""
    
    def test_read_file(self, tmp_path):
        """Test reading a file."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        content = FileHandler.read_file(test_file)
        
        assert content == test_content
    
    def test_read_file_not_found(self, tmp_path):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            FileHandler.read_file(tmp_path / "nonexistent.txt")
    
    def test_write_file(self, tmp_path):
        """Test writing a file."""
        test_file = tmp_path / "output.txt"
        test_content = "Test content"
        
        FileHandler.write_file(test_file, test_content)
        
        assert test_file.exists()
        assert test_file.read_text() == test_content
    
    def test_write_file_creates_dirs(self, tmp_path):
        """Test that write_file creates parent directories."""
        test_file = tmp_path / "subdir" / "nested" / "file.txt"
        
        FileHandler.write_file(test_file, "content")
        
        assert test_file.exists()
        assert test_file.parent.exists()
    
    def test_read_json(self, tmp_path):
        """Test reading JSON file."""
        import json
        
        test_file = tmp_path / "data.json"
        test_data = {"key": "value", "number": 42}
        test_file.write_text(json.dumps(test_data))
        
        data = FileHandler.read_json(test_file)
        
        assert data == test_data
    
    def test_write_json(self, tmp_path):
        """Test writing JSON file."""
        test_file = tmp_path / "output.json"
        test_data = {"name": "test", "values": [1, 2, 3]}
        
        FileHandler.write_json(test_file, test_data)
        
        assert test_file.exists()
        
        import json
        loaded_data = json.loads(test_file.read_text())
        assert loaded_data == test_data
    
    def test_write_json_with_indent(self, tmp_path):
        """Test JSON formatting."""
        test_file = tmp_path / "formatted.json"
        test_data = {"a": 1}
        
        FileHandler.write_json(test_file, test_data, indent=4)
        
        content = test_file.read_text()
        assert "\n" in content  # Should be formatted
    
    def test_create_blog_structure(self, tmp_path):
        """Test creating blog directory structure."""
        paths = FileHandler.create_blog_structure("test-blog", tmp_path)
        
        assert "blog_dir" in paths
        assert "draft" in paths
        assert "final_md" in paths
        assert "final_html" in paths
        assert "image" in paths
        assert "metadata" in paths
        
        # Check directories were created
        assert paths["draft"].parent.exists()
        assert paths["final_md"].parent.exists()
        assert paths["image"].parent.exists()
    
    def test_create_blog_structure_safe_name(self, tmp_path):
        """Test blog structure with special characters in name."""
        paths = FileHandler.create_blog_structure(
            "Test Blog! With @Special# Characters",
            tmp_path
        )
        
        # Name should be sanitized
        blog_dir_name = paths["blog_dir"].name
        assert "@" not in blog_dir_name
        assert "#" not in blog_dir_name
        assert "!" not in blog_dir_name
    
    def test_save_blog_output(self, tmp_path):
        """Test saving all blog outputs."""
        paths = FileHandler.create_blog_structure("test", tmp_path)
        
        draft = "# Draft"
        final_md = "# Final"
        html = "<html></html>"
        metadata = {"score": 8.5}
        
        FileHandler.save_blog_output(
            paths,
            draft,
            final_md,
            html,
            metadata
        )
        
        assert paths["draft"].exists()
        assert paths["final_md"].exists()
        assert paths["final_html"].exists()
        assert paths["metadata"].exists()
        
        assert paths["draft"].read_text() == draft
        assert paths["final_md"].read_text() == final_md
        assert paths["final_html"].read_text() == html
        
        import json
        saved_metadata = json.loads(paths["metadata"].read_text())
        assert saved_metadata == metadata
