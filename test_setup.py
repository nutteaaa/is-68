#!/usr/bin/env python3
"""
Comprehensive test suite for setup.py RAGSetup class
Tests all configuration, file creation, and OpenSearch operations
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, call
from io import StringIO
import requests
from requests.auth import HTTPBasicAuth

# Import the class we're testing
from setup import RAGSetup


class TestRAGSetupInitialization:
    """Test RAGSetup class initialization"""

    def test_init_creates_empty_config(self):
        """Test that initialization creates an empty config dictionary"""
        setup = RAGSetup()
        assert hasattr(setup, 'config')
        assert isinstance(setup.config, dict)
        assert len(setup.config) == 0

    def test_multiple_instances_independent(self):
        """Test that multiple instances have independent configs"""
        setup1 = RAGSetup()
        setup2 = RAGSetup()

        setup1.config['test'] = 'value1'
        setup2.config['test'] = 'value2'

        assert setup1.config['test'] == 'value1'
        assert setup2.config['test'] == 'value2'


class TestSaveEnvFile:
    """Test save_env_file method"""

    def test_save_env_file_creates_file_with_correct_content(self):
        """Test that save_env_file creates .env with proper formatting"""
        setup = RAGSetup()
        setup.config = {
            'OPENAI_API_KEY': 'sk-test123',
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password123',
            'OPENSEARCH_INDEX': 'test_index',
            'OPENAI_MODEL': 'gpt-4o'
        }

        m_open = mock_open()
        with patch('builtins.open', m_open), \
             patch('builtins.print'):
            setup.save_env_file()

        # Verify file was opened for writing
        m_open.assert_called_once_with('.env', 'w')

        # Get all write calls and combine them
        write_calls = m_open().write.call_args_list
        written_content = ''.join(call[0][0] for call in write_calls)

        # Verify content contains all expected lines
        assert 'OPENAI_API_KEY=sk-test123' in written_content
        assert 'OPENSEARCH_ENDPOINT=https://localhost:9200' in written_content
        assert 'OPENSEARCH_USER=admin' in written_content
        assert 'OPENSEARCH_PASSWORD=password123' in written_content
        assert 'OPENSEARCH_INDEX=test_index' in written_content
        assert 'OPENAI_MODEL=gpt-4o' in written_content
        assert '# RAG Chatbot Configuration' in written_content

    def test_save_env_file_prints_success_message(self):
        """Test that save_env_file prints success message"""
        setup = RAGSetup()
        setup.config = {
            'OPENAI_API_KEY': 'test',
            'OPENSEARCH_ENDPOINT': 'test',
            'OPENSEARCH_USER': 'test',
            'OPENSEARCH_PASSWORD': 'test',
            'OPENSEARCH_INDEX': 'test',
            'OPENAI_MODEL': 'test'
        }

        with patch('builtins.open', mock_open()), \
             patch('builtins.print') as mock_print:
            setup.save_env_file()

            # Check that success message was printed
            mock_print.assert_called_with('\n✅ Configuration saved to .env')

    def test_save_env_file_with_special_characters(self):
        """Test that save_env_file handles special characters in values"""
        setup = RAGSetup()
        setup.config = {
            'OPENAI_API_KEY': 'sk-test!@#$%^&*()',
            'OPENSEARCH_ENDPOINT': 'https://test.com:9200',
            'OPENSEARCH_USER': 'user@domain.com',
            'OPENSEARCH_PASSWORD': 'p@ssw0rd!#$',
            'OPENSEARCH_INDEX': 'index-with-dashes',
            'OPENAI_MODEL': 'gpt-4o'
        }

        m_open = mock_open()
        with patch('builtins.open', m_open), \
             patch('builtins.print'):
            setup.save_env_file()

        write_calls = m_open().write.call_args_list
        written_content = ''.join(call[0][0] for call in write_calls)

        assert 'OPENAI_API_KEY=sk-test!@#$%^&*()' in written_content
        assert 'OPENSEARCH_PASSWORD=p@ssw0rd!#$' in written_content


class TestTestOpenSearch:
    """Test test_opensearch method"""

    @patch('requests.get')
    @patch('builtins.print')
    def test_opensearch_connection_success(self, mock_print, mock_get):
        """Test successful OpenSearch connection"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password'
        }

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = setup.test_opensearch()

        assert result is True
        mock_get.assert_called_once_with(
            'https://localhost:9200',
            auth=HTTPBasicAuth('admin', 'password'),
            verify=False,
            timeout=5
        )

        # Verify success messages were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Testing OpenSearch connection' in str(call) for call in print_calls)
        assert any('OpenSearch connection successful' in str(call) for call in print_calls)

    @patch('requests.get')
    @patch('builtins.print')
    def test_opensearch_connection_failure_status_code(self, mock_print, mock_get):
        """Test OpenSearch connection with non-200 status code"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'wrong_password'
        }

        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        result = setup.test_opensearch()

        assert result is False

        # Verify error message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('OpenSearch connection failed: 401' in str(call) for call in print_calls)

    @patch('requests.get')
    @patch('builtins.print')
    def test_opensearch_connection_exception(self, mock_print, mock_get):
        """Test OpenSearch connection with network exception"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://invalid-host:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password'
        }

        # Mock exception
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = setup.test_opensearch()

        assert result is False

        # Verify error message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Error connecting to OpenSearch' in str(call) for call in print_calls)

    @patch('requests.get')
    @patch('builtins.print')
    def test_opensearch_connection_timeout(self, mock_print, mock_get):
        """Test OpenSearch connection timeout"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://slow-host:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password'
        }

        # Mock timeout exception
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        result = setup.test_opensearch()

        assert result is False


class TestCreateIndex:
    """Test create_index method"""

    @patch('requests.delete')
    @patch('requests.put')
    @patch('builtins.print')
    def test_create_index_success(self, mock_print, mock_put, mock_delete):
        """Test successful index creation"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index'
        }

        # Mock successful responses
        mock_delete.return_value = Mock(status_code=200)
        mock_put_response = Mock()
        mock_put_response.status_code = 201
        mock_put.return_value = mock_put_response

        result = setup.create_index()

        assert result is True

        # Verify delete was called to remove existing index
        mock_delete.assert_called_once_with(
            'https://localhost:9200/test_index',
            auth=HTTPBasicAuth('admin', 'password'),
            verify=False
        )

        # Verify PUT request was made with correct settings
        assert mock_put.called
        put_call_args = mock_put.call_args
        assert put_call_args[0][0] == 'https://localhost:9200/test_index'

        # Verify index settings
        index_settings = put_call_args[1]['json']
        assert 'settings' in index_settings
        assert 'mappings' in index_settings
        assert index_settings['settings']['index']['knn'] is True
        assert index_settings['mappings']['properties']['embedding']['dimension'] == 1536

    @patch('requests.delete')
    @patch('requests.put')
    @patch('builtins.print')
    def test_create_index_with_200_status(self, mock_print, mock_put, mock_delete):
        """Test index creation with 200 status code (also valid)"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index'
        }

        mock_delete.return_value = Mock(status_code=200)
        mock_put_response = Mock()
        mock_put_response.status_code = 200  # 200 is also valid
        mock_put.return_value = mock_put_response

        result = setup.create_index()

        assert result is True

    @patch('requests.delete')
    @patch('requests.put')
    @patch('builtins.print')
    def test_create_index_failure(self, mock_print, mock_put, mock_delete):
        """Test failed index creation"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index'
        }

        mock_delete.return_value = Mock(status_code=200)
        mock_put_response = Mock()
        mock_put_response.status_code = 400
        mock_put_response.text = 'Bad request'
        mock_put.return_value = mock_put_response

        result = setup.create_index()

        assert result is False

        # Verify error message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Failed to create index' in str(call) for call in print_calls)

    @patch('requests.delete')
    @patch('requests.put')
    @patch('builtins.print')
    def test_create_index_exception(self, mock_print, mock_put, mock_delete):
        """Test index creation with exception"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index'
        }

        # Mock exception during PUT request
        mock_delete.return_value = Mock(status_code=200)
        mock_put.side_effect = Exception("Network error")

        result = setup.create_index()

        assert result is False

        # Verify error message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Error creating index' in str(call) for call in print_calls)

    @patch('requests.delete')
    @patch('requests.put')
    @patch('builtins.print')
    def test_create_index_settings_structure(self, mock_print, mock_put, mock_delete):
        """Test that index settings have correct structure"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index'
        }

        mock_delete.return_value = Mock(status_code=200)
        mock_put_response = Mock()
        mock_put_response.status_code = 201
        mock_put.return_value = mock_put_response

        setup.create_index()

        # Extract the index settings from the PUT call
        index_settings = mock_put.call_args[1]['json']

        # Verify settings structure
        assert 'settings' in index_settings
        assert 'mappings' in index_settings

        # Verify KNN settings
        assert index_settings['settings']['index']['knn'] is True
        assert index_settings['settings']['index']['knn.algo_param.ef_search'] == 100

        # Verify Thai analyzer
        assert 'analysis' in index_settings['settings']
        assert 'thai_analyzer' in index_settings['settings']['analysis']['analyzer']

        # Verify mappings
        mappings = index_settings['mappings']['properties']
        assert 'content' in mappings
        assert mappings['content']['type'] == 'text'
        assert mappings['content']['analyzer'] == 'thai_analyzer'

        # Verify embedding field
        assert 'embedding' in mappings
        assert mappings['embedding']['type'] == 'knn_vector'
        assert mappings['embedding']['dimension'] == 1536
        assert mappings['embedding']['method']['name'] == 'hnsw'
        assert mappings['embedding']['method']['space_type'] == 'cosinesimil'

        # Verify metadata field
        assert 'metadata' in mappings
        assert mappings['metadata']['type'] == 'object'


class TestDownloadDocuments:
    """Test download_documents method"""

    @patch('builtins.print')
    def test_download_documents_no_sources(self, mock_print):
        """Test download_documents with no sources provided"""
        setup = RAGSetup()
        setup.config = {}

        setup.download_documents()

        # Verify skip message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('No document sources provided' in str(call) for call in print_calls)

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    def test_download_documents_success(self, mock_print, mock_mkdir, mock_file, mock_get):
        """Test successful document download"""
        setup = RAGSetup()
        setup.config = {
            'DOCUMENT_SOURCES': [
                'https://example.com/doc1.md',
                'https://example.com/doc2.md'
            ]
        }

        # Mock successful responses
        mock_response1 = Mock()
        mock_response1.text = '# Document 1\nContent here'
        mock_response1.raise_for_status = Mock()

        mock_response2 = Mock()
        mock_response2.text = '# Document 2\nMore content'
        mock_response2.raise_for_status = Mock()

        mock_get.side_effect = [mock_response1, mock_response2]

        setup.download_documents()

        # Verify directory was created
        mock_mkdir.assert_called_once_with(exist_ok=True)

        # Verify both documents were requested
        assert mock_get.call_count == 2
        mock_get.assert_any_call('https://example.com/doc1.md', timeout=30)
        mock_get.assert_any_call('https://example.com/doc2.md', timeout=30)

        # Verify files were written
        assert mock_file.call_count == 2

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    def test_download_documents_with_failure(self, mock_print, mock_mkdir, mock_file, mock_get):
        """Test document download with one failure"""
        setup = RAGSetup()
        setup.config = {
            'DOCUMENT_SOURCES': [
                'https://example.com/doc1.md',
                'https://example.com/invalid.md'
            ]
        }

        # First succeeds, second fails
        mock_response1 = Mock()
        mock_response1.text = '# Document 1'
        mock_response1.raise_for_status = Mock()

        mock_get.side_effect = [
            mock_response1,
            requests.exceptions.HTTPError("404 Not Found")
        ]

        setup.download_documents()

        # Should continue despite error
        assert mock_get.call_count == 2

        # Verify error was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Error' in str(call) for call in print_calls)

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    def test_download_documents_timeout(self, mock_print, mock_mkdir, mock_file, mock_get):
        """Test document download with timeout"""
        setup = RAGSetup()
        setup.config = {
            'DOCUMENT_SOURCES': ['https://slow-site.com/doc.md']
        }

        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        setup.download_documents()

        # Verify timeout parameter was used
        mock_get.assert_called_once_with('https://slow-site.com/doc.md', timeout=30)

        # Verify error was handled
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Error' in str(call) for call in print_calls)


class TestCreateStreamlitConfig:
    """Test create_streamlit_config method"""

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    def test_create_streamlit_config_success(self, mock_print, mock_mkdir, mock_file):
        """Test successful Streamlit config creation"""
        setup = RAGSetup()
        setup.config = {
            'OPENAI_API_KEY': 'sk-test123',
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index'
        }

        setup.create_streamlit_config()

        # Verify directory was created
        mock_mkdir.assert_called_once_with(exist_ok=True)

        # Verify file was opened for writing
        # The Path object creates the full path, so we check it was opened
        assert mock_file.called

        # Get written content
        write_calls = mock_file().write.call_args_list
        written_content = ''.join(call[0][0] for call in write_calls)

        # Verify TOML format
        assert '[secrets]' in written_content
        assert 'OPENAI_API_KEY = "sk-test123"' in written_content
        assert 'OPENSEARCH_ENDPOINT = "https://localhost:9200"' in written_content
        assert 'OPENSEARCH_USER = "admin"' in written_content
        assert 'OPENSEARCH_PASSWORD = "password"' in written_content
        assert 'OPENSEARCH_INDEX = "test_index"' in written_content

        # Verify success message
        mock_print.assert_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    def test_create_streamlit_config_with_special_characters(self, mock_print, mock_mkdir, mock_file):
        """Test Streamlit config creation with special characters in values"""
        setup = RAGSetup()
        setup.config = {
            'OPENAI_API_KEY': 'sk-test!@#$',
            'OPENSEARCH_ENDPOINT': 'https://test.com:9200',
            'OPENSEARCH_USER': 'user@domain',
            'OPENSEARCH_PASSWORD': 'p@ss"word',
            'OPENSEARCH_INDEX': 'index-name'
        }

        setup.create_streamlit_config()

        # Get written content
        write_calls = mock_file().write.call_args_list
        written_content = ''.join(call[0][0] for call in write_calls)

        # Verify values are properly quoted in TOML format
        assert 'OPENAI_API_KEY = "sk-test!@#$"' in written_content
        assert 'OPENSEARCH_PASSWORD = "p@ss"word"' in written_content


class TestWelcome:
    """Test welcome method"""

    @patch('builtins.print')
    def test_welcome_prints_banner(self, mock_print):
        """Test that welcome prints the banner"""
        setup = RAGSetup()
        setup.welcome()

        # Verify multiple print calls for the banner
        assert mock_print.call_count >= 3

        # Check that banner elements are present
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('RAG CHATBOT SETUP WIZARD' in str(call) for call in print_calls)
        assert any('=' * 80 in str(call) for call in print_calls)


class TestIntegration:
    """Integration tests for RAGSetup workflow"""

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('requests.get')
    @patch('requests.delete')
    @patch('requests.put')
    @patch('builtins.print')
    def test_full_setup_workflow_success(self, mock_print, mock_put, mock_delete,
                                         mock_get, mock_mkdir, mock_file):
        """Test complete setup workflow from start to finish"""
        setup = RAGSetup()

        # Set up config
        setup.config = {
            'OPENAI_API_KEY': 'sk-test123',
            'OPENSEARCH_ENDPOINT': 'https://localhost:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password',
            'OPENSEARCH_INDEX': 'test_index',
            'OPENAI_MODEL': 'gpt-4o',
            'DOCUMENT_SOURCES': ['https://example.com/doc.md']
        }

        # Mock successful OpenSearch connection
        mock_opensearch_response = Mock()
        mock_opensearch_response.status_code = 200

        # Mock successful document download
        mock_doc_response = Mock()
        mock_doc_response.text = '# Test Document'
        mock_doc_response.raise_for_status = Mock()

        mock_get.side_effect = [mock_opensearch_response, mock_doc_response]

        # Mock successful index creation
        mock_delete.return_value = Mock(status_code=200)
        mock_put.return_value = Mock(status_code=201)

        # Execute workflow
        setup.save_env_file()
        opensearch_ok = setup.test_opensearch()
        index_ok = setup.create_index()
        setup.download_documents()
        setup.create_streamlit_config()

        # Verify all steps succeeded
        assert opensearch_ok is True
        assert index_ok is True

        # Verify files were created
        assert mock_file.call_count >= 2  # .env and secrets.toml

    @patch('requests.get')
    @patch('builtins.print')
    def test_workflow_handles_opensearch_failure(self, mock_print, mock_get):
        """Test that workflow handles OpenSearch connection failure gracefully"""
        setup = RAGSetup()
        setup.config = {
            'OPENSEARCH_ENDPOINT': 'https://invalid:9200',
            'OPENSEARCH_USER': 'admin',
            'OPENSEARCH_PASSWORD': 'password'
        }

        # Mock failed connection
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = setup.test_opensearch()

        assert result is False
        # Should not crash, just return False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
