"""
Tests for ContextStoreClient.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

from energyai_sdk.clients.context_store_client import ContextStoreClient


class TestContextStoreClient:
    """Test cases for ContextStoreClient."""

    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        config_manager = MagicMock()
        config_manager.get_settings.return_value = {
            'cosmos_endpoint': 'https://test.documents.azure.com:443/',
            'cosmos_key': 'test_key',
            'cosmos_database': 'TestDB',
            'cosmos_container': 'TestContainer'
        }
        return config_manager

    @pytest.fixture
    def mock_cosmos_client(self):
        """Mock Cosmos client and dependencies."""
        with patch('energyai_sdk.clients.context_store_client.CosmosClient') as mock_client:
            # Mock the client hierarchy
            mock_database = MagicMock()
            mock_container = MagicMock()
            
            mock_client.return_value.get_database_client.return_value = mock_database
            mock_database.get_container_client.return_value = mock_container
            
            yield {
                'client': mock_client,
                'database': mock_database,
                'container': mock_container
            }

    @pytest.fixture
    def client(self, mock_config_manager, mock_cosmos_client):
        """Create ContextStoreClient instance."""
        return ContextStoreClient(config_manager=mock_config_manager)

    def test_init_success(self, mock_config_manager, mock_cosmos_client):
        """Test successful initialization."""
        client = ContextStoreClient(config_manager=mock_config_manager)
        
        assert client.config_manager == mock_config_manager
        assert client.client is not None
        assert client.database is not None
        assert client.container is not None

    def test_init_missing_dependency(self):
        """Test initialization without azure-cosmos dependency."""
        with patch('energyai_sdk.clients.context_store_client.CosmosClient', None):
            with pytest.raises(ImportError, match="azure-cosmos is required"):
                ContextStoreClient()

    def test_init_missing_config(self, mock_cosmos_client):
        """Test initialization with missing configuration."""
        config_manager = MagicMock()
        config_manager.get_settings.return_value = {}
        
        with pytest.raises(ValueError, match="Cosmos DB configuration missing"):
            ContextStoreClient(config_manager=config_manager)

    def test_get_cosmos_config_from_dict(self, mock_config_manager):
        """Test configuration retrieval from dictionary."""
        client = ContextStoreClient.__new__(ContextStoreClient)
        client.config_manager = mock_config_manager
        
        config = client._get_cosmos_config()
        
        assert config['endpoint'] == 'https://test.documents.azure.com:443/'
        assert config['key'] == 'test_key'
        assert config['database_name'] == 'TestDB'
        assert config['container_name'] == 'TestContainer'

    def test_get_cosmos_config_from_env(self, mock_cosmos_client):
        """Test configuration retrieval from environment variables."""
        config_manager = MagicMock()
        config_manager.get_settings.return_value = {}
        
        with patch.dict('os.environ', {
            'COSMOS_ENDPOINT': 'https://env.documents.azure.com:443/',
            'COSMOS_KEY': 'env_key'
        }):
            client = ContextStoreClient.__new__(ContextStoreClient)
            client.config_manager = config_manager
            
            config = client._get_cosmos_config()
            
            assert config['endpoint'] == 'https://env.documents.azure.com:443/'
            assert config['key'] == 'env_key'

    def test_load_or_create_existing_session(self, client, mock_cosmos_client):
        """Test loading an existing session."""
        # Mock existing session
        existing_session = {
            'id': 'test_session',
            'subject': {'type': 'user', 'id': 'user123'},
            'thread': [{'id': '1', 'sender': 'user', 'content': 'hello'}]
        }
        mock_cosmos_client['container'].read_item.return_value = existing_session
        
        result = client.load_or_create('test_session', 'user123')
        
        assert result == existing_session
        mock_cosmos_client['container'].read_item.assert_called_once_with(
            item='test_session', 
            partition_key='user123'
        )

    def test_load_or_create_new_session(self, client, mock_cosmos_client):
        """Test creating a new session when none exists."""
        # Mock cosmos exception
        class MockCosmosResourceNotFoundError(Exception):
            pass
        
        with patch('energyai_sdk.clients.context_store_client.exceptions') as mock_exceptions:
            mock_exceptions.CosmosResourceNotFoundError = MockCosmosResourceNotFoundError
            
            # Mock session not found
            mock_cosmos_client['container'].read_item.side_effect = MockCosmosResourceNotFoundError()
        
            # Mock successful creation
            created_session = {
                'id': 'new_session',
                'subject': {'type': 'user', 'id': 'user456'},
                'thread': []
            }
            mock_cosmos_client['container'].create_item.return_value = created_session
            
            result = client.load_or_create('new_session', 'user456')
            
            assert result == created_session
            mock_cosmos_client['container'].create_item.assert_called_once()

    def test_update_and_save(self, client, mock_cosmos_client):
        """Test updating and saving session with new messages."""
        context_doc = {
            'id': 'test_session',
            'thread': [],
            'updated_at': '2024-01-01T00:00:00Z'
        }
        
        updated_doc = context_doc.copy()
        updated_doc['thread'] = [
            {'id': 'msg1', 'sender': 'user', 'content': 'Hello'},
            {'id': 'msg2', 'sender': 'agent', 'content': 'Hi there!'}
        ]
        mock_cosmos_client['container'].upsert_item.return_value = updated_doc
        
        with patch('uuid.uuid4', side_effect=['msg1', 'msg2']):
            with patch('energyai_sdk.clients.context_store_client.datetime') as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = '2024-01-01T12:00:00Z'
                mock_datetime.timezone = timezone
                
                result = client.update_and_save(context_doc, 'Hello', 'Hi there!')
        
        # Verify the context was updated
        assert len(context_doc['thread']) == 2
        assert context_doc['thread'][0]['content'] == 'Hello'
        assert context_doc['thread'][1]['content'] == 'Hi there!'
        assert context_doc['updated_at'] == '2024-01-01T12:00:00Z'
        
        mock_cosmos_client['container'].upsert_item.assert_called_once_with(body=context_doc)

    def test_update_and_save_with_agent_name(self, client, mock_cosmos_client):
        """Test updating with agent name included."""
        context_doc = {'id': 'test_session', 'thread': []}
        
        mock_cosmos_client['container'].upsert_item.return_value = context_doc
        
        with patch('uuid.uuid4', return_value='test_id'):
            with patch('energyai_sdk.clients.context_store_client.datetime') as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = '2024-01-01T12:00:00Z'
                mock_datetime.timezone = timezone
                
                client.update_and_save(context_doc, 'Hello', 'Hi there!', agent_name='TestAgent')
        
        agent_message = context_doc['thread'][1]  # Second message is agent
        assert agent_message['agent_name'] == 'TestAgent'

    def test_get_conversation_history(self, client, mock_cosmos_client):
        """Test retrieving conversation history."""
        session_doc = {
            'thread': [
                {'id': '1', 'sender': 'user', 'content': 'msg1'},
                {'id': '2', 'sender': 'agent', 'content': 'msg2'},
                {'id': '3', 'sender': 'user', 'content': 'msg3'},
            ]
        }
        mock_cosmos_client['container'].read_item.return_value = session_doc
        
        result = client.get_conversation_history('test_session', 'user123')
        
        assert len(result) == 3
        assert result[0]['content'] == 'msg1'

    def test_get_conversation_history_with_limit(self, client, mock_cosmos_client):
        """Test retrieving conversation history with limit."""
        session_doc = {
            'thread': [
                {'id': '1', 'sender': 'user', 'content': 'msg1'},
                {'id': '2', 'sender': 'agent', 'content': 'msg2'},
                {'id': '3', 'sender': 'user', 'content': 'msg3'},
            ]
        }
        mock_cosmos_client['container'].read_item.return_value = session_doc
        
        result = client.get_conversation_history('test_session', 'user123', limit=2)
        
        assert len(result) == 2
        assert result[0]['content'] == 'msg2'  # Last 2 messages
        assert result[1]['content'] == 'msg3'

    def test_get_conversation_history_session_not_found(self, client, mock_cosmos_client):
        """Test retrieving history for non-existent session."""
        # Mock cosmos exception
        class MockCosmosResourceNotFoundError(Exception):
            pass
        
        with patch('energyai_sdk.clients.context_store_client.exceptions') as mock_exceptions:
            mock_exceptions.CosmosResourceNotFoundError = MockCosmosResourceNotFoundError
            
            mock_cosmos_client['container'].read_item.side_effect = MockCosmosResourceNotFoundError()
        
            result = client.get_conversation_history('nonexistent', 'user123')
            
            assert result == []

    def test_update_session_metadata(self, client, mock_cosmos_client):
        """Test updating session metadata."""
        session_doc = {
            'id': 'test_session',
            'metadata': {'existing': 'value'},
            'updated_at': '2024-01-01T00:00:00Z'
        }
        mock_cosmos_client['container'].read_item.return_value = session_doc
        mock_cosmos_client['container'].upsert_item.return_value = session_doc
        
        new_metadata = {'new_key': 'new_value'}
        
        with patch('energyai_sdk.clients.context_store_client.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2024-01-01T12:00:00Z'
            mock_datetime.timezone = timezone
            
            result = client.update_session_metadata('test_session', 'user123', new_metadata)
        
        assert session_doc['metadata']['existing'] == 'value'
        assert session_doc['metadata']['new_key'] == 'new_value'
        assert session_doc['updated_at'] == '2024-01-01T12:00:00Z'

    def test_close_session(self, client, mock_cosmos_client):
        """Test closing a session."""
        session_doc = {
            'id': 'test_session',
            'status': 'active',
            'updated_at': '2024-01-01T00:00:00Z'
        }
        mock_cosmos_client['container'].read_item.return_value = session_doc
        mock_cosmos_client['container'].upsert_item.return_value = session_doc
        
        with patch('energyai_sdk.clients.context_store_client.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2024-01-01T12:00:00Z'
            mock_datetime.timezone = timezone
            
            result = client.close_session('test_session', 'user123')
        
        assert session_doc['status'] == 'closed'
        assert session_doc['updated_at'] == '2024-01-01T12:00:00Z'

    def test_create_new_session_document(self, client):
        """Test creating a new session document."""
        with patch('energyai_sdk.clients.context_store_client.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2024-01-01T12:00:00Z'
            mock_datetime.timezone = timezone
            
            doc = client._create_new_session_document('session123', 'user456')
        
        assert doc['id'] == 'session123'
        assert doc['subject']['id'] == 'user456'
        assert doc['subject']['type'] == 'user'
        assert doc['session_type'] == 'chat'
        assert doc['status'] == 'active'
        assert doc['created_at'] == '2024-01-01T12:00:00Z'
        assert doc['updated_at'] == '2024-01-01T12:00:00Z'
        assert doc['metadata']['schema_version'] == '1.5.0'
        assert doc['thread'] == []
        assert doc['context']['memory'] == []
        assert doc['context']['state'] == {}

    def test_create_new_session_document_custom_subject_type(self, client):
        """Test creating a new session document with custom subject type."""
        with patch('energyai_sdk.clients.context_store_client.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2024-01-01T12:00:00Z'
            mock_datetime.timezone = timezone
            
            doc = client._create_new_session_document('session123', 'org456', 'organization')
        
        assert doc['subject']['id'] == 'org456'
        assert doc['subject']['type'] == 'organization'
