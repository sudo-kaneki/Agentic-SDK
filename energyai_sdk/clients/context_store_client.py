"""
Context Store Client for Azure Cosmos DB integration.

This module provides stateful, multi-turn conversation capabilities by connecting
the SDK to an Azure Cosmos DB container that stores session history.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from azure.cosmos import CosmosClient, exceptions
except ImportError:
    CosmosClient = None
    exceptions = None

# Avoid circular import - import at runtime
# from ..config import ConfigurationManager


class SimpleConfigManager:
    """Simple configuration manager fallback to avoid circular imports."""

    def get_settings(self):
        """Get settings from environment variables."""
        return {
            "cosmos_endpoint": os.getenv("COSMOS_ENDPOINT") or os.getenv("AZURE_COSMOS_ENDPOINT"),
            "cosmos_key": os.getenv("COSMOS_KEY") or os.getenv("AZURE_COSMOS_KEY"),
            "cosmos_database": os.getenv("COSMOS_DATABASE", "AgenticPlatform"),
            "cosmos_container": os.getenv("COSMOS_CONTAINER", "Context"),
        }


class ContextStoreClient:
    """
    Client for managing conversation context in Azure Cosmos DB.

    This client handles session creation, retrieval, and updates for stateful
    conversations between users and agents.
    """

    def __init__(self, config_manager: Optional[Any] = None):
        """
        Initializes the client and connects to Cosmos DB.

        Args:
            config_manager: Optional configuration manager. If not provided,
                          will create a default one.

        Raises:
            ImportError: If azure-cosmos is not installed
            Exception: If connection to Cosmos DB fails
        """
        if CosmosClient is None:
            raise ImportError(
                "azure-cosmos is required for ContextStoreClient. "
                "Install with: pip install azure-cosmos"
            )

        self.logger = logging.getLogger(__name__)

        # Import ConfigurationManager at runtime to avoid circular imports
        if config_manager is None:
            try:
                from ..config import ConfigurationManager

                self.config_manager = ConfigurationManager()
            except ImportError:
                # Fallback: create a simple config manager
                self.config_manager = SimpleConfigManager()
        else:
            self.config_manager = config_manager

        try:
            # Get Cosmos DB configuration
            cosmos_config = self._get_cosmos_config()

            # Initialize Cosmos client
            self.client = CosmosClient(cosmos_config["endpoint"], cosmos_config["key"])

            # Get database and container references
            self.database = self.client.get_database_client(
                cosmos_config.get("database_name", "AgenticPlatform")
            )
            self.container = self.database.get_container_client(
                cosmos_config.get("container_name", "Context")
            )

            self.logger.info("ContextStoreClient initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing ContextStoreClient: {e}")
            raise

    def _get_cosmos_config(self) -> Dict[str, str]:
        """
        Retrieves Cosmos DB configuration from the settings.

        Returns:
            Dictionary containing Cosmos DB configuration

        Raises:
            ValueError: If required configuration is missing
        """
        config = self.config_manager.get_settings()

        # Try to get from various configuration sources
        cosmos_endpoint = None
        cosmos_key = None

        if isinstance(config, dict):
            # Dictionary-based configuration
            cosmos_endpoint = (
                config.get("cosmos_endpoint")
                or config.get("COSMOS_ENDPOINT")
                or config.get("cosmos", {}).get("endpoint")
            )
            cosmos_key = (
                config.get("cosmos_key")
                or config.get("COSMOS_KEY")
                or config.get("cosmos", {}).get("key")
            )
        else:
            # Pydantic settings object
            cosmos_endpoint = getattr(config, "cosmos_endpoint", None) or getattr(
                config, "COSMOS_ENDPOINT", None
            )
            cosmos_key = getattr(config, "cosmos_key", None) or getattr(config, "COSMOS_KEY", None)

        # Fall back to environment variables
        if not cosmos_endpoint:
            cosmos_endpoint = os.getenv("COSMOS_ENDPOINT") or os.getenv("AZURE_COSMOS_ENDPOINT")
        if not cosmos_key:
            cosmos_key = os.getenv("COSMOS_KEY") or os.getenv("AZURE_COSMOS_KEY")

        if not cosmos_endpoint or not cosmos_key:
            raise ValueError(
                "Cosmos DB configuration missing. Please set COSMOS_ENDPOINT "
                "and COSMOS_KEY in your configuration or environment variables."
            )

        # Get database and container names
        database_name = "AgenticPlatform"
        container_name = "Context"

        if isinstance(config, dict):
            database_name = (
                config.get("cosmos_database")
                or config.get("COSMOS_DATABASE")
                or config.get("cosmos", {}).get("database_name", database_name)
            )
            container_name = (
                config.get("cosmos_container")
                or config.get("COSMOS_CONTAINER")
                or config.get("cosmos", {}).get("container_name", container_name)
            )
        else:
            database_name = getattr(config, "cosmos_database", database_name)
            container_name = getattr(config, "cosmos_container", container_name)

        return {
            "endpoint": cosmos_endpoint,
            "key": cosmos_key,
            "database_name": database_name,
            "container_name": container_name,
        }

    def load_or_create(self, session_id: str, subject_id: str) -> Dict[str, Any]:
        """
        Loads a session context, creating a new one if it doesn't exist.

        Args:
            session_id: Unique identifier for the session
            subject_id: Identifier for the subject (user/entity) of the session

        Returns:
            Dictionary containing the session document

        Raises:
            Exception: If there's an error accessing Cosmos DB
        """
        try:
            # Attempt to read existing session
            # read_item is efficient for point reads
            session_doc = self.container.read_item(item=session_id, partition_key=subject_id)

            self.logger.info(f"Loaded existing session {session_id}")
            return session_doc

        except exceptions.CosmosResourceNotFoundError:
            self.logger.info(f"Session {session_id} not found. Creating new session.")

            # Create new session document
            new_session_doc = self._create_new_session_document(session_id, subject_id)

            # Save to Cosmos DB
            created_doc = self.container.create_item(body=new_session_doc)
            self.logger.info(f"Created new session {session_id}")

            return created_doc

        except Exception as e:
            self.logger.error(f"Error loading/creating session {session_id}: {e}")
            raise

    def update_and_save(
        self,
        context_doc: Dict[str, Any],
        user_input: str,
        agent_output: str,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Appends the latest turn to the thread and saves the document.

        Args:
            context_doc: The existing context document
            user_input: The user's input message
            agent_output: The agent's response
            agent_name: Optional name of the responding agent

        Returns:
            Updated context document

        Raises:
            Exception: If there's an error saving to Cosmos DB
        """
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Append user message
            user_message = {
                "id": str(uuid.uuid4()),
                "sender": "user",
                "timestamp": now,
                "content": user_input,
                "type": "text",
            }
            context_doc["thread"].append(user_message)

            # Append agent response
            agent_message = {
                "id": str(uuid.uuid4()),
                "sender": "agent",
                "timestamp": now,
                "content": agent_output,
                "type": "text",
            }

            # Add agent name if provided
            if agent_name:
                agent_message["agent_name"] = agent_name

            context_doc["thread"].append(agent_message)

            # Update timestamp
            context_doc["updated_at"] = now

            # Save to Cosmos DB using upsert
            updated_doc = self.container.upsert_item(body=context_doc)

            self.logger.info(f"Context for session {context_doc['id']} saved")
            return updated_doc

        except Exception as e:
            self.logger.error(f"Error updating session {context_doc.get('id')}: {e}")
            raise

    def get_conversation_history(
        self, session_id: str, subject_id: str, limit: Optional[int] = None
    ) -> list:
        """
        Retrieves conversation history for a session.

        Args:
            session_id: Unique identifier for the session
            subject_id: Identifier for the subject of the session
            limit: Optional limit on number of messages to return

        Returns:
            List of conversation messages
        """
        try:
            session_doc = self.container.read_item(item=session_id, partition_key=subject_id)

            thread = session_doc.get("thread", [])

            if limit:
                thread = thread[-limit:]

            return thread

        except exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Session {session_id} not found")
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving history for session {session_id}: {e}")
            raise

    def update_session_metadata(
        self, session_id: str, subject_id: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates metadata for a session.

        Args:
            session_id: Unique identifier for the session
            subject_id: Identifier for the subject of the session
            metadata: Metadata to update

        Returns:
            Updated session document
        """
        try:
            session_doc = self.container.read_item(item=session_id, partition_key=subject_id)

            # Update metadata
            session_doc["metadata"].update(metadata)
            session_doc["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Save changes
            updated_doc = self.container.upsert_item(body=session_doc)

            self.logger.info(f"Metadata updated for session {session_id}")
            return updated_doc

        except Exception as e:
            self.logger.error(f"Error updating metadata for session {session_id}: {e}")
            raise

    def _create_new_session_document(
        self, session_id: str, subject_id: str, subject_type: str = "user"
    ) -> Dict[str, Any]:
        """
        Creates a new, empty session document based on the platform schema.

        Args:
            session_id: Unique identifier for the session
            subject_id: Identifier for the subject of the session
            subject_type: Type of subject (default: "user")

        Returns:
            New session document dictionary
        """
        now = datetime.now(timezone.utc).isoformat()

        return {
            "id": session_id,
            "subject": {"type": subject_type, "id": subject_id},
            "session_type": "chat",
            "initiator": {"type": "user", "id": subject_id},
            "created_at": now,
            "updated_at": now,
            "status": "active",
            "metadata": {"schema_version": "1.5.0", "sdk_version": "1.0.0"},
            "thread": [],
            "context": {"memory": [], "state": {}},
            "security": {},
        }

    def close_session(self, session_id: str, subject_id: str) -> Dict[str, Any]:
        """
        Marks a session as closed.

        Args:
            session_id: Unique identifier for the session
            subject_id: Identifier for the subject of the session

        Returns:
            Updated session document
        """
        try:
            session_doc = self.container.read_item(item=session_id, partition_key=subject_id)

            # Update status and timestamp
            session_doc["status"] = "closed"
            session_doc["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Save changes
            updated_doc = self.container.upsert_item(body=session_doc)

            self.logger.info(f"Session {session_id} closed")
            return updated_doc

        except Exception as e:
            self.logger.error(f"Error closing session {session_id}: {e}")
            raise
