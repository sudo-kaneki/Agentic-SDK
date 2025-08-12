"""
Context Store Client for session persistence in Cosmos DB.

This client handles loading and saving session documents based on the
Context Store JSON Schema, enabling stateful conversations across requests.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from azure.cosmos import exceptions as cosmos_exceptions
    from azure.cosmos.aio import CosmosClient

    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False

from ..exceptions import SDKError


@dataclass
class SessionDocument:
    """Session document structure for context store."""

    session_id: str
    subject_id: str
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    ttl: Optional[int] = None  # Time to live in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Cosmos DB storage."""
        return {
            "id": self.session_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context,
            "metadata": self.metadata or {},
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionDocument":
        """Create from Cosmos DB document."""
        return cls(
            session_id=data["session_id"],
            subject_id=data["subject_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            context=data.get("context", {}),
            metadata=data.get("metadata"),
            ttl=data.get("ttl"),
        )


class ContextStoreClient:
    """
    Client for the Context Store in Cosmos DB.

    Handles session persistence, enabling stateful conversations by storing
    and retrieving conversation context across agent interactions.
    """

    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_key: str,
        database_name: str = "energyai_platform",
        sessions_container: str = "sessions",
        default_ttl: int = 3600,  # 1 hour default TTL
    ):
        """
        Initialize the Context Store Client.

        Args:
            cosmos_endpoint: Cosmos DB endpoint URL
            cosmos_key: Cosmos DB primary key
            database_name: Database name (default: energyai_platform)
            sessions_container: Sessions container name (default: sessions)
            default_ttl: Default time to live in seconds (default: 3600)
        """
        if not COSMOS_AVAILABLE:
            raise SDKError(
                "Azure Cosmos DB SDK not available. Install with: pip install azure-cosmos"
            )

        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.database_name = database_name
        self.sessions_container = sessions_container
        self.default_ttl = default_ttl

        self.logger = logging.getLogger(__name__)
        self._client: Optional[CosmosClient] = None
        self._database = None
        self._sessions_container = None

    async def _get_client(self) -> CosmosClient:
        """Get or create Cosmos DB client."""
        if self._client is None:
            self._client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
            self._database = self._client.get_database_client(self.database_name)
            self._sessions_container = self._database.get_container_client(self.sessions_container)

        return self._client

    async def create_session(
        self,
        session_id: str,
        subject_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> SessionDocument:
        """
        Create a new session document.

        Args:
            session_id: Unique session identifier
            subject_id: User/subject identifier
            initial_context: Initial context data
            metadata: Additional metadata
            ttl: Time to live in seconds (uses default if not provided)

        Returns:
            Created SessionDocument
        """
        try:
            await self._get_client()

            now = datetime.now(timezone.utc)
            session_doc = SessionDocument(
                session_id=session_id,
                subject_id=subject_id,
                created_at=now,
                updated_at=now,
                context=initial_context or {},
                metadata=metadata,
                ttl=ttl or self.default_ttl,
            )

            # Create document in Cosmos DB
            await self._sessions_container.create_item(body=session_doc.to_dict())

            self.logger.info(f"Created session: {session_id} for subject: {subject_id}")
            return session_doc

        except cosmos_exceptions.CosmosResourceExistsError:
            self.logger.warning(f"Session already exists: {session_id}")
            raise SDKError(f"Session {session_id} already exists")
        except Exception as e:
            self.logger.error(f"Error creating session {session_id}: {e}")
            raise SDKError(f"Failed to create session: {e}") from e

    async def get_session(self, session_id: str) -> Optional[SessionDocument]:
        """
        Retrieve a session document by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionDocument if found, None otherwise
        """
        try:
            await self._get_client()

            response = await self._sessions_container.read_item(
                item=session_id, partition_key=session_id
            )

            session_doc = SessionDocument.from_dict(response)
            self.logger.info(f"Retrieved session: {session_id}")
            return session_doc

        except cosmos_exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Session not found: {session_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving session {session_id}: {e}")
            raise SDKError(f"Failed to retrieve session: {e}") from e

    async def update_session(
        self, session_id: str, context: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> SessionDocument:
        """
        Update session context and metadata.

        Args:
            session_id: Session identifier
            context: Updated context data
            metadata: Updated metadata (optional)

        Returns:
            Updated SessionDocument
        """
        try:
            await self._get_client()

            # Get existing session
            existing_session = await self.get_session(session_id)
            if not existing_session:
                raise SDKError(f"Session not found: {session_id}")

            # Update fields
            existing_session.context = context
            existing_session.updated_at = datetime.now(timezone.utc)
            if metadata is not None:
                existing_session.metadata = metadata

            # Save to Cosmos DB
            await self._sessions_container.upsert_item(body=existing_session.to_dict())

            self.logger.info(f"Updated session: {session_id}")
            return existing_session

        except Exception as e:
            self.logger.error(f"Error updating session {session_id}: {e}")
            raise SDKError(f"Failed to update session: {e}") from e

    async def append_to_context(self, session_id: str, key: str, value: Any) -> SessionDocument:
        """
        Append data to session context.

        Args:
            session_id: Session identifier
            key: Context key to update
            value: Value to append or set

        Returns:
            Updated SessionDocument
        """
        try:
            session_doc = await self.get_session(session_id)
            if not session_doc:
                raise SDKError(f"Session not found: {session_id}")

            # Handle different append scenarios
            if key in session_doc.context:
                existing_value = session_doc.context[key]
                if isinstance(existing_value, list):
                    existing_value.append(value)
                elif isinstance(existing_value, dict) and isinstance(value, dict):
                    existing_value.update(value)
                else:
                    session_doc.context[key] = value
            else:
                session_doc.context[key] = value

            return await self.update_session(session_id, session_doc.context, session_doc.metadata)

        except Exception as e:
            self.logger.error(f"Error appending to context {session_id}: {e}")
            raise SDKError(f"Failed to append to context: {e}") from e

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session document.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            await self._get_client()

            await self._sessions_container.delete_item(item=session_id, partition_key=session_id)

            self.logger.info(f"Deleted session: {session_id}")
            return True

        except cosmos_exceptions.CosmosResourceNotFoundError:
            self.logger.warning(f"Session not found for deletion: {session_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting session {session_id}: {e}")
            raise SDKError(f"Failed to delete session: {e}") from e

    async def list_sessions_by_subject(
        self, subject_id: str, limit: int = 100
    ) -> List[SessionDocument]:
        """
        List sessions for a specific subject.

        Args:
            subject_id: Subject identifier
            limit: Maximum number of sessions to return

        Returns:
            List of SessionDocument objects
        """
        try:
            await self._get_client()

            query = "SELECT * FROM c WHERE c.subject_id = @subject_id ORDER BY c.updated_at DESC"
            parameters = [{"name": "@subject_id", "value": subject_id}]

            sessions = []
            async for item in self._sessions_container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            ):
                sessions.append(SessionDocument.from_dict(item))
                if len(sessions) >= limit:
                    break

            self.logger.info(f"Retrieved {len(sessions)} sessions for subject: {subject_id}")
            return sessions

        except Exception as e:
            self.logger.error(f"Error listing sessions for subject {subject_id}: {e}")
            raise SDKError(f"Failed to list sessions: {e}") from e

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (if TTL is not handled automatically).

        Returns:
            Number of sessions cleaned up
        """
        try:
            await self._get_client()

            # Note: If TTL is properly configured in Cosmos DB, this may not be needed
            # as Cosmos will automatically delete expired documents

            now = datetime.now(timezone.utc)
            query = """
            SELECT c.id, c.session_id, c.created_at, c.ttl
            FROM c
            WHERE c.ttl IS NOT NULL
            """

            expired_sessions = []
            async for item in self._sessions_container.query_items(
                query=query, enable_cross_partition_query=True
            ):
                created_at = datetime.fromisoformat(item["created_at"])
                ttl_seconds = item.get("ttl", self.default_ttl)

                if (now - created_at).total_seconds() > ttl_seconds:
                    expired_sessions.append(item["session_id"])

            # Delete expired sessions
            deleted_count = 0
            for session_id in expired_sessions:
                if await self.delete_session(session_id):
                    deleted_count += 1

            self.logger.info(f"Cleaned up {deleted_count} expired sessions")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise SDKError(f"Failed to cleanup sessions: {e}") from e

    async def health_check(self) -> bool:
        """
        Check if the context store service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self._get_client()

            # Simple query to test connectivity
            query = "SELECT VALUE COUNT(1) FROM c"
            result = [
                item
                async for item in self._sessions_container.query_items(
                    query=query, enable_cross_partition_query=True
                )
            ]

            self.logger.info("Context store health check passed")
            return True

        except Exception as e:
            self.logger.error(f"Context store health check failed: {e}")
            return False

    async def close(self):
        """Close the Cosmos DB client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._database = None
            self._sessions_container = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Mock client for development/testing
class MockContextStoreClient(ContextStoreClient):
    """Mock context store client for development and testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._sessions: Dict[str, SessionDocument] = {}
        self.default_ttl = 3600

    async def create_session(
        self,
        session_id: str,
        subject_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> SessionDocument:
        """Mock session creation."""
        if session_id in self._sessions:
            raise SDKError(f"Session {session_id} already exists")

        now = datetime.now(timezone.utc)
        session_doc = SessionDocument(
            session_id=session_id,
            subject_id=subject_id,
            created_at=now,
            updated_at=now,
            context=initial_context or {},
            metadata=metadata,
            ttl=ttl or self.default_ttl,
        )

        self._sessions[session_id] = session_doc
        return session_doc

    async def get_session(self, session_id: str) -> Optional[SessionDocument]:
        """Mock session retrieval."""
        return self._sessions.get(session_id)

    async def update_session(
        self, session_id: str, context: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> SessionDocument:
        """Mock session update."""
        if session_id not in self._sessions:
            raise SDKError(f"Session not found: {session_id}")

        session_doc = self._sessions[session_id]
        session_doc.context = context
        session_doc.updated_at = datetime.now(timezone.utc)
        if metadata is not None:
            session_doc.metadata = metadata

        return session_doc

    async def append_to_context(self, session_id: str, key: str, value: Any) -> SessionDocument:
        """Mock context append."""
        session_doc = await self.get_session(session_id)
        if not session_doc:
            raise SDKError(f"Session not found: {session_id}")

        if key in session_doc.context:
            existing_value = session_doc.context[key]
            if isinstance(existing_value, list):
                existing_value.append(value)
            elif isinstance(existing_value, dict) and isinstance(value, dict):
                existing_value.update(value)
            else:
                session_doc.context[key] = value
        else:
            session_doc.context[key] = value

        return await self.update_session(session_id, session_doc.context, session_doc.metadata)

    async def delete_session(self, session_id: str) -> bool:
        """Mock session deletion."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def list_sessions_by_subject(
        self, subject_id: str, limit: int = 100
    ) -> List[SessionDocument]:
        """Mock session listing."""
        sessions = [
            session for session in self._sessions.values() if session.subject_id == subject_id
        ]
        return sorted(sessions, key=lambda x: x.updated_at, reverse=True)[:limit]

    async def cleanup_expired_sessions(self) -> int:
        """Mock cleanup."""
        now = datetime.now(timezone.utc)
        expired_sessions = []

        for session_id, session in self._sessions.items():
            ttl_seconds = session.ttl or self.default_ttl
            if (now - session.created_at).total_seconds() > ttl_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self._sessions[session_id]

        return len(expired_sessions)

    async def health_check(self) -> bool:
        """Mock health check."""
        return True

    async def close(self):
        """Mock close."""
        pass
