# FastAPI application framework

import asyncio
import logging
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field, validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .clients import ContextStoreClient

# Monitoring client - unified observability
try:
    from .clients.monitoring import MonitoringClient, get_monitoring_client

    MONITORING_AVAILABLE = True
except ImportError:
    MonitoringClient = None
    get_monitoring_client = None
    MONITORING_AVAILABLE = False

# Import from our SDK
from .core import AgentRequest, CoreAgent, agent_registry, initialize_sdk, monitor
from .middleware import MiddlewareContext, MiddlewarePipeline, create_default_pipeline

# API models


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""

    message: str = Field(..., description="The user message", min_length=1, max_length=10000)
    agent_id: Optional[str] = Field(None, description="Specific agent to use")
    session_id: Optional[str] = Field(
        None, description="Session identifier for conversation continuity"
    )
    subject_id: Optional[str] = Field(None, description="Subject/user identifier for context store")
    user_id: Optional[str] = Field(None, description="User identifier (deprecated, use subject_id)")
    stream: bool = Field(False, description="Enable streaming response")
    temperature: Optional[float] = Field(
        None, description="Model temperature override", ge=0.0, le=2.0
    )
    max_tokens: Optional[int] = Field(None, description="Maximum tokens override", ge=1, le=4000)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator("temperature")
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    content: str = Field(..., description="The agent's response")
    agent_id: str = Field(..., description="ID of the agent that processed the request")
    session_id: Optional[str] = Field(None, description="Session identifier")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    model_used: Optional[str] = Field(None, description="Model used for generation")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = Field(None, description="Error message if any")


class AgentInfo(BaseModel):
    """Model for agent information."""

    agent_id: str
    name: str
    description: str
    type: str
    capabilities: list[str]
    models: list[str]
    tools: list[str]
    skills: list[str]
    is_available: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthCheck(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    agents_count: int
    uptime_seconds: float
    telemetry_configured: bool
    components: dict[str, str]


class StreamingChatResponse(BaseModel):
    """Model for streaming chat response chunks."""

    chunk: str
    is_final: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Application core


class EnergyAIApplication:
    """
    Core application class that manages agents, middleware, and HTTP endpoints.
    """

    def __init__(
        self,
        title: str = "EnergyAI Agentic SDK",
        version: str = "1.0.0",
        description: str = "AI Agent Platform for Energy Analytics",
        middleware_pipeline: Optional[MiddlewarePipeline] = None,
        enable_cors: bool = True,
        enable_gzip: bool = True,
        debug: bool = False,
        context_store_client: Optional[ContextStoreClient] = None,
        langfuse_monitoring_client: Optional[MonitoringClient] = None,
    ):
        self.title = title
        self.version = version
        self.description = description
        self.debug = debug
        self.start_time = datetime.now(timezone.utc)
        self.logger = logging.getLogger(__name__)

        # Initialize external service clients
        self.context_store_client = context_store_client
        # Legacy langfuse_client support - now use unified monitoring client
        self.langfuse_monitoring_client = langfuse_monitoring_client

        # Initialize middleware pipeline
        self.middleware_pipeline = middleware_pipeline or create_default_pipeline()

        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app(enable_cors, enable_gzip)
        else:
            self.app = None
            self.logger.warning("FastAPI not available. Web endpoints will not be created.")

        # Application state
        self.is_ready = False
        self.components_status = {}

    def _create_fastapi_app(self, enable_cors: bool, enable_gzip: bool) -> FastAPI:
        """Create and configure FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()

        app = FastAPI(
            title=self.title, version=self.version, description=self.description, lifespan=lifespan
        )

        # Add middleware
        if enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        if enable_gzip:
            app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Add routes
        self._add_routes(app)

        return app

    async def _startup(self):
        """Application startup logic."""
        self.logger.info("Starting EnergyAI Application...")

        # Check component status
        self.components_status = {
            "agent_registry": "healthy" if agent_registry.agents else "no_agents",
            "middleware_pipeline": "healthy" if self.middleware_pipeline else "not_configured",
            "observability": (
                "healthy" if get_monitoring_client() is not None else "not_configured"
            ),
            "context_store": ("healthy" if self.context_store_client else "not_configured"),
            "langfuse_monitoring": self._get_langfuse_status(),
        }

        self.is_ready = True
        self.logger.info("EnergyAI Application started successfully")

    def _get_langfuse_status(self) -> str:
        """Get Langfuse monitoring status."""
        try:
            monitoring_client = get_monitoring_client()
            if monitoring_client and hasattr(monitoring_client, "langfuse_client"):
                if (
                    monitoring_client.langfuse_client
                    and monitoring_client.langfuse_client.is_enabled()
                ):
                    return "healthy"
            return "not_configured"
        except Exception:
            return "error"

    async def _shutdown(self):
        """Application shutdown logic."""
        self.logger.info("Shutting down EnergyAI Application...")

        # Cleanup external clients
        if self.context_store_client:
            try:
                # ContextStoreClient doesn't have async close method
                self.logger.info("Context store client cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up context store client: {e}")

        # Flush observability data
        monitoring_client = get_monitoring_client()
        if monitoring_client:
            try:
                monitoring_client.flush()
            except Exception as e:
                self.logger.error(f"Error flushing observability data: {e}")

        self.is_ready = False
        self.logger.info("EnergyAI Application shut down")

    def _add_routes(self, app: FastAPI):
        """Add all API routes to FastAPI app."""

        # Health check endpoint
        @app.get("/health", response_model=HealthCheck)
        async def health_check():
            """Get application health status."""
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            return HealthCheck(
                status="healthy" if self.is_ready else "starting",
                timestamp=datetime.now(timezone.utc),
                version=self.version,
                agents_count=len(agent_registry.agents),
                uptime_seconds=uptime,
                observability_configured=bool(get_monitoring_client() is not None),
                components=self.components_status,
            )

        # List agents endpoint
        @app.get("/agents", response_model=list[AgentInfo])
        async def list_agents():
            """List all available agents."""
            agents_info = []

            for agent_id, agent in agent_registry.agents.items():
                try:
                    capabilities = agent.get_capabilities()
                    agents_info.append(
                        AgentInfo(
                            agent_id=agent_id,
                            name=capabilities.get("agent_name", agent_id),
                            description=capabilities.get("description", "No description available"),
                            type=capabilities.get("type", "unknown"),
                            capabilities=capabilities.get("capabilities", []),
                            models=capabilities.get("models", []),
                            tools=capabilities.get("tools", []),
                            skills=capabilities.get("skills", []),
                            is_available=True,
                            metadata=capabilities.get("metadata", {}),
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error getting capabilities for agent {agent_id}: {e}")
                    agents_info.append(
                        AgentInfo(
                            agent_id=agent_id,
                            name=agent_id,
                            description="Error retrieving agent information",
                            type="unknown",
                            capabilities=[],
                            models=[],
                            tools=[],
                            skills=[],
                            is_available=False,
                            metadata={"error": str(e)},
                        )
                    )

            return agents_info

        # Get specific agent info
        @app.get("/agents/{agent_id}", response_model=AgentInfo)
        async def get_agent_info(agent_id: str):
            """Get information about a specific agent."""
            agent = agent_registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            try:
                capabilities = agent.get_capabilities()
                return AgentInfo(
                    agent_id=agent_id,
                    name=capabilities.get("agent_name", agent_id),
                    description=capabilities.get("description", "No description available"),
                    type=capabilities.get("type", "unknown"),
                    capabilities=capabilities.get("capabilities", []),
                    models=capabilities.get("models", []),
                    tools=capabilities.get("tools", []),
                    skills=capabilities.get("skills", []),
                    is_available=True,
                    metadata=capabilities.get("metadata", {}),
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error retrieving agent information: {e}"
                )

        # Chat endpoint for any agent
        @app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
            """Chat with an agent."""
            return await self._process_chat_request(request, background_tasks)

        # Agent-specific chat endpoint
        @app.post("/agents/{agent_id}/chat", response_model=ChatResponse)
        async def agent_chat(
            agent_id: str, request: ChatRequest, background_tasks: BackgroundTasks
        ):
            """Chat with a specific agent."""
            request.agent_id = agent_id
            return await self._process_chat_request(request, background_tasks)

        # Streaming chat endpoint
        @app.post("/chat/stream")
        async def stream_chat(request: ChatRequest):
            """Stream chat response from an agent."""
            if not request.stream:
                request.stream = True

            return StreamingResponse(self._stream_chat_response(request), media_type="text/plain")

        # Agent-specific streaming chat
        @app.post("/agents/{agent_id}/chat/stream")
        async def agent_stream_chat(agent_id: str, request: ChatRequest):
            """Stream chat response from a specific agent."""
            request.agent_id = agent_id
            request.stream = True

            return StreamingResponse(self._stream_chat_response(request), media_type="text/plain")

        # Reset agent context
        @app.post("/agents/{agent_id}/reset")
        async def reset_agent_context(agent_id: str):
            """Reset agent conversation context."""
            agent = agent_registry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            try:
                agent.reset_context()
                return {"status": "success", "message": f"Context reset for agent {agent_id}"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error resetting context: {e}")

        # Registry capabilities endpoint
        @app.get("/registry/capabilities")
        async def get_registry_capabilities():
            """Get overall registry capabilities."""
            return agent_registry.get_capabilities()

        # OpenAPI schema for tools (mentioned in the architecture doc)
        @app.get("/tools/openapi")
        async def get_tools_openapi():
            """Get OpenAPI schema for available tools."""
            return self._generate_tools_openapi()

        # Add route to load tools/agents from external registry
        @app.post("/registry/reload")
        async def reload_from_registry():
            """Reload tools and agents from external registry."""
            try:
                # Load tools from registry
                tools = await self.registry_client.list_tools()
                loaded_tools = len(tools)

                # Load agents from registry
                agents = await self.registry_client.list_agents()
                loaded_agents = len(agents)

                return {
                    "status": "success",
                    "tools_available": loaded_tools,
                    "agents_available": loaded_agents,
                    "timestamp": datetime.now(timezone.utc),
                }

            except Exception as e:
                self.logger.error(f"Error checking registry: {e}")
                raise HTTPException(status_code=500, detail=f"Registry check failed: {e}")

        # Session management endpoints
        @app.post("/sessions/{session_id}")
        async def create_session(session_id: str, subject_id: str = "anonymous"):
            """Create or load a session."""
            if not self.context_store_client:
                raise HTTPException(status_code=503, detail="Context store not available")

            try:
                session_doc = self.context_store_client.load_or_create(session_id, subject_id)
                return {
                    "status": "ready",
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "message_count": len(session_doc.get("thread", [])),
                    "created_at": session_doc.get("created_at"),
                    "updated_at": session_doc.get("updated_at"),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Session creation failed: {e}")

        @app.get("/sessions/{session_id}/history")
        async def get_session_history(
            session_id: str, subject_id: str = "anonymous", limit: int = None
        ):
            """Get session conversation history."""
            if not self.context_store_client:
                raise HTTPException(status_code=503, detail="Context store not available")

            try:
                history = self.context_store_client.get_conversation_history(
                    session_id, subject_id, limit
                )
                return {"session_id": session_id, "history": history, "message_count": len(history)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {e}")

        @app.post("/sessions/{session_id}/close")
        async def close_session(session_id: str, subject_id: str = "anonymous"):
            """Close a session."""
            if not self.context_store_client:
                raise HTTPException(status_code=503, detail="Context store not available")

            try:
                self.context_store_client.close_session(session_id, subject_id)
                return {"status": "closed", "session_id": session_id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to close session: {e}")

    @monitor("application.process_chat_request")
    async def _process_chat_request(
        self, request: ChatRequest, background_tasks: BackgroundTasks
    ) -> ChatResponse:
        """Process a chat request through the middleware pipeline with Langfuse tracing."""
        start_time = time.time()

        # Determine agent to use
        agent_id = request.agent_id
        if not agent_id:
            # Use the first available agent or a default
            available_agents = agent_registry.list_agents()
            if not available_agents:
                raise HTTPException(status_code=503, detail="No agents available")
            agent_id = available_agents[0]

        # Get the agent
        agent = agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Create Langfuse trace for the entire request
        trace = None
        generation = None
        context_span = None

        monitoring_client = get_monitoring_client()
        if (
            monitoring_client
            and hasattr(monitoring_client, "langfuse_client")
            and monitoring_client.langfuse_client
            and monitoring_client.langfuse_client.is_enabled()
        ):
            trace = monitoring_client.create_trace(
                name=f"agent-run:{agent_id}",
                user_id=request.subject_id or request.user_id or "anonymous",
                session_id=request.session_id,
                input_data={
                    "message": request.message,
                    "agent_id": agent_id,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                },
                metadata={
                    "agent_name": agent_id,
                    "request_type": "chat",
                    "streaming": request.stream,
                    "environment": "production",
                },
                tags=["agent-interaction", "chat", agent_id],
            )

        # Handle session persistence if session_id provided
        subject_id = request.subject_id or request.user_id or "anonymous"
        session_doc = None
        conversation_history = ""

        # Create span for context loading
        if trace and monitoring_client:
            context_span = monitoring_client.create_span(
                trace,
                name="context-loading",
                input_data={
                    "session_id": request.session_id,
                    "subject_id": subject_id,
                },
                metadata={"operation": "session_context_retrieval"},
            )

        if request.session_id and self.context_store_client:
            try:
                # Load or create session context
                session_doc = self.context_store_client.load_or_create(
                    request.session_id, subject_id
                )

                # Build conversation history for context
                thread = session_doc.get("thread", [])
                if thread:
                    history_lines = []
                    for msg in thread[-10:]:  # Last 10 messages for context
                        sender = msg.get("sender", "unknown")
                        content = msg.get("content", "")
                        agent_name = msg.get("agent_name", "Assistant")

                        if sender == "user":
                            history_lines.append(f"User: {content}")
                        elif sender == "agent":
                            history_lines.append(f"{agent_name}: {content}")

                    conversation_history = "\n".join(history_lines)
                    self.logger.info(
                        f"Loaded session {request.session_id} with {len(thread)} messages"
                    )
                else:
                    self.logger.info(f"Created new session {request.session_id}")

                # End context span with success
                if context_span and monitoring_client:
                    monitoring_client.end_span(
                        context_span,
                        output={
                            "session_found": bool(thread),
                            "message_count": len(thread),
                            "history_length": len(conversation_history),
                        },
                    )

            except Exception as e:
                self.logger.warning(f"Error handling session {request.session_id}: {e}")
                session_doc = None

                # End context span with error
                if context_span and monitoring_client:
                    monitoring_client.end_span(context_span, level="ERROR", status_message=str(e))

        # Create agent request with conversation history
        message_with_context = request.message
        if conversation_history:
            message_with_context = f"Previous conversation:\n{conversation_history}\n\nCurrent message: {request.message}"

        agent_request = AgentRequest(
            message=message_with_context,
            agent_id=agent_id,
            session_id=request.session_id,
            user_id=subject_id,
            metadata={
                **request.metadata,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "streaming": request.stream,
                "has_conversation_history": bool(conversation_history),
                "original_message": request.message,  # Keep original for context saving
                "session_context": session_doc.get("context", {}) if session_doc else {},
            },
        )

        # Create middleware context
        context = MiddlewareContext(request=agent_request)

        try:
            # Create Langfuse generation for the main LLM call
            if trace and monitoring_client:
                agent_capabilities = getattr(agent, "get_capabilities", lambda: {})()
                model_name = (
                    agent_capabilities.get("models", ["unknown"])[0]
                    if agent_capabilities.get("models")
                    else "unknown"
                )

                generation = monitoring_client.create_generation(
                    trace,
                    name="agent-invocation",
                    input_data={
                        "query": request.message,
                        "history": conversation_history,
                        "agent_context": message_with_context,
                    },
                    model=model_name,
                    model_parameters={
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens,
                    },
                    metadata={
                        "agent_name": agent_id,
                        "has_conversation_history": bool(conversation_history),
                        "message_count_in_context": (
                            len(conversation_history.split("\n")) if conversation_history else 0
                        ),
                    },
                )

            # Execute middleware pipeline
            context = await self.middleware_pipeline.execute(context)

            # If middleware didn't process the request (no response), process with agent
            if not context.response and not context.error:
                context.response = await agent.process_request(agent_request)

            # Handle errors
            if context.error:
                # End generation with error
                if generation and monitoring_client:
                    monitoring_client.end_generation(
                        generation, level="ERROR", status_message=str(context.error)
                    )
                raise HTTPException(status_code=500, detail=str(context.error))

            if not context.response:
                # End generation with error
                if generation and monitoring_client:
                    monitoring_client.end_generation(
                        generation, level="ERROR", status_message="No response generated"
                    )
                raise HTTPException(status_code=500, detail="No response generated")

            # End generation with success
            if generation and monitoring_client:
                # Try to extract token usage if available
                response_metadata = getattr(context.response, "metadata", {})
                usage = response_metadata.get("usage", {})

                monitoring_client.end_generation(
                    generation,
                    output=context.response.content,
                    usage=usage if usage else None,
                    level="DEFAULT",
                )

            # Update session context if session_id provided
            if (
                request.session_id
                and context.response
                and session_doc
                and self.context_store_client
            ):
                try:
                    # Get the original message from metadata
                    original_message = agent_request.metadata.get(
                        "original_message", request.message
                    )

                    # Get agent name for better context
                    agent_name = getattr(agent, "name", None) or agent_id

                    # Save the conversation turn
                    self.context_store_client.update_and_save(
                        session_doc,
                        user_input=original_message,
                        agent_output=context.response.content,
                        agent_name=agent_name,
                    )

                    self.logger.info(
                        f"Updated session {request.session_id} with new conversation turn"
                    )

                except Exception as e:
                    self.logger.warning(f"Error updating session context: {e}")

            # Convert to API response
            execution_time = int((time.time() - start_time) * 1000)

            # Record metrics (monitoring client would be used here if available)

            return ChatResponse(
                content=context.response.content,
                agent_id=context.response.agent_id,
                session_id=context.response.session_id,
                execution_time_ms=execution_time,
                model_used=context.response.metadata.get("model_used"),
                metadata={
                    **context.response.metadata,
                    **context.metadata,
                    "pipeline_phases": len(context.execution_phases),
                },
                error=context.response.error,
            )

        except HTTPException:
            # Update trace with HTTP error
            if trace and monitoring_client:
                monitoring_client.update_trace(
                    trace, level="ERROR", status_message="HTTP Exception occurred"
                )
            raise
        except Exception as e:
            self.logger.error(f"Error processing chat request: {e}")
            self.logger.error(traceback.format_exc())

            # End generation and trace with error
            if generation and monitoring_client:
                monitoring_client.end_generation(generation, level="ERROR", status_message=str(e))
            if trace and monitoring_client:
                monitoring_client.update_trace(trace, level="ERROR", status_message=str(e))

            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        finally:
            # Always update trace with final status and flush Langfuse data
            if trace and monitoring_client:
                execution_time = int((time.time() - start_time) * 1000)
                monitoring_client.update_trace(
                    trace,
                    metadata={
                        "execution_time_ms": execution_time,
                        "session_updated": bool(request.session_id and session_doc),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                # Flush all telemetry data to Langfuse
                if monitoring_client:
                    monitoring_client.flush()

    async def _stream_chat_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream chat response chunks."""
        # Note: This is a simplified streaming implementation
        # Real streaming would require agent-level streaming support

        try:
            # Process request normally first
            chat_request = ChatRequest(**request.dict())
            chat_request.stream = False  # Process normally, then chunk the response

            response = await self._process_chat_request(chat_request, None)

            # Simulate streaming by chunking the response
            content = response.content
            chunk_size = 50  # Characters per chunk

            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]

                chunk_response = StreamingChatResponse(
                    chunk=chunk,
                    is_final=(i + chunk_size >= len(content)),
                    metadata={"chunk_index": i // chunk_size},
                )

                yield f"data: {chunk_response.json()}\n\n"

                # Add delay to simulate real streaming
                await asyncio.sleep(0.1)

            # Send final chunk
            final_chunk = StreamingChatResponse(
                chunk="", is_final=True, metadata={"total_chunks": (len(content) // chunk_size) + 1}
            )

            yield f"data: {final_chunk.json()}\n\n"

        except Exception as e:
            error_chunk = StreamingChatResponse(
                chunk=f"Error: {str(e)}", is_final=True, metadata={"error": True}
            )
            yield f"data: {error_chunk.json()}\n\n"

    def _generate_tools_openapi(self) -> dict[str, Any]:
        """Generate OpenAPI schema for registered tools."""
        openapi_schema = {
            "openapi": "3.0.0",
            "info": {
                "title": "EnergyAI Tools API",
                "version": self.version,
                "description": "Available tools for EnergyAI agents",
            },
            "paths": {},
            "components": {"schemas": {}},
        }

        for tool_name, tool_def in agent_registry.tools.items():
            # Convert tool definition to OpenAPI path
            path = f"/tools/{tool_name}"

            # Build parameters schema
            parameters = []
            for param_name, param_info in tool_def.parameters.items():
                parameters.append(
                    {
                        "name": param_name,
                        "in": "query",
                        "required": param_info.get("required", False),
                        "schema": {
                            "type": param_info.get("type", "string"),
                            "default": param_info.get("default"),
                        },
                    }
                )

            openapi_schema["paths"][path] = {
                "post": {
                    "summary": tool_def.description,
                    "parameters": parameters,
                    "responses": {
                        "200": {
                            "description": "Tool execution result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "result": {"type": "string"},
                                            "metadata": {"type": "object"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            }

        return openapi_schema

    def add_agent(self, agent: CoreAgent):
        """Add an agent to the application."""
        agent_registry.register_agent(agent.agent_name, agent)
        self.logger.info(f"Agent {agent.agent_name} added to application")

    def set_middleware_pipeline(self, pipeline: MiddlewarePipeline):
        """Set the middleware pipeline."""
        self.middleware_pipeline = pipeline
        self.logger.info("Middleware pipeline updated")

    def get_fastapi_app(self) -> Optional[FastAPI]:
        """Get the FastAPI application instance."""
        return self.app


# ==============================================================================
# APPLICATION FACTORY AND CONVENIENCE FUNCTIONS
# ==============================================================================


# This function was already defined above - removing duplicate


# This function was already redefined above to avoid duplication


# ==============================================================================
# DEVELOPMENT SERVER
# ==============================================================================


class DevelopmentServer:
    """Development server for testing and local development."""

    def __init__(
        self,
        application: EnergyAIApplication,
        host: str = "0.0.0.0",
        port: int = 8000,  # nosec B104
    ):
        self.application = application
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)

    def run(self, reload: bool = True):
        """Run the development server."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI is required to run the development server")

        try:
            import uvicorn

            self.logger.info(f"Starting development server on {self.host}:{self.port}")

            uvicorn.run(
                "energyai_sdk.application:app",
                host=self.host,
                port=self.port,
                reload=reload,
                log_level="info" if not self.application.debug else "debug",
            )

        except ImportError:
            self.logger.error("uvicorn is required to run the development server")
            raise RuntimeError("Please install uvicorn: pip install uvicorn")
        except Exception as e:
            self.logger.error(f"Error starting development server: {e}")
            raise


def create_application(
    title: str = "EnergyAI Agentic SDK",
    version: str = "1.0.0",
    description: str = "AI Agent Platform for Energy Analytics",
    debug: bool = False,
    enable_context_store: bool = True,
    enable_observability: bool = True,
    enable_langfuse_monitoring: bool = False,  # For backward compatibility
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: str = "https://cloud.langfuse.com",
    langfuse_environment: str = "production",
    azure_monitor_connection_string: Optional[str] = None,
    enable_cors: bool = True,
    enable_gzip: bool = True,
) -> EnergyAIApplication:
    """Create an EnergyAI application with optional context store and observability."""

    # Initialize context store
    context_store_client = None
    if enable_context_store:
        try:
            context_store_client = ContextStoreClient()
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not initialize ContextStoreClient: {e}")

    # Initialize monitoring (if not already initialized)
    if enable_observability:
        from .clients.monitoring import (
            MonitoringConfig,
            get_monitoring_client,
            initialize_monitoring,
        )

        # Check if monitoring is already initialized
        if get_monitoring_client() is None:
            try:
                # Create monitoring config
                monitoring_config = MonitoringConfig(
                    environment=langfuse_environment,
                    enable_langfuse=enable_langfuse_monitoring
                    and bool(langfuse_public_key and langfuse_secret_key),
                    langfuse_public_key=langfuse_public_key,
                    langfuse_secret_key=langfuse_secret_key,
                    langfuse_host=langfuse_host,
                    enable_opentelemetry=bool(azure_monitor_connection_string),
                    azure_monitor_connection_string=azure_monitor_connection_string,
                )

                # Initialize monitoring
                initialize_monitoring(monitoring_config)
                logging.getLogger(__name__).info("Monitoring initialized")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not initialize monitoring: {e}")

    # Legacy monitoring support is now handled by the unified MonitoringClient

    return EnergyAIApplication(
        title=title,
        version=version,
        description=description,
        debug=debug,
        enable_cors=enable_cors,
        enable_gzip=enable_gzip,
        context_store_client=context_store_client,
        langfuse_monitoring_client=None,  # Now handled by unified monitoring client
    )


def create_production_application(
    title: str = "EnergyAI Production Platform",
    enable_context_store: bool = True,
    enable_observability: bool = True,
    enable_langfuse_monitoring: bool = True,
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: str = "https://cloud.langfuse.com",
    azure_monitor_connection_string: Optional[str] = None,
    **kwargs,
) -> EnergyAIApplication:
    """Create a production-ready EnergyAI application with monitoring."""
    return create_application(
        title=title,
        debug=False,
        enable_context_store=enable_context_store,
        enable_observability=enable_observability,
        enable_langfuse_monitoring=enable_langfuse_monitoring,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        langfuse_environment="production",
        azure_monitor_connection_string=azure_monitor_connection_string,
        **kwargs,
    )


def run_development_server(
    agents: Optional[list[CoreAgent]] = None,
    host: str = "0.0.0.0",  # nosec B104
    port: int = 8000,
    reload: bool = True,
    enable_telemetry: bool = False,
):
    """Quick start development server with optional agents."""

    # Initialize SDK
    initialize_sdk(log_level="DEBUG" if reload else "INFO")

    # Create application
    app = create_application(debug=reload)

    # Add agents if provided
    if agents:
        for agent in agents:
            app.add_agent(agent)

    # Create and run server
    server = DevelopmentServer(app, host, port)
    server.run(reload=reload)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

# Global application instance for uvicorn
app_instance = None


def get_application() -> EnergyAIApplication:
    """Get or create the global application instance."""
    global app_instance
    if app_instance is None:
        app_instance = create_application()
    return app_instance


# FastAPI app for uvicorn
app = None
if FASTAPI_AVAILABLE:
    app = get_application().get_fastapi_app()

if __name__ == "__main__":
    # Example usage with decorator-based agents
    from .agents import bootstrap_agents
    from .decorators import agent, tool

    # Define a sample agent using decorators
    @tool(name="sample_calculation", description="Sample calculation tool")
    def sample_calculation(value: float) -> dict:
        return {"result": value * 2, "operation": "double"}

    @agent(
        name="SampleEnergyAgent",
        description="Sample energy analysis agent for testing",
        system_prompt="You are a helpful energy analyst assistant.",
        tools=["sample_calculation"],
    )
    class SampleEnergyAgent:
        temperature = 0.7
        max_tokens = 1000

    try:
        # Configure Azure OpenAI
        azure_config = {
            "deployment_name": "gpt-4o",
            "endpoint": "https://your-endpoint.openai.azure.com/",
            "api_key": "your-api-key",
            "service_type": "azure_openai",
        }

        # Bootstrap agents using decorators
        agents = bootstrap_agents(azure_openai_config=azure_config)

        if "SampleEnergyAgent" in agents:
            sample_agent = agents["SampleEnergyAgent"]
            print("Sample agent created successfully using decorators")

            # Run development server
            run_development_server(agents=[sample_agent], host="127.0.0.1", port=8000, reload=True)
        else:
            print("Sample agent not created. Check configuration.")

    except Exception as e:
        print(f"Error setting up development environment: {e}")
        print("Run with proper Azure OpenAI credentials to test full functionality")
