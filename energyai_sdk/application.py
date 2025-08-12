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

# Import from our SDK
from .core import (
    AgentRequest,
    CoreAgent,
    agent_registry,
    initialize_sdk,
    monitor,
    telemetry_manager,
)
from .middleware import MiddlewareContext, MiddlewarePipeline, create_default_pipeline

# API models


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""

    message: str = Field(..., description="The user message", min_length=1, max_length=10000)
    agent_id: Optional[str] = Field(None, description="Specific agent to use")
    session_id: Optional[str] = Field(
        None, description="Session identifier for conversation continuity"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
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
    ):
        self.title = title
        self.version = version
        self.description = description
        self.debug = debug
        self.start_time = datetime.now(timezone.utc)
        self.logger = logging.getLogger(__name__)

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
            "telemetry": (
                "healthy"
                if telemetry_manager.azure_tracer or telemetry_manager.langfuse_client
                else "not_configured"
            ),
        }

        self.is_ready = True
        self.logger.info("EnergyAI Application started successfully")

    async def _shutdown(self):
        """Application shutdown logic."""
        self.logger.info("Shutting down EnergyAI Application...")

        # Flush telemetry
        if telemetry_manager.langfuse_client:
            try:
                telemetry_manager.langfuse_client.flush()
            except Exception as e:
                self.logger.error(f"Error flushing Langfuse telemetry: {e}")

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
                telemetry_configured=bool(
                    telemetry_manager.azure_tracer or telemetry_manager.langfuse_client
                ),
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

    @monitor("application.process_chat_request")
    async def _process_chat_request(
        self, request: ChatRequest, background_tasks: BackgroundTasks
    ) -> ChatResponse:
        """Process a chat request through the middleware pipeline."""
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

        # Create agent request
        agent_request = AgentRequest(
            message=request.message,
            agent_id=agent_id,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={
                **request.metadata,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "streaming": request.stream,
            },
        )

        # Create middleware context
        context = MiddlewareContext(request=agent_request)

        try:
            # Execute middleware pipeline
            context = await self.middleware_pipeline.execute(context)

            # If middleware didn't process the request (no response), process with agent
            if not context.response and not context.error:
                context.response = await agent.process_request(agent_request)

            # Handle errors
            if context.error:
                raise HTTPException(status_code=500, detail=str(context.error))

            if not context.response:
                raise HTTPException(status_code=500, detail="No response generated")

            # Convert to API response
            execution_time = int((time.time() - start_time) * 1000)

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
            raise
        except Exception as e:
            self.logger.error(f"Error processing chat request: {e}")
            self.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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


def create_application(
    title: str = "EnergyAI Agentic SDK",
    version: str = "1.0.0",
    description: str = "AI Agent Platform for Energy Analytics",
    enable_default_middleware: bool = True,
    enable_cors: bool = True,
    debug: bool = False,
    **middleware_config,
) -> EnergyAIApplication:
    """Create an EnergyAI application with optional default configuration."""

    # Create middleware pipeline if requested
    middleware_pipeline = None
    if enable_default_middleware:
        middleware_pipeline = create_default_pipeline(**middleware_config)

    return EnergyAIApplication(
        title=title,
        version=version,
        description=description,
        middleware_pipeline=middleware_pipeline,
        enable_cors=enable_cors,
        debug=debug,
    )


def create_production_application(
    api_keys: list[str],
    azure_monitor_connection_string: Optional[str] = None,
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    max_requests_per_minute: int = 100,
    enable_caching: bool = True,
) -> EnergyAIApplication:
    """Create a production-ready application with security and monitoring."""

    # Initialize SDK with telemetry
    initialize_sdk(
        azure_monitor_connection_string=azure_monitor_connection_string,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        log_level="INFO",
    )

    # Import production middleware
    from .middleware import create_production_pipeline

    # Create production middleware pipeline
    middleware_pipeline = create_production_pipeline(
        api_keys=set(api_keys),
        max_requests_per_minute=max_requests_per_minute,
        enable_detailed_errors=False,  # Security: hide detailed errors in production
    )

    return EnergyAIApplication(
        title="EnergyAI Production Platform",
        version="1.0.0",
        description="Production AI Agent Platform for Energy Analytics",
        middleware_pipeline=middleware_pipeline,
        enable_cors=False,  # More restrictive CORS in production
        debug=False,
    )


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
