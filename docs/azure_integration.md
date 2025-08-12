# Azure Integration Guide

This guide explains how to integrate EnergyAI SDK with Azure services for production deployment.

## 🏗️ Architecture Overview

The EnergyAI SDK now supports three external Azure services:

1. **Externalized Agentic Registry** (Cosmos DB)
   - Stores agent and tool definitions
   - Enables dynamic agent/tool loading at runtime
   - Supports versioning and metadata

2. **Externalized Context Store** (Cosmos DB)
   - Persists conversation sessions across requests
   - Enables stateful agent interactions
   - Supports TTL and cleanup

3. **Integrated Observability** (OpenTelemetry + Azure Monitor)
   - Distributed tracing across agent operations
   - Metrics collection and monitoring
   - Integration with Azure Monitor and OTLP endpoints

## 🛠️ Setup Instructions

### 1. Azure Cosmos DB Setup

#### Create Cosmos DB Account
```bash
# Create resource group
az group create --name energyai-rg --location eastus

# Create Cosmos DB account
az cosmosdb create \
  --name energyai-cosmos \
  --resource-group energyai-rg \
  --default-consistency-level Session \
  --locations regionName=eastus
```

#### Create Databases and Containers
```bash
# Create database
az cosmosdb sql database create \
  --account-name energyai-cosmos \
  --resource-group energyai-rg \
  --name energyai_platform

# Create agents container
az cosmosdb sql container create \
  --account-name energyai-cosmos \
  --database-name energyai_platform \
  --resource-group energyai-rg \
  --name agents \
  --partition-key-path "/id"

# Create tools container
az cosmosdb sql container create \
  --account-name energyai-cosmos \
  --database-name energyai_platform \
  --resource-group energyai-rg \
  --name tools \
  --partition-key-path "/id"

# Create sessions container with TTL
az cosmosdb sql container create \
  --account-name energyai-cosmos \
  --database-name energyai_platform \
  --resource-group energyai-rg \
  --name sessions \
  --partition-key-path "/session_id" \
  --ttl 3600
```

#### Get Connection Information
```bash
# Get endpoint
az cosmosdb show \
  --name energyai-cosmos \
  --resource-group energyai-rg \
  --query documentEndpoint

# Get primary key
az cosmosdb keys list \
  --name energyai-cosmos \
  --resource-group energyai-rg \
  --query primaryMasterKey
```

### 2. Azure Monitor Setup

#### Create Application Insights
```bash
# Create Application Insights
az monitor app-insights component create \
  --app energyai-insights \
  --location eastus \
  --resource-group energyai-rg \
  --application-type web

# Get instrumentation key
az monitor app-insights component show \
  --app energyai-insights \
  --resource-group energyai-rg \
  --query instrumentationKey
```

### 3. Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Cosmos DB Configuration
COSMOS_ENDPOINT="https://energyai-cosmos.documents.azure.com:443/"
COSMOS_KEY="your_primary_key_here"

# Azure Monitor Configuration
AZURE_MONITOR_CONNECTION_STRING="InstrumentationKey=your_instrumentation_key_here"

# Optional: OTLP Endpoint
OTLP_ENDPOINT="https://your-otlp-collector:4317"

# API Security
API_KEYS="key1,key2,key3"
```

## 💻 Usage Examples

### Basic Production Application

```python
from energyai_sdk.application import create_production_application

app = create_production_application(
    api_keys=["your_secure_api_key"],
    cosmos_endpoint="https://energyai-cosmos.documents.azure.com:443/",
    cosmos_key="your_cosmos_key",
    azure_monitor_connection_string="InstrumentationKey=your_key",
    max_requests_per_minute=100
)

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app.get_fastapi_app(), host="0.0.0.0", port=8000)
```

### Custom Client Configuration

```python
from energyai_sdk.clients import (
    RegistryClient,
    ContextStoreClient,
    MonitoringClient,
    MonitoringConfig
)
from energyai_sdk.application import create_application

# Initialize clients
registry_client = RegistryClient(
    cosmos_endpoint="https://energyai-cosmos.documents.azure.com:443/",
    cosmos_key="your_cosmos_key",
    database_name="energyai_platform"
)

context_store_client = ContextStoreClient(
    cosmos_endpoint="https://energyai-cosmos.documents.azure.com:443/",
    cosmos_key="your_cosmos_key",
    default_ttl=3600  # 1 hour
)

monitoring_config = MonitoringConfig(
    service_name="energyai-production",
    environment="production",
    azure_monitor_connection_string="InstrumentationKey=your_key"
)
monitoring_client = MonitoringClient(monitoring_config)

# Create application with external clients
app = create_application(
    title="EnergyAI Production Platform",
    registry_client=registry_client,
    context_store_client=context_store_client,
    monitoring_client=monitoring_client
)
```

### Dynamic Tool Loading

```python
from energyai_sdk.core import KernelFactory
from energyai_sdk.clients import RegistryClient

async def setup_dynamic_agents():
    # Create kernel
    kernel = KernelFactory.create_kernel()

    # Configure Azure OpenAI
    KernelFactory.configure_azure_openai_service(
        kernel,
        deployment_name="gpt-4",
        endpoint="https://your-endpoint.openai.azure.com/",
        api_key="your_openai_key"
    )

    # Load tools from registry
    registry_client = RegistryClient(cosmos_endpoint, cosmos_key)
    loaded_tools = await KernelFactory.load_tools_from_registry(kernel, registry_client)

    print(f"Loaded {loaded_tools} tools from registry")
    return kernel
```

## 📊 Data Schemas

### Agent Definition (Cosmos DB)
```json
{
  "id": "energy_analyst_v1",
  "name": "Energy Analyst",
  "description": "Expert energy analyst for renewable projects",
  "system_prompt": "You are an expert energy analyst...",
  "model_config": {
    "deployment_name": "gpt-4",
    "temperature": 0.3
  },
  "tools": ["lcoe_calculator", "capacity_factor"],
  "capabilities": ["financial_analysis", "technical_analysis"],
  "version": "1.0.0",
  "tags": ["energy", "analysis", "finance"],
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Tool Definition (Cosmos DB)
```json
{
  "id": "lcoe_calculator",
  "name": "LCOE Calculator",
  "description": "Calculate Levelized Cost of Energy",
  "category": "energy_finance",
  "schema": {
    "type": "function",
    "function": {
      "name": "calculate_lcoe",
      "parameters": {
        "type": "object",
        "properties": {
          "capex": {"type": "number"},
          "opex": {"type": "number"},
          "generation": {"type": "number"}
        },
        "required": ["capex", "opex", "generation"]
      }
    }
  },
  "endpoint_url": "https://api.energyai.com/tools/lcoe",
  "auth_config": {
    "api_key": "tool_api_key"
  },
  "version": "1.2.0",
  "tags": ["energy", "finance", "calculation"]
}
```

### Session Document (Cosmos DB)
```json
{
  "id": "session_123",
  "session_id": "session_123",
  "subject_id": "user_456",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:30:00Z",
  "context": {
    "messages": [
      {
        "user": "Calculate LCOE for 100MW solar",
        "assistant": "I'll help you calculate that...",
        "timestamp": "2024-01-01T00:00:00Z"
      }
    ],
    "user_preferences": {
      "units": "metric",
      "currency": "USD"
    }
  },
  "metadata": {
    "client_type": "web",
    "session_type": "consultation"
  },
  "ttl": 3600
}
```

## 🔍 Monitoring and Observability

### Key Metrics Tracked
- `chat_request_count` - Total chat requests
- `chat_request_duration` - Request processing time
- `agent_execution_count` - Agent execution count
- `agent_execution_duration` - Agent execution time
- `tool_execution_count` - Tool usage metrics
- `tool_execution_duration` - Tool execution time

### Trace Spans
- `application.process_chat_request` - Full request processing
- `agent.execute` - Individual agent execution
- `tool.execute` - Tool execution
- `registry.fetch` - Registry operations
- `context_store.update` - Context persistence

### Custom Dashboards

Create Azure Monitor dashboards to track:
1. Request throughput and latency
2. Agent performance metrics
3. Tool usage patterns
4. Error rates and types
5. Session persistence metrics

## 🚀 Deployment

### Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY energyai_sdk/ ./energyai_sdk/
COPY examples/ ./examples/

# Set environment variables
ENV PYTHONPATH=/app

# Run application
CMD ["python", "examples/production_server.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: energyai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: energyai-platform
  template:
    metadata:
      labels:
        app: energyai-platform
    spec:
      containers:
      - name: energyai
        image: energyai/platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: COSMOS_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: azure-secrets
              key: cosmos-endpoint
        - name: COSMOS_KEY
          valueFrom:
            secretKeyRef:
              name: azure-secrets
              key: cosmos-key
        - name: AZURE_MONITOR_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-secrets
              key: monitor-connection-string
```

## 🔒 Security Best Practices

1. **Use Managed Identity** for Azure service authentication when possible
2. **Rotate keys regularly** - Cosmos DB and API keys
3. **Network isolation** - Use VNets and private endpoints
4. **API rate limiting** - Configure appropriate limits
5. **Input validation** - Validate all external inputs
6. **Audit logging** - Enable comprehensive audit trails
7. **Secret management** - Use Azure Key Vault for secrets

## 🧪 Testing

Run the integration example:
```bash
python examples/azure_integration_example.py
```

Run comprehensive tests:
```bash
pytest tests/test_azure_integration.py -v
```

## 📚 Additional Resources

- [Azure Cosmos DB Documentation](https://docs.microsoft.com/en-us/azure/cosmos-db/)
- [Azure Monitor Documentation](https://docs.microsoft.com/en-us/azure/azure-monitor/)
- [OpenTelemetry Python Documentation](https://opentelemetry-python.readthedocs.io/)
- [EnergyAI SDK Examples](../examples/)

## 🆘 Troubleshooting

### Common Issues

1. **Cosmos DB Connection Errors**
   - Verify endpoint URL and key
   - Check network connectivity
   - Ensure database and containers exist

2. **OpenTelemetry Setup Issues**
   - Verify instrumentation key
   - Check endpoint URLs
   - Validate configuration format

3. **Tool Loading Failures**
   - Check tool endpoint URLs
   - Verify authentication configuration
   - Validate schema format

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed client logging
logging.getLogger("energyai_sdk.clients").setLevel(logging.DEBUG)
```
