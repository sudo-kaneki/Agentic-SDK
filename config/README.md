# Configuration Directory

This directory contains configuration files for different environments and deployment scenarios.

## 📁 Files Overview

| File | Purpose | Usage |
|------|---------|-------|
| `.env.example` | Environment variables template | Copy to `.env` and customize |
| `development.yaml` | Development environment settings | Local development |
| `production.yaml` | Production deployment settings | Live/staging environments |
| `testing.yaml` | Test environment settings | Running tests |

## 🚀 Quick Start

### 1. Environment Variables (Recommended)
```bash
# Copy the example file
cp config/.env.example .env

# Edit with your actual values
nano .env
```

### 2. YAML Configuration Files
```python
from energyai_sdk.config import config_manager

# Load specific environment configuration
config = config_manager.load_from_file("config/development.yaml")

# Or use environment variables (automatic)
config = config_manager.get_settings()
```

## ⚙️ Configuration Options

### 🤖 AI Model Configuration
```yaml
azure_openai:
  endpoint: "https://your-resource.openai.azure.com/"
  api_key: "your-api-key"
  deployment_name: "gpt-4o"
  api_version: "2024-02-01"
```

### 🌐 Application Settings
```yaml
application:
  title: "EnergyAI Platform"
  version: "1.0.0"
  debug: false
  host: "0.0.0.0"
  port: 8000
```

### 📊 Telemetry & Monitoring
```yaml
telemetry:
  enabled: true
  langfuse:
    public_key: "pk_lf_..."
    secret_key: "sk_lf_..."
  azure_monitor:
    connection_string: "InstrumentationKey=..."
```

### 🔒 Security Settings
```yaml
security:
  enable_cors: true
  api_key_required: true
  rate_limiting:
    enabled: true
    requests_per_minute: 100
```

## 🏗️ Environment-Specific Features

### Development (`development.yaml`)
- ✅ Debug mode enabled
- ✅ Verbose logging
- ✅ Auto-reload
- ✅ Test endpoints enabled
- ❌ Rate limiting disabled

### Production (`production.yaml`)
- ✅ Security hardened
- ✅ Monitoring enabled
- ✅ Rate limiting enabled
- ✅ Environment variable substitution
- ❌ Debug mode disabled

### Testing (`testing.yaml`)
- ✅ Mock APIs enabled
- ✅ Fast startup
- ✅ Deterministic settings
- ❌ External telemetry disabled
- ❌ Caching disabled

## 🔧 Usage Examples

### Basic Usage
```python
from energyai_sdk.config import get_settings, config_manager

# Using environment variables
settings = get_settings()

# Using specific config file
config = config_manager.load_from_file("config/production.yaml")
```

### With EnergyAI Platform
```python
from energyai_sdk.config import create_energy_platform

# Create platform with specific config
app = create_energy_platform("config/production.yaml")

# Or use environment variables
app = create_energy_platform()  # Uses .env or system env vars
```

### Environment Variable Override
```bash
# Set environment variables to override config
export AZURE_OPENAI_ENDPOINT="https://my-custom-endpoint.com/"
export ENERGYAI_LOG_LEVEL="DEBUG"

# Run your application
python your_app.py
```

## 🛡️ Security Best Practices

1. **Never commit API keys** - Use `.env` files (gitignored)
2. **Use environment variables** in production
3. **Rotate keys regularly**
4. **Enable rate limiting** in production
5. **Use HTTPS** for all endpoints
6. **Validate CORS origins**

## 📝 Configuration Schema

For detailed configuration options, see:
- `energyai_sdk/config.py` - Configuration models and validation
- Environment variable names follow the pattern: `ENERGYAI_<SETTING>`
- YAML files support environment variable substitution with `${VAR_NAME}`

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   # Check environment variables
   echo $AZURE_OPENAI_API_KEY
   ```

2. **Invalid YAML Syntax**
   ```bash
   # Validate YAML
   python -c "import yaml; yaml.safe_load(open('config/development.yaml'))"
   ```

3. **Permission Issues**
   ```bash
   # Check file permissions
   ls -la config/
   ```

### Debug Configuration Loading
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from energyai_sdk.config import config_manager
config = config_manager.get_settings()
```
