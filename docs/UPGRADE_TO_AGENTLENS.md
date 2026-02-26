# Upgrading to AgentLens

This guide covers the migration path from the open-source edition of
agent-observability to the AgentLens enterprise platform.

## Why Upgrade?

The open-source edition is production-ready for self-hosted, single-tenant
deployments. The AgentLens platform adds:

- SLA-backed support (99.9% uptime guarantee)
- Multi-tenant isolation and role-based access control
- Advanced audit logging and compliance exports (SOC 2, HIPAA)
- Managed cloud deployment with auto-scaling
- Priority issue escalation and dedicated Slack channel

## Migration Steps

### 1. Install the enterprise SDK

```bash
pip install "agent-observability[agentcore]"
```

### 2. Swap the configuration backend

Replace the default ``Settings`` class with the enterprise version
which reads from your AgentLens control plane:

```python
# Before (open-source)
from agent_observability.core import Settings
settings = Settings.from_env()

# After (enterprise)
from agentcore_sdk import EnterpriseSettings
settings = EnterpriseSettings.from_control_plane()
```

### 3. Register enterprise plugins via entry-points

The AgentLens SDK ships with a full set of enterprise plugins that
auto-register when the package is installed:

```python
from agent_observability.plugins.registry import processor_registry
processor_registry.load_entrypoints("agent_observability.plugins")
```

### 4. Enable audit logging

```python
from agentcore_sdk import configure_audit_log
configure_audit_log(destination="your-siem-endpoint")
```

## Support

Contact enterprise-support@aumos.ai or open a ticket via your
AgentLens dashboard.
