"""
Debugging tools for the agent.

Provides introspection capabilities to query available MCP tools and
other debugging information via the Agent Gateway.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from sap_cloud_sdk.agentgateway.exceptions import (
    AgentGatewaySDKError,
    AgentGatewayServerError,
    MCPServerNotFoundError,
)
from tools.common import get_agent_gateway_client, get_current_jwt_token

logger = logging.getLogger(__name__)


def _format_mcp_error(e: Exception, context: str) -> str:
    """Return a structured, human-readable error string for MCP failures."""
    if isinstance(e, MCPServerNotFoundError):
        return (
            f"MCP server not found while {context}.\n"
            f"The MCP server or destination could not be resolved.\n"
            f"Detail: {e}"
        )
    if isinstance(e, AgentGatewayServerError):
        code_info = f" (error code: {e.error_code})" if e.error_code is not None else ""
        return (
            f"Agent Gateway server error while {context}{code_info}.\n"
            f"Detail: {e.server_message}"
        )
    if isinstance(e, AgentGatewaySDKError):
        message = str(e)
        if "401" in message:
            return (
                f"Permission denied while {context}.\n"
                f"Your user does not have sufficient permissions to access this resource.\n"
                f"Detail: {message}"
            )
        return (
            f"Agent Gateway SDK error while {context}.\n"
            f"Detail: {message}"
        )
    return (
        f"Unexpected error while {context}.\n"
        f"Type: {type(e).__name__}\n"
        f"Detail: {e}"
    )


# MCP Server IDs
S4_PRODUCT_MCP_SERVER_ID = "sap.s4:apiResource:API_PRODUCT_0002_MCP:v1"
S4_PRODUCT_TOOL_NAME = "list_ProductPlantCosting_for_sap_self"

ARIBA_SOURCING_EVENTS_MCP_SERVER_ID = "sap.aribas4:apiResource:sourcing_event_MCP:v1"
ARIBA_EVENTS_TOOL_NAME = "getEventsIds"


@tool
async def list_available_tools() -> str:
    """List all MCP tools currently available via the Agent Gateway."""
    try:
        client = get_agent_gateway_client()
        user_token = get_current_jwt_token()

        if user_token:
            logger.info("Calling list_mcp_tools with user token")
        else:
            logger.warning("No user token available for list_mcp_tools")

        tools = await client.list_mcp_tools(user_token=user_token)

        if not tools:
            return "No MCP tools are currently available."

        lines = ["**Available MCP Tools:**\n"]
        for t in tools:
            tool_name = getattr(t, "name", "unknown")
            server_name = getattr(t, "server_name", "unknown")
            tool_id = getattr(t, "id", "unknown")
            tool_description = getattr(t, "description", "")

            lines.append(f"- **{tool_name}** (id: `{tool_id}`, server: `{server_name}`)")
            if tool_description:
                lines.append(f"  {tool_description}")

        lines.append(f"\n*Total: {len(tools)} tool(s)*")
        return "\n".join(lines)

    except Exception as e:
        logger.exception("Failed to list MCP tools")
        return _format_mcp_error(e, "listing available MCP tools")


@tool
async def list_products_from_s4(top: int = 10) -> str:
    """Retrieve a list of products from the SAP S/4HANA system.

    Args:
        top: Maximum number of products to retrieve (default 10).
    """
    try:
        client = get_agent_gateway_client()
        user_token = get_current_jwt_token()

        if user_token:
            logger.info("Calling MCP tool %s with user token", S4_PRODUCT_TOOL_NAME)
        else:
            logger.warning("No user token available for MCP tool call")

        logger.info(
            "Calling MCP tool: server=%s, tool=%s, top=%s",
            S4_PRODUCT_MCP_SERVER_ID,
            S4_PRODUCT_TOOL_NAME,
            top,
        )

        result = await client.call_mcp_tool(
            tool=S4_PRODUCT_TOOL_NAME,
            ord_id=S4_PRODUCT_MCP_SERVER_ID,
            user_token=user_token,
        )

        import json

        if not result:
            return "No data returned from MCP tool."

        return f"**IMPORTANT: DO NOT PARSE THE RESPONSE, JUST RETURN RAW RESPONSE!!**\n**S4 Product Data ({S4_PRODUCT_TOOL_NAME}):**\n```json\n{json.dumps(result, indent=2, default=str)}\n```"

    except Exception as e:
        logger.exception("Failed to call MCP tool %s", S4_PRODUCT_TOOL_NAME)
        return _format_mcp_error(e, f"calling S4 product tool '{S4_PRODUCT_TOOL_NAME}'")


@tool
async def list_ariba_events() -> str:
    """Retrieve sourcing events from Ariba (events from the past year to the next year)."""
    import time

    try:
        client = get_agent_gateway_client()
        user_token = get_current_jwt_token()

        if user_token:
            logger.info("Calling MCP tool %s with user token", ARIBA_EVENTS_TOOL_NAME)
        else:
            logger.warning("No user token available for MCP tool call")

        now_ms = int(time.time() * 1000)
        one_year_ms = 365 * 24 * 60 * 60 * 1000
        create_date_from = now_ms - one_year_ms
        create_date_to = now_ms + one_year_ms

        filter_expr = f"(createDateFrom gt {create_date_from} and createDateTo lt {create_date_to})"

        logger.info(
            "Calling MCP tool: server=%s, tool=%s, filter=%s",
            ARIBA_SOURCING_EVENTS_MCP_SERVER_ID,
            ARIBA_EVENTS_TOOL_NAME,
            filter_expr,
        )

        result = await client.call_mcp_tool(
            tool=ARIBA_EVENTS_TOOL_NAME,
            ord_id=ARIBA_SOURCING_EVENTS_MCP_SERVER_ID,
            filter=filter_expr,
            user_token=user_token,
        )

        import json

        if not result:
            return "No events returned from Ariba."

        return f"**IMPORTANT: DO NOT PARSE THE RESPONSE, JUST RETURN RAW RESPONSE!!**\n**Ariba Sourcing Events ({ARIBA_EVENTS_TOOL_NAME}):**\n```json\n{json.dumps(result, indent=2, default=str)}\n```"

    except Exception as e:
        logger.exception("Failed to call MCP tool %s", ARIBA_EVENTS_TOOL_NAME)
        return _format_mcp_error(e, f"calling Ariba events tool '{ARIBA_EVENTS_TOOL_NAME}'")
