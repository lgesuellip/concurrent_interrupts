# Concurrent Interrupts with LangGraph

A LangGraph application demonstrating concurrent interrupt handling in agent workflows.

## Prerequisites

- Python 3.11 or higher
- LangGraph CLI installed

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables:
Create a `.env` file with your required environment variables (API keys, etc.)

## Running the Agent

### Option 1: Run with SDK (without API)
```bash
uv run src/agent.py
```

### Option 2: Run as LangGraph API

To start the LangGraph API server, run:

```bash
langgraph up
```

This command will:
- Load the configuration from `langgraph.json`
- Start the API server with the agent defined in `src/agent.py`
- Make the API available for handling concurrent interrupts

The API will be accessible at `http://localhost:8000` by default.

## API Testing

A Postman collection is included (`LangGraph Multiple Interrupts.postman_collection.json`) for testing the API endpoints.

## Project Structure

- `src/agent.py` - Main agent implementation with the `build_agent` function
- `langgraph.json` - LangGraph configuration file
- `.env` - Environment variables (create this file with your settings)