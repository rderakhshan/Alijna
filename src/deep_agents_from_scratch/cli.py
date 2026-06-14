"""Command-line interface runner for deep_agents_from_scratch."""

import asyncio
import os
import sys
from datetime import datetime

from deepagents import create_deep_agent
from rich.console import Console
from rich.panel import Panel

from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from deep_agents_from_scratch.research_tools import (
    get_today_str,
    tavily_search,
    think_tool,
)
from deep_agents_from_scratch.utils import stream_agent

console = Console()

# Memory adapter — instantiated only if MEMORY_PROVIDER is set
_memory_adapter = None


def get_model():
    """Dynamically construct the language model based on available environment variables."""
    # Check for DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key and not deepseek_key.startswith("your_"):
        from langchain_openai import ChatOpenAI

        console.print(
            "[bold green]Using DeepSeek Model (via OpenAI-compatible API)[/bold green]"
        )
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.0,
            api_key=deepseek_key,
            base_url="https://api.deepseek.com/v1",
        )

    # Check for OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("your_"):
        from langchain_openai import ChatOpenAI

        console.print("[bold green]Using OpenAI Model (gpt-4o)[/bold green]")
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            api_key=openai_key,
        )

    # Check for Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and not anthropic_key.startswith("your_"):
        from langchain_anthropic import ChatAnthropic

        console.print(
            "[bold green]Using Anthropic Model (claude-3-5-sonnet-latest)[/bold green]"
        )
        return ChatAnthropic(
            model="claude-3-5-sonnet-latest",
            temperature=0.0,
            api_key=anthropic_key,
        )

    # Fallback/Default behavior
    console.print(
        "[yellow]No custom API key found or keys have placeholder values. Falling back to default OpenAI client (gpt-4o-mini)...[/yellow]"
    )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )


def main():
    console.print(
        Panel.fit(
            "[bold cyan]🧱 Deep Agent CLI Runner[/bold cyan]\n"
            "Welcome to the standalone command-line interface for the Deep Agent.",
            border_style="cyan",
        )
    )

    # Check dependencies/keys
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key or tavily_key.startswith("your_"):
        console.print(
            "[bold red]Warning: TAVILY_API_KEY is not set. Web search tools will fail.[/bold red]\n"
            "Please update your `.env` file with a valid Tavily API Key."
        )

    # Initialize model
    try:
        model = get_model()
    except Exception as e:
        console.print(f"[bold red]Error initializing model: {e}[/bold red]")
        sys.exit(1)

    # Initialize memory adapter if requested
    global _memory_adapter
    memory_provider = os.getenv("MEMORY_PROVIDER")
    if memory_provider:
        from deep_agents_from_scratch.adapter.memory_adapter import MemoryAdapter

        try:
            memory_storage = os.getenv("MEMORY_STORAGE_DIR")
            _memory_adapter = MemoryAdapter(
                provider_name=memory_provider,
                model=model,
                storage_dir=memory_storage,
            )
            if _memory_adapter.initialize():
                console.print(
                    f"[bold green]Memory adapter ({memory_provider}) initialized[/bold green]"
                )
            else:
                console.print(
                    f"[bold yellow]Memory adapter ({memory_provider}) init returned False[/bold yellow]"
                )
        except Exception as e:
            console.print(f"[bold red]Failed to initialize memory: {e}[/bold red]")
            _memory_adapter = None

    # Limits
    max_concurrent_research_units = 3
    max_researcher_iterations = 3

    # Build prompt
    SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
        date=datetime.now().strftime("%a %b %#d, %Y"),
    )

    INSTRUCTIONS = (
        "# TODO MANAGEMENT\n"
        + TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + SUBAGENT_INSTRUCTIONS
    )

    # Create research sub-agent configuration
    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
        "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
        "tools": [tavily_search, think_tool],
    }

    # Tools available to the agent
    sub_agent_tools = [tavily_search, think_tool]

    # Initialize full deep agent using the deepagents package
    console.print("Initializing Deep Agent graph...")
    try:
        agent = create_deep_agent(
            tools=sub_agent_tools,
            system_prompt=INSTRUCTIONS,
            subagents=[research_sub_agent],
            model=model,
        )
        console.print("[bold green]Deep Agent initialization: SUCCESS[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize deep agent: {e}[/bold red]")
        sys.exit(1)

    # Start the CLI chat loop
    console.print(
        "\n[bold white]Type your query below to interact with the agent (or type 'exit'/'quit' to quit):[/bold white]"
    )
    while True:
        try:
            user_input = input("\n🧑 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[bold cyan]Goodbye![/bold cyan]")
                break

            console.print("\n🤖 Processing query...")

            # Inject relevant memory context if available
            messages: list = [{"role": "user", "content": user_input}]
            if _memory_adapter is not None:
                memory_context = _memory_adapter.provide_context(user_input)
                if memory_context:
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": (
                                "The following are relevant memories from past interactions "
                                "that may help you respond to the current query:\n\n"
                                + memory_context
                            ),
                        },
                    )

            # Stream agent execution
            final_state = asyncio.run(stream_agent(agent, {"messages": messages}))

            # Absorb trajectory into memory
            if _memory_adapter is not None and final_state is not None:
                _absorb_trajectory(_memory_adapter, user_input, final_state)

        except KeyboardInterrupt:
            console.print("\n[bold cyan]Exiting CLI... Goodbye![/bold cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")


def _absorb_trajectory(
    adapter,
    user_input: str,
    state: dict,
) -> None:
    """Extract messages from final state and feed them to the memory adapter."""
    messages = state.get("messages", []) or state.get("agent_messages", [])
    if not messages:
        return
    result_text = ""
    if hasattr(messages[-1], "content"):
        result_text = messages[-1].content
    adapter.absorb_trajectory(
        query=user_input,
        messages=messages,
        result=str(result_text) if result_text else None,
    )


if __name__ == "__main__":
    main()
