#!/usr/bin/env python
# coding=utf-8

"""
AnalysisAgent for MemEvolve trajectory analysis.

Implements the AnalysisAgent class expected by MemEvolve's PhaseAnalyzer.
Uses ToolCallingAgent with trajectory analysis tools to analyze
task execution logs and identify memory system improvement opportunities.
"""

from typing import Optional

from base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """
    Agent specialized for analyzing task execution trajectories
    and memory system performance.

    Provides tools for viewing trajectory summaries, inspecting
    specific steps, and examining memory database contents.
    """

    def __init__(self, model, task_logs_dir: str, max_steps: int = 30):
        from MemEvolve.utils.trajectory_tools import (
            TrajectoryViewerTool,
            StepViewerTool,
            MemoryDatabaseViewerTool,
        )
        from FlashOAgents import ToolCallingAgent

        super().__init__(model)
        self.task_logs_dir = task_logs_dir

        tools = [
            TrajectoryViewerTool(task_logs_dir),
            StepViewerTool(task_logs_dir),
            MemoryDatabaseViewerTool(),
        ]

        self.agent_fn = ToolCallingAgent(
            model=model,
            tools=tools,
            max_steps=max_steps,
        )
