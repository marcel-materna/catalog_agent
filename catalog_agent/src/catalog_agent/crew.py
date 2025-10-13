# crew_integration/src/crewai_mcp_integration/crew.py
from platform import python_branch
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from mcp import StdioServerParameters
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
import sys
import yaml
import logging
from dotenv import load_dotenv
from crewai.llm import LLM
from pathlib import Path
# English comment: Silence httpx / LiteLLM info logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)  # optional


# Load environment variables
load_dotenv()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# print python version
print(sys.version)

@CrewBase
class CatalogAgentCrew:
    """
    Crew für CatalogAgent mit MCP Stdio-Server.

    This CrewBase class uses CrewAI's MCP integration to connect to a DuckDB MCP
    server over stdio and expose its tools to the agents.
    """
    
    agents: List[BaseAgent]
    tasks: List[Task]


    mcp_server_params = StdioServerParameters(
        command="uvx",
        args=[
            "mcp-server-duckdb",
            "--db-path",
            "C:/Projekte/agent_duckdb/catalog_agent/data/metadata.duckdb",
            ]
    )
    

    @agent
    def natural_language_query_agent(self) -> Agent:
        """Agent for natural language queries using MCP tools."""
        return Agent(
            config=self.agents_config['natural_language_query_agent'],  # type: ignore[index]
            tools=self.get_mcp_tools(),
            verbose=True,            
        )

    @agent
    def data_presentation_agent(self) -> Agent:
        """Agent for presenting results in human language."""
        return Agent(
            config=self.agents_config['data_presentation_agent'],  # type: ignore[index]
            tools=[],
            verbose=True,
        )

    @task
    def natural_language_query_task(self) -> Task:
        """Task to execute the natural language to SQL flow."""
        return Task(
            config=self.tasks_config['natural_language_query_task'],  # type: ignore[index]
            agent=self.natural_language_query_agent(),
            verbose=True,
        )

    @task
    def data_presentation_task(self) -> Task:
        """Task to present the raw results in natural language."""
        return Task(
            config=self.tasks_config['data_presentation_task'],  # type: ignore[index]
            agent=self.data_presentation_agent(),
            verbose=True,
            context=[self.natural_language_query_task()],
            output_file='data/data_presentation_task.md',
        )

    @crew
    def crew(self) -> Crew:
        """Create the crew with sequential processing."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            full_output=True,
        )