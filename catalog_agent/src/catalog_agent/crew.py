# crew_integration/src/crewai_mcp_integration/crew.py
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from mcp import StdioServerParameters
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
import sys
import logging
from dotenv import load_dotenv
from crewai.llm import LLM
# English comment: Silence httpx / LiteLLM info logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)  # optional

from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
# Load environment variables
load_dotenv()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# print python version
print(sys.version)

tracer_provider = register(
    project_name="fedcatalog-agent",
    batch=True,
    endpoint="https://app.phoenix.arize.com/s/marcel-koch/v1/traces",
    auto_instrument=True
)

CrewAIInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

llm = LLM(
    model="ollama/mistral-small3.1:24b",
    api_base="https://ollama.ai.materna.net/api/generate",# Modellname aus .env Datei einlesen
    # model="openai/gpt-5",
    temperature=0.1,
    # max_tokens=150,
    # top_p=0.9,
    # frequency_penalty=0.1,
    # presence_penalty=0.1,
    # stop=["END"],
    # seed=42
)

@CrewBase
class CatalogAgentCrew:
    """
    Crew für CatalogAgent mit MCP Stdio-Server.

    This CrewBase class uses CrewAI's MCP integration to connect to a DuckDB MCP
    server over stdio and expose its tools to the agents.
    """
    
    agents: List[BaseAgent]
    tasks: List[Task]


    # mcp_server_params = StdioServerParameters(
    #     command="uvx",
    #     args=[
    #         "mcp-server-duckdb",
    #         "--db-path",
    #         "C:/Projekte/fedcatalog_agent/catalog_agent/data/metadata.duckdb",
    #         "--readonly",
    #         ]
    # )
    # mcp_server_params = StdioServerParameters(
    #     command="uvx",
    #     args=[
    #         "mcp-server-motherduck",
    #         "--db-path",
    #         "C:/Projekte/fedcatalog_agent/catalog_agent/data/metadata.duckdb",
    #         "--read-only",
    #         ]
    # )
    
    mcp_server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "mcp-server-sqlite",
            "--db-path",
            "C:/Projekte/fedcatalog_agent/catalog_agent/data/metadata.sqlite",
            ]
    )
    
    

    @agent
    def natural_language_query_agent(self) -> Agent:
        """Agent for natural language queries using MCP tools."""
        return Agent(
            config=self.agents_config['natural_language_query_agent'],  # type: ignore[index]
            tools=self.get_mcp_tools("list_tables", "describe_table", "read_query"),
            verbose=True,
            reasoning=False,
            allow_delegation=False,
            memory=False,
            max_iter=3,
            max_execution_time=300, 
            llm=llm
        )

    @agent
    def data_presentation_agent(self) -> Agent:
        """Agent for presenting results in human language."""
        return Agent(
            config=self.agents_config['data_presentation_agent'],  # type: ignore[index]
            tools=[],
            llm=llm,
            verbose=True,
            reasoning=False,
            allow_delegation=False,
            memory=False,
            max_iter=3,
            max_execution_time=300,
        )
    @task
    def natural_language_query_task(self) -> Task:
        """Task to execute the natural language to SQL flow."""
        return Task(
            config=self.tasks_config['natural_language_query_task'],  # type: ignore[index]
            agent=self.natural_language_query_agent(),
        )

    # @task
    # def execute_sql_task(self) -> Task:
    #     """Task to execute the SQL produced by the previous task using MCP query tool."""
    #     return Task(
    #         config=self.tasks_config['execute_sql_task'],  # type: ignore[index]
    #         agent=self.natural_language_query_agent(),
    #     )

    @task
    def data_presentation_task(self) -> Task:
        """Task to present the raw results in natural language."""
        return Task(
            config=self.tasks_config['data_presentation_task'],  # type: ignore[index]
            agent=self.data_presentation_agent(),
            output_file='data/data_presentation_task.md',
        )

    @crew
    def crew(self) -> Crew:
        """Create the crew with sequential processing."""
        return Crew(
            agents=self.agents,
            tasks=[
                self.natural_language_query_task(),
                # self.execute_sql_task(),
                self.data_presentation_task(),
            ],
            process=Process.sequential,
            verbose=True,
            full_output=True,
        )