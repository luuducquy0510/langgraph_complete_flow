"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Currently includes tools for web search
and other external integrations.
"""

from langchain_core.tools.base import BaseTool

from .web_search_tools import tavily_search_tool

tools: list[BaseTool] = [tavily_search_tool]