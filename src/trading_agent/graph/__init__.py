"""LangGraph orchestration: AgentState, graph builder, and the node implementations.

Cycle shape: fetch -> features -> predict -> analyze -> risk gate -> execute -> notify.
"""
