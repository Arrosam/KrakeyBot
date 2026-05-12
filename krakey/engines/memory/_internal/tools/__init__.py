"""Reusable memory primitives shared between GraphMemory + KnowledgeBase.

Each tool takes a connection + table name + parameters and returns
plain rows or scored rows. Tools own NO persistent state — callers
hold the connection. This lets GM and KB compose the same vec/FTS/
graph algorithms without inheritance or duplication.
"""
