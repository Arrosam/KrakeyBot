"""``reranker`` Engine — score-based reordering for recall + KB dedup.

Default impl ``DefaultRerankerEngine`` (in ``default.py``) embeds the
no-LLM fallback so the slot always has a working impl — there is no
``reranker = None`` tri-state any more. When the user has bound a
reranker tag the Engine forwards to that client; when unbound or the
upstream call fails the Engine returns preserve-order scores
(decreasing floats) so callers' stable sort leaves the input order
intact.
"""
from krakey.engines.reranker.default import DefaultRerankerEngine

__all__ = ["DefaultRerankerEngine"]
