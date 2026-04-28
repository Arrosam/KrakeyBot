"""``recall`` plugin — passive auto-recall + active explicit recall.

Two components ship together because they are **two stages of one
discovery pipeline**, not two unrelated capabilities:

  1. The Reflect (passive, per-beat) vec-searches GM against incoming
     ``[STIMULUS]`` content and fills the ``[GRAPH MEMORY]`` prompt
     layer with the highest-weighted nodes that fit the token budget.
     KB *index nodes* (placed by Sleep when migrating mature GM
     content into a KnowledgeBase) surface here naturally.

  2. The Tool (active, on dispatch) lets Self drill down into a
     KB she has noticed in ``[GRAPH MEMORY]``. Without step 1 first
     surfacing a KB index node, Self has no way to learn that any
     particular KB exists — so without step 1 she would never call
     step 2. Disabling either half breaks the chain.

Sharing the package keeps the two halves enabled or disabled together
(consistent UX), gives the shared GM-query primitive a clean home
(``gm_query.py``), and makes the pipeline visible to anyone reading
the plugin folder. Same packaging pattern as ``in_mind_note``
(reflect-owns-state + tool-mutates-state).
"""
