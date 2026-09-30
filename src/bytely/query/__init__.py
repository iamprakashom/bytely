"""Queries over the code graph: skeleton, callers, grep, and ask.

Every query reads a graph that `refresh_graph()` has just brought up to
date, so answers describe the code as it is now, including uncommitted
edits. Output is plain text, deterministic for a given tree.
"""
