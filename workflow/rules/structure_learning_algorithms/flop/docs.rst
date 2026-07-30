FLOP
====

This module pins vanilla ``flopsearch==0.3.0`` (MPL-2.0), standardizes every
input column, and calls ``flop(X, 2.0, restarts=50)`` in production.  It maps
FLOP's symmetric value-2 undirected endpoints to Benchpress symmetric
adjacency entries while preserving value-1 directed edges.
