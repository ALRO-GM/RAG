"""Orchestrator: composes the platform's pieces into products.

It reads a product config (``config/products/*.yaml``), follows the
``live.event.v1`` stream of a conversation and, when the trigger rules
say so, asks the knowledge and reasoner services (through ports) for a
suggestion. It never imports ``src`` or ``transcriber``: the services
are reached over HTTP with the contracts of ``contracts/``.
"""
