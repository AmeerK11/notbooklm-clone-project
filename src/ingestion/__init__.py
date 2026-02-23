"""Ingestion package.

Exports the storage adapter for local development.
"""
from .storage import StorageAdapter, LocalStorageAdapter

__all__ = ["StorageAdapter", "LocalStorageAdapter"]
