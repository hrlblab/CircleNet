"""Resolve optional task/backbone imports only when selected."""
from collections.abc import Mapping
from importlib import import_module


class LazyRegistry(Mapping):
    def __init__(self, package, entries):
        self.package = package
        self.entries = entries
        self.resolved = {}

    def __getitem__(self, key):
        if key not in self.resolved:
            module, name = self.entries[key]
            self.resolved[key] = getattr(import_module(module, self.package), name)
        return self.resolved[key]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)
