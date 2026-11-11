"""Resolve optional task/backbone imports only when selected."""
from collections.abc import MutableMapping
from importlib import import_module


class LazyRegistry(MutableMapping):
    def __init__(self, package, entries):
        self.package = package
        self.entries = dict(entries)
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

    def __contains__(self, key):
        # Membership tests must not import an optional component.
        return key in self.entries

    def __setitem__(self, key, value):
        # Preserve the original dictionaries' custom-component registration API.
        self.entries[key] = None
        self.resolved[key] = value

    def __delitem__(self, key):
        del self.entries[key]
        self.resolved.pop(key, None)
