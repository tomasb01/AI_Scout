"""Base scanner interface for AI Scout plugins."""

from abc import ABC, abstractmethod

from aiscout.models import ScannerConfig, ScanResult


class BaseScanner(ABC):
    """Abstract base class for all AI Scout scanners."""

    @abstractmethod
    def get_config(self) -> ScannerConfig:
        """Return scanner configuration (name, required credentials)."""
        ...

    @abstractmethod
    def scan(self, **kwargs) -> ScanResult:
        """Run discovery and return scan results."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable scanner name."""
        ...
