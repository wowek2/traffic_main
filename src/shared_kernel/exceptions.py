class DomainError(Exception):
    """Base class for all domain-specific errors in system."""
    pass

class InfrastructureError(DomainError):
    """Raised when infrastructure layer fails."""
    pass