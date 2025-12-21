class DomainError(Exception):
    """Base class for domain-specific errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class InvalidBoundingBoxError(DomainError):
    """Raised when bounding box coordinates violate invariants."""
    pass
