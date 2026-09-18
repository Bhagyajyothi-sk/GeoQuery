"""
GeoQueryAI — centralised HTTP error helpers.

Keeps route handlers clean: raise a typed exception instead of manually
constructing HTTPException every time.
"""

from fastapi import HTTPException


class AIServiceError(HTTPException):
    """Raised when the AI (Gemini) query parser fails."""

    def __init__(self, detail: str = "AI service unavailable"):
        super().__init__(status_code=503, detail=detail)


class SearchServiceError(HTTPException):
    """Raised when the semantic search (RemoteCLIP + FAISS) layer fails."""

    def __init__(self, detail: str = "Semantic search service unavailable"):
        super().__init__(status_code=503, detail=detail)


class NoCandidatesError(HTTPException):
    """Raised when the semantic search returns zero candidates."""

    def __init__(self, detail: str = "No candidate tiles found for the given query"):
        super().__init__(status_code=404, detail=detail)


class GeoAnalysisError(HTTPException):
    """Raised when the geospatial / STAC pipeline fails."""

    def __init__(self, detail: str = "Geospatial analysis failed"):
        super().__init__(status_code=503, detail=detail)


class NoSatelliteDataError(HTTPException):
    """Raised when no suitable Sentinel-2 scene is found for the AOI."""

    def __init__(self, detail: str = "No satellite data found for the given area and date range"):
        super().__init__(status_code=404, detail=detail)
