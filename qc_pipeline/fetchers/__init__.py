"""Data fetchers package for multi-platform data retrieval."""
from __future__ import annotations

from qc_pipeline.fetchers.base import DataFetcherProtocol, FetchResult, PLATFORMS
from qc_pipeline.fetchers.local import LocalFileFetcher, create_local_fetcher

# Lazy imports for platform-specific fetchers
def get_labellerr_fetcher():
    """Get the LabellerrFetcher class (requires labellerr SDK)."""
    from qc_pipeline.fetchers.labellerr import (
        LabellerrFetcher, 
        create_labellerr_fetcher, 
        fetch_labellerr_data,
        is_labellerr_available,
        STATUS_MAPPING,
        map_statuses_to_api,
    )
    return LabellerrFetcher, create_labellerr_fetcher, fetch_labellerr_data, is_labellerr_available, STATUS_MAPPING, map_statuses_to_api


__all__ = [
    "DataFetcherProtocol",
    "FetchResult",
    "PLATFORMS",
    "LocalFileFetcher",
    "create_local_fetcher",
    "get_labellerr_fetcher",
]
