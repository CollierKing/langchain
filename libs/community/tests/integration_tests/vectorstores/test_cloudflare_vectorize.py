"""Integration tests for the Cloudflare Vectorize module."""

import os
import uuid
from typing import Generator, List, Dict
import requests

import pytest

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.cloudflare_vectorize import CloudflareVectorize
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


class TestCloudflareVectorize(VectorStoreIntegrationTests):
    """Test Cloudflare Vectorize vectorstore."""

    vectorstore_cls = CloudflareVectorize

    @staticmethod
    def get_embeddings() -> Embeddings:
        """Return embedding model."""
        return FakeEmbeddings(size=128)

    @pytest.fixture(scope="function")
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        """Initialize vector store."""
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        d1_api_token = os.environ.get("CLOUDFLARE_D1_API_TOKEN")
        d1_database_id = os.environ.get("CLOUDFLARE_D1_DATABASE_ID")
        vectorize_api_token = os.environ.get("CLOUDFLARE_VECTORIZE_API_TOKEN")
        ai_api_token = os.environ.get("CLOUDFLARE_AI_API_TOKEN")
        
        if not account_id or not api_token:
            pytest.skip("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN needed for tests")
            
        index_name = os.environ.get("CLOUDFLARE_INDEX_NAME", f"langchain-test-{uuid.uuid4()}")
        
        # Check if we should create our own test index
        create_index = "CLOUDFLARE_INDEX_NAME" not in os.environ
        
        if create_index:
            try:
                # Try to delete the index first if it exists (clean slate)
                try:
                    CloudflareVectorize.delete_index(
                        account_id=account_id,
                        api_token=api_token,
                        index_name=index_name,
                    )
                    
                except Exception:
                    # Ignore errors if index doesn't exist
                    pass
                
                # Create a fresh index for testing
                CloudflareVectorize.create_index(
                    account_id=account_id,
                    api_token=api_token,
                    index_name=index_name,
                    dimensions=128,
                )
                
            except Exception as e:
                pytest.skip(f"Failed to create test index: {e}")
        
        store = CloudflareVectorize(
            embedding=self.get_embeddings(),
            account_id=account_id,
            api_token=api_token,
            index_name=index_name,
        )
        
        yield store
        
        # Clean up
        if create_index:
            try:
                CloudflareVectorize.delete_index(
                    account_id=account_id,
                    api_token=api_token,
                    index_name=index_name,
                )
                
            except Exception as e:
                print(f"Error cleaning up test index: {e}")

    @property
    def has_async(self) -> bool:
        """Return whether vectorstore supports async API."""
        return True
