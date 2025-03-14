---
page_type: sample
languages:
  - python
products:
  - azure
  - azure-search
---

# Samples for Azure Cognitive Search client library for Python

These code samples show common scenario operations with the Azure Cognitive
Search client library.

Authenticate the client with a Azure Cognitive Search [API Key Credential](https://learn.microsoft.com/azure/search/search-security-api-keys):

[https://github.com/Azure/azure-sdk-for-python/blob/master/sdk/search/azure-search-documents/samples/sample_authentication.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_authentication.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_authentication_async.py))

Then for common search index operations:

* Get a document by key: [sample_get_document.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_get_document.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_get_document_async.py))

* Perform a simple text query: [sample_simple_query.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_simple_query.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_simple_query_async.py))

* Perform a filtered query: [sample_filter_query.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_filter_query.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_filter_query_async.py))

* Perform a faceted query: [sample_facet_query.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_facet_query.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_facet_query_async.py))

* Get auto-completions: [sample_autocomplete.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_autocomplete.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_autocomplete_async.py))

* Get search suggestions: [sample_suggestions.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_suggestions.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_suggestions_async.py))

* Perform basic document updates: [sample_crud_operations.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_crud_operations.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_crud_operations_async.py))

* CRUD operations for index: [sample_index_crud_operations.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_index_crud_operations.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_index_crud_operations_async.py))

* Analyze text: [sample_analyze_text.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_analyze_text.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_analyze_text_async.py))

* CRUD operations for indexers: [sample_indexers_operations.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_indexers_operations.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_indexers_operations_async.py))

* General workflow of indexer, datasource and index: [sample_indexer_datasource_skillset.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_indexer_datasource_skillset.py)

* Semantic search: [sample_semantic_search.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_semantic_search.py)

* Vector search: [sample_vector_search.py](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_vector_search.py) ([async version](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/async_samples/sample_vector_search_async.py))

## Prerequisites

* Python 3.8 or later is required to use this package
* You must have an [Azure subscription](https://azure.microsoft.com/free/)
* You must create the "Hotels" sample index [in the Azure Portal](https://learn.microsoft.com/azure/search/search-get-started-portal)

## Setup

1. Install the Azure Cognitive Search client library for Python with [pip](https://pypi.org/project/pip/):

   ```bash
   pip install azure-search-documents --pre
   ```

2. Clone or download [this repository](https://github.com/Azure/azure-sdk-for-python)
3. Open this sample folder in [Visual Studio Code](https://code.visualstudio.com) or your IDE of choice.

## Running the samples

1. Open a terminal window and `cd` to the directory that the samples are saved in.
2. Set the environment variables specified in the sample file you wish to run.
3. Follow the usage described in the file, e.g. `python sample_simple_query.py`

## Next steps

Check out the [API reference documentation](https://learn.microsoft.com/rest/api/searchservice/)
to learn more about what you can do with the Azure Cognitive Search client library.
