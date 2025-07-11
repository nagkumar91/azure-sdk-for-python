# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import Any, Callable, cast, List, Optional, Tuple, Union
from urllib.parse import unquote

from azure.core.exceptions import HttpResponseError
from azure.core.paging import ItemPaged, PageIterator

from ._deserialize import (
    get_blob_properties_from_generated_code,
    load_many_xml_nodes,
    load_xml_int,
    load_xml_string,
    parse_tags
)
from ._generated.models import BlobItemInternal, BlobPrefix as GenBlobPrefix, FilterBlobItem
from ._generated._utils.serialization import Deserializer
from ._models import BlobProperties, FilteredBlob
from ._shared.models import DictMixin
from ._shared.response_handlers import (
    process_storage_error,
    return_context_and_deserialized,
    return_raw_deserialized
)


class IgnoreListBlobsDeserializer(Deserializer):
    def __call__(self, target_obj, response_data, content_type=None):  # pylint: disable=inconsistent-return-statements
        if target_obj == "ListBlobsFlatSegmentResponse":
            return None
        super().__call__(target_obj, response_data, content_type)


class BlobPropertiesPaged(PageIterator):
    """An Iterable of Blob properties."""

    service_endpoint: Optional[str]
    """The service URL."""
    prefix: Optional[str]
    """A blob name prefix being used to filter the list."""
    marker: Optional[str]
    """The continuation token of the current page of results."""
    results_per_page: Optional[int]
    """The maximum number of results retrieved per API call."""
    continuation_token: Optional[str]
    """The continuation token to retrieve the next page of results."""
    location_mode: Optional[str]
    """The location mode being used to list results. The available
    options include "primary" and "secondary"."""
    current_page: Optional[List[BlobProperties]]
    """The current page of listed results."""
    container: Optional[str]
    """The container that the blobs are listed from."""
    delimiter: Optional[str]
    """A delimiting character used for hierarchy listing."""
    command: Callable
    """Function to retrieve the next page of items."""

    def __init__(
        self, command: Callable,
        container: str,
        prefix: Optional[str] = None,
        results_per_page: Optional[int] = None,
        continuation_token: Optional[str] = None,
        delimiter: Optional[str] = None,
        location_mode: Optional[str] = None,
    ) -> None:
        super(BlobPropertiesPaged, self).__init__(
            get_next=self._get_next_cb,
            extract_data=self._extract_data_cb,
            continuation_token=continuation_token or ""
        )
        self._command = command
        self.service_endpoint = None
        self.prefix = prefix
        self.marker = None
        self.results_per_page = results_per_page
        self.container = container
        self.delimiter = delimiter
        self.current_page = None
        self.location_mode = location_mode

    def _get_next_cb(self, continuation_token):
        try:
            return self._command(
                prefix=self.prefix,
                marker=continuation_token or None,
                maxresults=self.results_per_page,
                cls=return_context_and_deserialized,
                use_location=self.location_mode)
        except HttpResponseError as error:
            process_storage_error(error)

    def _extract_data_cb(self, get_next_return):
        self.location_mode, self._response = cast(Tuple[Optional[str], Any], get_next_return)
        self.service_endpoint = self._response.service_endpoint
        self.prefix = self._response.prefix
        self.marker = self._response.marker
        self.results_per_page = self._response.max_results
        self.container = self._response.container_name
        self.current_page = [self._build_item(item) for item in self._response.segment.blob_items]

        return self._response.next_marker or None, self.current_page

    def _build_item(self, item: Union[BlobItemInternal, BlobProperties]) -> BlobProperties:
        if isinstance(item, BlobProperties):
            return item
        if isinstance(item, BlobItemInternal):
            blob = get_blob_properties_from_generated_code(item)
            blob.container = self.container  # type: ignore [assignment]
            return blob
        return item


class BlobNamesPaged(PageIterator):
    """An Iterable of Blob names."""

    service_endpoint: Optional[str]
    """The service URL."""
    prefix: Optional[str]
    """A blob name prefix being used to filter the list."""
    marker: Optional[str]
    """The continuation token of the current page of results."""
    results_per_page: Optional[int]
    """The maximum number of blobs to retrieve per call."""
    continuation_token: Optional[str]
    """The continuation token to retrieve the next page of results."""
    location_mode: Optional[str]
    """The location mode being used to list results. The available
    options include "primary" and "secondary"."""
    current_page: Optional[List[BlobProperties]]
    """The current page of listed results."""
    container: Optional[str]
    """The container that the blobs are listed from."""
    delimiter: Optional[str]
    """A delimiting character used for hierarchy listing."""
    command: Callable
    """Function to retrieve the next page of items."""

    def __init__(
        self, command: Callable,
        container: Optional[str] = None,
        prefix: Optional[str] = None,
        results_per_page: Optional[int] = None,
        continuation_token: Optional[str] = None,
        location_mode: Optional[str] = None
    ) -> None:
        super(BlobNamesPaged, self).__init__(
            get_next=self._get_next_cb,
            extract_data=self._extract_data_cb,
            continuation_token=continuation_token or ""
        )
        self._command = command
        self.service_endpoint = None
        self.prefix = prefix
        self.marker = None
        self.results_per_page = results_per_page
        self.container = container
        self.current_page = None
        self.location_mode = location_mode

    def _get_next_cb(self, continuation_token):
        try:
            return self._command(
                prefix=self.prefix,
                marker=continuation_token or None,
                maxresults=self.results_per_page,
                cls=return_raw_deserialized,
                use_location=self.location_mode)
        except HttpResponseError as error:
            process_storage_error(error)

    def _extract_data_cb(self, get_next_return):
        self.location_mode, self._response = get_next_return
        self.service_endpoint = self._response.get('ServiceEndpoint')
        self.prefix = load_xml_string(self._response, 'Prefix')
        self.marker = load_xml_string(self._response, 'Marker')
        self.results_per_page = load_xml_int(self._response, 'MaxResults')
        self.container = self._response.get('ContainerName')

        blobs = load_many_xml_nodes(self._response, 'Blob', wrapper='Blobs')
        self.current_page = [load_xml_string(blob, 'Name') for blob in blobs]

        next_marker = load_xml_string(self._response, 'NextMarker')
        return next_marker or None, self.current_page


class BlobPrefixPaged(BlobPropertiesPaged):
    def __init__(self, *args, **kwargs):
        super(BlobPrefixPaged, self).__init__(*args, **kwargs)
        self.name = self.prefix

    def _extract_data_cb(self, get_next_return):
        continuation_token, _ = super(BlobPrefixPaged, self)._extract_data_cb(get_next_return)
        self.current_page = self._response.segment.blob_prefixes + self._response.segment.blob_items
        self.current_page = [self._build_item(item) for item in self.current_page]
        self.delimiter = self._response.delimiter

        return continuation_token, self.current_page

    def _build_item(self, item):
        item = super(BlobPrefixPaged, self)._build_item(item)
        if isinstance(item, GenBlobPrefix):
            if item.name.encoded:
                name = unquote(item.name.content)
            else:
                name = item.name.content
            return BlobPrefix(
                self._command,
                container=self.container,
                prefix=name,
                results_per_page=self.results_per_page,
                location_mode=self.location_mode)
        return item


class BlobPrefix(ItemPaged, DictMixin):
    """An Iterable of Blob properties.

    Returned from walk_blobs when a delimiter is used.
    Can be thought of as a virtual blob directory."""

    name: str
    """The prefix, or "directory name" of the blob."""
    service_endpoint: Optional[str]
    """The service URL."""
    prefix: str
    """A blob name prefix being used to filter the list."""
    marker: Optional[str]
    """The continuation token of the current page of results."""
    results_per_page: Optional[int]
    """The maximum number of results retrieved per API call."""
    next_marker: Optional[str]
    """The continuation token to retrieve the next page of results."""
    location_mode: str
    """The location mode being used to list results. The available
    options include "primary" and "secondary"."""
    current_page: Optional[List[BlobProperties]]
    """The current page of listed results."""
    delimiter: str
    """A delimiting character used for hierarchy listing."""
    command: Callable
    """Function to retrieve the next page of items."""
    container: str
    """The name of the container."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(BlobPrefix, self).__init__(*args, page_iterator_class=BlobPrefixPaged, **kwargs)
        self.name = kwargs.get('prefix')  # type: ignore [assignment]
        self.prefix = kwargs.get('prefix')  # type: ignore [assignment]
        self.results_per_page = kwargs.get('results_per_page')
        self.container = kwargs.get('container')  # type: ignore [assignment]
        self.delimiter = kwargs.get('delimiter')  # type: ignore [assignment]
        self.location_mode = kwargs.get('location_mode')  # type: ignore [assignment]


class FilteredBlobPaged(PageIterator):
    """An Iterable of Blob properties."""

    service_endpoint: Optional[str]
    """The service URL."""
    prefix: Optional[str]
    """A blob name prefix being used to filter the list."""
    marker: Optional[str]
    """The continuation token of the current page of results."""
    results_per_page: Optional[int]
    """The maximum number of results retrieved per API call."""
    continuation_token: Optional[str]
    """The continuation token to retrieve the next page of results."""
    location_mode: Optional[str]
    """The location mode being used to list results. The available
    options include "primary" and "secondary"."""
    current_page: Optional[List[BlobProperties]]
    """The current page of listed results."""
    command: Callable
    """Function to retrieve the next page of items."""
    container: Optional[str]
    """The name of the container."""

    def __init__(
        self, command: Callable,
        container: Optional[str] = None,
        results_per_page: Optional[int] = None,
        continuation_token: Optional[str] = None,
        location_mode: Optional[str] = None
    ) -> None:
        super(FilteredBlobPaged, self).__init__(
            get_next=self._get_next_cb,
            extract_data=self._extract_data_cb,
            continuation_token=continuation_token or ""
        )
        self._command = command
        self.service_endpoint = None
        self.marker = continuation_token
        self.results_per_page = results_per_page
        self.container = container
        self.current_page = None
        self.location_mode = location_mode

    def _get_next_cb(self, continuation_token):
        try:
            return self._command(
                marker=continuation_token or None,
                maxresults=self.results_per_page,
                cls=return_context_and_deserialized,
                use_location=self.location_mode)
        except HttpResponseError as error:
            process_storage_error(error)

    def _extract_data_cb(self, get_next_return):
        self.location_mode, self._response = get_next_return
        self.service_endpoint = self._response.service_endpoint
        self.marker = self._response.next_marker
        self.current_page = [self._build_item(item) for item in self._response.blobs]

        return self._response.next_marker or None, self.current_page

    @staticmethod
    def _build_item(item):
        if isinstance(item, FilterBlobItem):
            tags = parse_tags(item.tags)
            blob = FilteredBlob(name=item.name, container_name=item.container_name, tags=tags)
            return blob
        return item
