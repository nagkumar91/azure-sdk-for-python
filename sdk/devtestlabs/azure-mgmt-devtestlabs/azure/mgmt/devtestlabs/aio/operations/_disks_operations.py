# pylint: disable=too-many-lines,too-many-statements
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from io import IOBase
import sys
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, IO, Optional, Type, TypeVar, Union, cast, overload
import urllib.parse

from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ResourceNotModifiedError,
    StreamClosedError,
    StreamConsumedError,
    map_error,
)
from azure.core.pipeline import PipelineResponse
from azure.core.polling import AsyncLROPoller, AsyncNoPolling, AsyncPollingMethod
from azure.core.rest import AsyncHttpResponse, HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.tracing.decorator_async import distributed_trace_async
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.async_arm_polling import AsyncARMPolling

from ... import models as _models
from ...operations._disks_operations import (
    build_attach_request,
    build_create_or_update_request,
    build_delete_request,
    build_detach_request,
    build_get_request,
    build_list_request,
    build_update_request,
)

if sys.version_info >= (3, 9):
    from collections.abc import MutableMapping
else:
    from typing import MutableMapping  # type: ignore  # pylint: disable=ungrouped-imports
T = TypeVar("T")
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]


class DisksOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.devtestlabs.aio.DevTestLabsClient`'s
        :attr:`disks` attribute.
    """

    models = _models

    def __init__(self, *args, **kwargs) -> None:
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop("client")
        self._config = input_args.pop(0) if input_args else kwargs.pop("config")
        self._serialize = input_args.pop(0) if input_args else kwargs.pop("serializer")
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop("deserializer")

    @distributed_trace
    def list(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        expand: Optional[str] = None,
        filter: Optional[str] = None,
        top: Optional[int] = None,
        orderby: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable["_models.Disk"]:
        """List disks in a given user profile.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param expand: Specify the $expand query. Example: 'properties($select=diskType)'. Default
         value is None.
        :type expand: str
        :param filter: The filter to apply to the operation. Example: '$filter=contains(name,'myName').
         Default value is None.
        :type filter: str
        :param top: The maximum number of resources to return from the operation. Example: '$top=10'.
         Default value is None.
        :type top: int
        :param orderby: The ordering expression for the results, using OData notation. Example:
         '$orderby=name desc'. Default value is None.
        :type orderby: str
        :return: An iterator like instance of either Disk or the result of cls(response)
        :rtype: ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.devtestlabs.models.Disk]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[_models.DiskList] = kwargs.pop("cls", None)

        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        def prepare_request(next_link=None):
            if not next_link:

                _request = build_list_request(
                    resource_group_name=resource_group_name,
                    lab_name=lab_name,
                    user_name=user_name,
                    subscription_id=self._config.subscription_id,
                    expand=expand,
                    filter=filter,
                    top=top,
                    orderby=orderby,
                    api_version=api_version,
                    headers=_headers,
                    params=_params,
                )
                _request.url = self._client.format_url(_request.url)

            else:
                # make call to next link with the client's api-version
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict(
                    {
                        key: [urllib.parse.quote(v) for v in value]
                        for key, value in urllib.parse.parse_qs(_parsed_next_link.query).items()
                    }
                )
                _next_request_params["api-version"] = self._config.api_version
                _request = HttpRequest(
                    "GET", urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params
                )
                _request.url = self._client.format_url(_request.url)
                _request.method = "GET"
            return _request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize("DiskList", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)  # type: ignore
            return deserialized.next_link or None, AsyncList(list_of_elem)

        async def get_next(next_link=None):
            _request = prepare_request(next_link)

            _stream = False
            pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
                _request, stream=_stream, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response

        return AsyncItemPaged(get_next, extract_data)

    @distributed_trace_async
    async def get(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        expand: Optional[str] = None,
        **kwargs: Any
    ) -> _models.Disk:
        """Get disk.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param expand: Specify the $expand query. Example: 'properties($select=diskType)'. Default
         value is None.
        :type expand: str
        :return: Disk or the result of cls(response)
        :rtype: ~azure.mgmt.devtestlabs.models.Disk
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[_models.Disk] = kwargs.pop("cls", None)

        _request = build_get_request(
            resource_group_name=resource_group_name,
            lab_name=lab_name,
            user_name=user_name,
            name=name,
            subscription_id=self._config.subscription_id,
            expand=expand,
            api_version=api_version,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _stream = False
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize("Disk", pipeline_response.http_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    async def _create_or_update_initial(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        disk: Union[_models.Disk, IO[bytes]],
        **kwargs: Any
    ) -> AsyncIterator[bytes]:
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: Optional[str] = kwargs.pop("content_type", _headers.pop("Content-Type", None))
        cls: ClsType[AsyncIterator[bytes]] = kwargs.pop("cls", None)

        content_type = content_type or "application/json"
        _json = None
        _content = None
        if isinstance(disk, (IOBase, bytes)):
            _content = disk
        else:
            _json = self._serialize.body(disk, "Disk")

        _request = build_create_or_update_request(
            resource_group_name=resource_group_name,
            lab_name=lab_name,
            user_name=user_name,
            name=name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            content=_content,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _decompress = kwargs.pop("decompress", True)
        _stream = True
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 201]:
            try:
                await response.read()  # Load the body in memory and close the socket
            except (StreamConsumedError, StreamClosedError):
                pass
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = response.stream_download(self._client._pipeline, decompress=_decompress)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @overload
    async def begin_create_or_update(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        disk: _models.Disk,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> AsyncLROPoller[_models.Disk]:
        """Create or replace an existing disk. This operation can take a while to complete.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param disk: A Disk. Required.
        :type disk: ~azure.mgmt.devtestlabs.models.Disk
        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.
         Default value is "application/json".
        :paramtype content_type: str
        :return: An instance of AsyncLROPoller that returns either Disk or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[~azure.mgmt.devtestlabs.models.Disk]
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @overload
    async def begin_create_or_update(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        disk: IO[bytes],
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> AsyncLROPoller[_models.Disk]:
        """Create or replace an existing disk. This operation can take a while to complete.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param disk: A Disk. Required.
        :type disk: IO[bytes]
        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.
         Default value is "application/json".
        :paramtype content_type: str
        :return: An instance of AsyncLROPoller that returns either Disk or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[~azure.mgmt.devtestlabs.models.Disk]
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @distributed_trace_async
    async def begin_create_or_update(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        disk: Union[_models.Disk, IO[bytes]],
        **kwargs: Any
    ) -> AsyncLROPoller[_models.Disk]:
        """Create or replace an existing disk. This operation can take a while to complete.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param disk: A Disk. Is either a Disk type or a IO[bytes] type. Required.
        :type disk: ~azure.mgmt.devtestlabs.models.Disk or IO[bytes]
        :return: An instance of AsyncLROPoller that returns either Disk or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[~azure.mgmt.devtestlabs.models.Disk]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: Optional[str] = kwargs.pop("content_type", _headers.pop("Content-Type", None))
        cls: ClsType[_models.Disk] = kwargs.pop("cls", None)
        polling: Union[bool, AsyncPollingMethod] = kwargs.pop("polling", True)
        lro_delay = kwargs.pop("polling_interval", self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop("continuation_token", None)
        if cont_token is None:
            raw_result = await self._create_or_update_initial(
                resource_group_name=resource_group_name,
                lab_name=lab_name,
                user_name=user_name,
                name=name,
                disk=disk,
                api_version=api_version,
                content_type=content_type,
                cls=lambda x, y, z: x,
                headers=_headers,
                params=_params,
                **kwargs
            )
            await raw_result.http_response.read()  # type: ignore
        kwargs.pop("error_map", None)

        def get_long_running_output(pipeline_response):
            deserialized = self._deserialize("Disk", pipeline_response.http_response)
            if cls:
                return cls(pipeline_response, deserialized, {})  # type: ignore
            return deserialized

        if polling is True:
            polling_method: AsyncPollingMethod = cast(AsyncPollingMethod, AsyncARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(AsyncPollingMethod, AsyncNoPolling())
        else:
            polling_method = polling
        if cont_token:
            return AsyncLROPoller[_models.Disk].from_continuation_token(
                polling_method=polling_method,
                continuation_token=cont_token,
                client=self._client,
                deserialization_callback=get_long_running_output,
            )
        return AsyncLROPoller[_models.Disk](
            self._client, raw_result, get_long_running_output, polling_method  # type: ignore
        )

    async def _delete_initial(
        self, resource_group_name: str, lab_name: str, user_name: str, name: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[AsyncIterator[bytes]] = kwargs.pop("cls", None)

        _request = build_delete_request(
            resource_group_name=resource_group_name,
            lab_name=lab_name,
            user_name=user_name,
            name=name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _decompress = kwargs.pop("decompress", True)
        _stream = True
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 202, 204]:
            try:
                await response.read()  # Load the body in memory and close the socket
            except (StreamConsumedError, StreamClosedError):
                pass
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = response.stream_download(self._client._pipeline, decompress=_decompress)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @distributed_trace_async
    async def begin_delete(
        self, resource_group_name: str, lab_name: str, user_name: str, name: str, **kwargs: Any
    ) -> AsyncLROPoller[None]:
        """Delete disk. This operation can take a while to complete.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :return: An instance of AsyncLROPoller that returns either None or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[None]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        cls: ClsType[None] = kwargs.pop("cls", None)
        polling: Union[bool, AsyncPollingMethod] = kwargs.pop("polling", True)
        lro_delay = kwargs.pop("polling_interval", self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop("continuation_token", None)
        if cont_token is None:
            raw_result = await self._delete_initial(
                resource_group_name=resource_group_name,
                lab_name=lab_name,
                user_name=user_name,
                name=name,
                api_version=api_version,
                cls=lambda x, y, z: x,
                headers=_headers,
                params=_params,
                **kwargs
            )
            await raw_result.http_response.read()  # type: ignore
        kwargs.pop("error_map", None)

        def get_long_running_output(pipeline_response):  # pylint: disable=inconsistent-return-statements
            if cls:
                return cls(pipeline_response, None, {})  # type: ignore

        if polling is True:
            polling_method: AsyncPollingMethod = cast(AsyncPollingMethod, AsyncARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(AsyncPollingMethod, AsyncNoPolling())
        else:
            polling_method = polling
        if cont_token:
            return AsyncLROPoller[None].from_continuation_token(
                polling_method=polling_method,
                continuation_token=cont_token,
                client=self._client,
                deserialization_callback=get_long_running_output,
            )
        return AsyncLROPoller[None](self._client, raw_result, get_long_running_output, polling_method)  # type: ignore

    @distributed_trace_async
    async def update(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> _models.Disk:
        """Allows modifying tags of disks. All other properties will be ignored.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param tags: The tags of the resource. Default value is None.
        :type tags: dict[str, str]
        :return: Disk or the result of cls(response)
        :rtype: ~azure.mgmt.devtestlabs.models.Disk
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: str = kwargs.pop("content_type", _headers.pop("Content-Type", "application/json"))
        cls: ClsType[_models.Disk] = kwargs.pop("cls", None)

        _disk = _models.DiskFragment(tags=tags)
        _json = self._serialize.body(_disk, "DiskFragment")

        _request = build_update_request(
            resource_group_name=resource_group_name,
            lab_name=lab_name,
            user_name=user_name,
            name=name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _stream = False
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize("Disk", pipeline_response.http_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    async def _attach_initial(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        leased_by_lab_vm_id: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[bytes]:
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: str = kwargs.pop("content_type", _headers.pop("Content-Type", "application/json"))
        cls: ClsType[AsyncIterator[bytes]] = kwargs.pop("cls", None)

        _attach_disk_properties = _models.AttachDiskProperties(leased_by_lab_vm_id=leased_by_lab_vm_id)
        _json = self._serialize.body(_attach_disk_properties, "AttachDiskProperties")

        _request = build_attach_request(
            resource_group_name=resource_group_name,
            lab_name=lab_name,
            user_name=user_name,
            name=name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _decompress = kwargs.pop("decompress", True)
        _stream = True
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 202]:
            try:
                await response.read()  # Load the body in memory and close the socket
            except (StreamConsumedError, StreamClosedError):
                pass
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = response.stream_download(self._client._pipeline, decompress=_decompress)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @distributed_trace_async
    async def begin_attach(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        leased_by_lab_vm_id: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncLROPoller[None]:
        """Attach and create the lease of the disk to the virtual machine. This operation can take a while
        to complete.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param leased_by_lab_vm_id: The resource ID of the Lab virtual machine to which the disk is
         attached. Default value is None.
        :type leased_by_lab_vm_id: str
        :return: An instance of AsyncLROPoller that returns either None or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[None]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: str = kwargs.pop("content_type", _headers.pop("Content-Type", "application/json"))
        cls: ClsType[None] = kwargs.pop("cls", None)
        polling: Union[bool, AsyncPollingMethod] = kwargs.pop("polling", True)
        lro_delay = kwargs.pop("polling_interval", self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop("continuation_token", None)
        if cont_token is None:
            raw_result = await self._attach_initial(
                resource_group_name=resource_group_name,
                lab_name=lab_name,
                user_name=user_name,
                name=name,
                leased_by_lab_vm_id=leased_by_lab_vm_id,
                api_version=api_version,
                content_type=content_type,
                cls=lambda x, y, z: x,
                headers=_headers,
                params=_params,
                **kwargs
            )
            await raw_result.http_response.read()  # type: ignore
        kwargs.pop("error_map", None)

        def get_long_running_output(pipeline_response):  # pylint: disable=inconsistent-return-statements
            if cls:
                return cls(pipeline_response, None, {})  # type: ignore

        if polling is True:
            polling_method: AsyncPollingMethod = cast(AsyncPollingMethod, AsyncARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(AsyncPollingMethod, AsyncNoPolling())
        else:
            polling_method = polling
        if cont_token:
            return AsyncLROPoller[None].from_continuation_token(
                polling_method=polling_method,
                continuation_token=cont_token,
                client=self._client,
                deserialization_callback=get_long_running_output,
            )
        return AsyncLROPoller[None](self._client, raw_result, get_long_running_output, polling_method)  # type: ignore

    async def _detach_initial(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        leased_by_lab_vm_id: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[bytes]:
        error_map: MutableMapping[int, Type[HttpResponseError]] = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: str = kwargs.pop("content_type", _headers.pop("Content-Type", "application/json"))
        cls: ClsType[AsyncIterator[bytes]] = kwargs.pop("cls", None)

        _detach_disk_properties = _models.DetachDiskProperties(leased_by_lab_vm_id=leased_by_lab_vm_id)
        _json = self._serialize.body(_detach_disk_properties, "DetachDiskProperties")

        _request = build_detach_request(
            resource_group_name=resource_group_name,
            lab_name=lab_name,
            user_name=user_name,
            name=name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            headers=_headers,
            params=_params,
        )
        _request.url = self._client.format_url(_request.url)

        _decompress = kwargs.pop("decompress", True)
        _stream = True
        pipeline_response: PipelineResponse = await self._client._pipeline.run(  # pylint: disable=protected-access
            _request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 202]:
            try:
                await response.read()  # Load the body in memory and close the socket
            except (StreamConsumedError, StreamClosedError):
                pass
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = response.stream_download(self._client._pipeline, decompress=_decompress)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    @distributed_trace_async
    async def begin_detach(
        self,
        resource_group_name: str,
        lab_name: str,
        user_name: str,
        name: str,
        leased_by_lab_vm_id: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncLROPoller[None]:
        """Detach and break the lease of the disk attached to the virtual machine. This operation can take
        a while to complete.

        :param resource_group_name: The name of the resource group. Required.
        :type resource_group_name: str
        :param lab_name: The name of the lab. Required.
        :type lab_name: str
        :param user_name: The name of the user profile. Required.
        :type user_name: str
        :param name: The name of the disk. Required.
        :type name: str
        :param leased_by_lab_vm_id: The resource ID of the Lab VM to which the disk is attached.
         Default value is None.
        :type leased_by_lab_vm_id: str
        :return: An instance of AsyncLROPoller that returns either None or the result of cls(response)
        :rtype: ~azure.core.polling.AsyncLROPoller[None]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: str = kwargs.pop("api_version", _params.pop("api-version", self._config.api_version))
        content_type: str = kwargs.pop("content_type", _headers.pop("Content-Type", "application/json"))
        cls: ClsType[None] = kwargs.pop("cls", None)
        polling: Union[bool, AsyncPollingMethod] = kwargs.pop("polling", True)
        lro_delay = kwargs.pop("polling_interval", self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop("continuation_token", None)
        if cont_token is None:
            raw_result = await self._detach_initial(
                resource_group_name=resource_group_name,
                lab_name=lab_name,
                user_name=user_name,
                name=name,
                leased_by_lab_vm_id=leased_by_lab_vm_id,
                api_version=api_version,
                content_type=content_type,
                cls=lambda x, y, z: x,
                headers=_headers,
                params=_params,
                **kwargs
            )
            await raw_result.http_response.read()  # type: ignore
        kwargs.pop("error_map", None)

        def get_long_running_output(pipeline_response):  # pylint: disable=inconsistent-return-statements
            if cls:
                return cls(pipeline_response, None, {})  # type: ignore

        if polling is True:
            polling_method: AsyncPollingMethod = cast(AsyncPollingMethod, AsyncARMPolling(lro_delay, **kwargs))
        elif polling is False:
            polling_method = cast(AsyncPollingMethod, AsyncNoPolling())
        else:
            polling_method = polling
        if cont_token:
            return AsyncLROPoller[None].from_continuation_token(
                polling_method=polling_method,
                continuation_token=cont_token,
                client=self._client,
                deserialization_callback=get_long_running_output,
            )
        return AsyncLROPoller[None](self._client, raw_result, get_long_running_output, polling_method)  # type: ignore
