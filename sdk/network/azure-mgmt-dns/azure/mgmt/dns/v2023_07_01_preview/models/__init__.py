# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._models_py3 import ARecord
from ._models_py3 import AaaaRecord
from ._models_py3 import CaaRecord
from ._models_py3 import CloudErrorBody
from ._models_py3 import CnameRecord
from ._models_py3 import DelegationSignerInfo
from ._models_py3 import Digest
from ._models_py3 import DnsResourceReference
from ._models_py3 import DnsResourceReferenceRequest
from ._models_py3 import DnsResourceReferenceResult
from ._models_py3 import DnssecConfig
from ._models_py3 import DnssecConfigListResult
from ._models_py3 import DsRecord
from ._models_py3 import MxRecord
from ._models_py3 import NaptrRecord
from ._models_py3 import NsRecord
from ._models_py3 import PtrRecord
from ._models_py3 import RecordSet
from ._models_py3 import RecordSetListResult
from ._models_py3 import RecordSetUpdateParameters
from ._models_py3 import Resource
from ._models_py3 import SigningKey
from ._models_py3 import SoaRecord
from ._models_py3 import SrvRecord
from ._models_py3 import SubResource
from ._models_py3 import SystemData
from ._models_py3 import TlsaRecord
from ._models_py3 import TxtRecord
from ._models_py3 import Zone
from ._models_py3 import ZoneListResult
from ._models_py3 import ZoneUpdate

from ._dns_management_client_enums import CreatedByType
from ._dns_management_client_enums import RecordType
from ._dns_management_client_enums import ZoneType
from ._patch import __all__ as _patch_all
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "ARecord",
    "AaaaRecord",
    "CaaRecord",
    "CloudErrorBody",
    "CnameRecord",
    "DelegationSignerInfo",
    "Digest",
    "DnsResourceReference",
    "DnsResourceReferenceRequest",
    "DnsResourceReferenceResult",
    "DnssecConfig",
    "DnssecConfigListResult",
    "DsRecord",
    "MxRecord",
    "NaptrRecord",
    "NsRecord",
    "PtrRecord",
    "RecordSet",
    "RecordSetListResult",
    "RecordSetUpdateParameters",
    "Resource",
    "SigningKey",
    "SoaRecord",
    "SrvRecord",
    "SubResource",
    "SystemData",
    "TlsaRecord",
    "TxtRecord",
    "Zone",
    "ZoneListResult",
    "ZoneUpdate",
    "CreatedByType",
    "RecordType",
    "ZoneType",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()