# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


class Action(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The action type in requests for bulk upload or download of a DNS resolver domain list."""

    UPLOAD = "Upload"
    DOWNLOAD = "Download"


class ActionType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of action to take."""

    ALLOW = "Allow"
    ALERT = "Alert"
    BLOCK = "Block"


class CreatedByType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The type of identity that created the resource."""

    USER = "User"
    APPLICATION = "Application"
    MANAGED_IDENTITY = "ManagedIdentity"
    KEY = "Key"


class DnsResolverState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The current status of the DNS resolver. This is a read-only property and any attempt to set
    this value will be ignored.
    """

    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"


class DnsSecurityRuleState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The state of DNS security rule."""

    ENABLED = "Enabled"
    DISABLED = "Disabled"


class ForwardingRuleState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The state of forwarding rule."""

    ENABLED = "Enabled"
    DISABLED = "Disabled"


class IpAllocationMethod(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Private IP address allocation method."""

    STATIC = "Static"
    DYNAMIC = "Dynamic"


class ProvisioningState(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The current provisioning state of the resource."""

    CREATING = "Creating"
    UPDATING = "Updating"
    DELETING = "Deleting"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELED = "Canceled"
