import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import wrappers_pb2 as _wrappers_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CameraFrame(_message.Message):
    __slots__ = ("image_data", "width", "height", "timestamp", "format", "camera_id")
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    width: int
    height: int
    timestamp: _timestamp_pb2.Timestamp
    format: str
    camera_id: int
    def __init__(self, image_data: _Optional[bytes] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., format: _Optional[str] = ..., camera_id: _Optional[int] = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("x_min", "y_min", "x_max", "y_max", "label", "confidence")
    X_MIN_FIELD_NUMBER: _ClassVar[int]
    Y_MIN_FIELD_NUMBER: _ClassVar[int]
    X_MAX_FIELD_NUMBER: _ClassVar[int]
    Y_MAX_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float
    def __init__(self, x_min: _Optional[int] = ..., y_min: _Optional[int] = ..., x_max: _Optional[int] = ..., y_max: _Optional[int] = ..., label: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class DetectionResult(_message.Message):
    __slots__ = ("frame_timestamp", "json", "camera_id")
    FRAME_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    JSON_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    frame_timestamp: _timestamp_pb2.Timestamp
    json: str
    camera_id: int
    def __init__(self, frame_timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., json: _Optional[str] = ..., camera_id: _Optional[int] = ...) -> None: ...

class DeviceStatus(_message.Message):
    __slots__ = ("device_id", "state", "cpu_temperature_c", "frame_rate_fps", "processing_latency_ms", "firmware_version", "is_sleeping")
    class ConnectionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DISCONNECTED: _ClassVar[DeviceStatus.ConnectionState]
        BLE_PAIRING: _ClassVar[DeviceStatus.ConnectionState]
        WLAN_CONNECTED: _ClassVar[DeviceStatus.ConnectionState]
        GRPC_READY: _ClassVar[DeviceStatus.ConnectionState]
    DISCONNECTED: DeviceStatus.ConnectionState
    BLE_PAIRING: DeviceStatus.ConnectionState
    WLAN_CONNECTED: DeviceStatus.ConnectionState
    GRPC_READY: DeviceStatus.ConnectionState
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CPU_TEMPERATURE_C_FIELD_NUMBER: _ClassVar[int]
    FRAME_RATE_FPS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    FIRMWARE_VERSION_FIELD_NUMBER: _ClassVar[int]
    IS_SLEEPING_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    state: DeviceStatus.ConnectionState
    cpu_temperature_c: float
    frame_rate_fps: int
    processing_latency_ms: int
    firmware_version: str
    is_sleeping: bool
    def __init__(self, device_id: _Optional[str] = ..., state: _Optional[_Union[DeviceStatus.ConnectionState, str]] = ..., cpu_temperature_c: _Optional[float] = ..., frame_rate_fps: _Optional[int] = ..., processing_latency_ms: _Optional[int] = ..., firmware_version: _Optional[str] = ..., is_sleeping: bool = ...) -> None: ...

class Command(_message.Message):
    __slots__ = ("config_request", "control_action", "heartbeat", "detection_result", "camera_frame")
    CONFIG_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CONTROL_ACTION_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    DETECTION_RESULT_FIELD_NUMBER: _ClassVar[int]
    CAMERA_FRAME_FIELD_NUMBER: _ClassVar[int]
    config_request: ConfigRequest
    control_action: ControlAction
    heartbeat: Heartbeat
    detection_result: DetectionResult
    camera_frame: CameraFrame
    def __init__(self, config_request: _Optional[_Union[ConfigRequest, _Mapping]] = ..., control_action: _Optional[_Union[ControlAction, _Mapping]] = ..., heartbeat: _Optional[_Union[Heartbeat, _Mapping]] = ..., detection_result: _Optional[_Union[DetectionResult, _Mapping]] = ..., camera_frame: _Optional[_Union[CameraFrame, _Mapping]] = ...) -> None: ...

class ConfigRequest(_message.Message):
    __slots__ = ("detection_threshold", "sleep_timeout_sec")
    DETECTION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SLEEP_TIMEOUT_SEC_FIELD_NUMBER: _ClassVar[int]
    detection_threshold: _wrappers_pb2.FloatValue
    sleep_timeout_sec: _wrappers_pb2.UInt32Value
    def __init__(self, detection_threshold: _Optional[_Union[_wrappers_pb2.FloatValue, _Mapping]] = ..., sleep_timeout_sec: _Optional[_Union[_wrappers_pb2.UInt32Value, _Mapping]] = ...) -> None: ...

class ControlAction(_message.Message):
    __slots__ = ("action",)
    class ActionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        REBOOT: _ClassVar[ControlAction.ActionType]
        START_STREAMING: _ClassVar[ControlAction.ActionType]
        STOP_STREAMING: _ClassVar[ControlAction.ActionType]
    REBOOT: ControlAction.ActionType
    START_STREAMING: ControlAction.ActionType
    STOP_STREAMING: ControlAction.ActionType
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: ControlAction.ActionType
    def __init__(self, action: _Optional[_Union[ControlAction.ActionType, str]] = ...) -> None: ...

class Heartbeat(_message.Message):
    __slots__ = ("timestamp",)
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("camera_frame", "detection_result", "device_status", "config_response")
    CAMERA_FRAME_FIELD_NUMBER: _ClassVar[int]
    DETECTION_RESULT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_STATUS_FIELD_NUMBER: _ClassVar[int]
    CONFIG_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    camera_frame: CameraFrame
    detection_result: DetectionResult
    device_status: DeviceStatus
    config_response: ConfigResponse
    def __init__(self, camera_frame: _Optional[_Union[CameraFrame, _Mapping]] = ..., detection_result: _Optional[_Union[DetectionResult, _Mapping]] = ..., device_status: _Optional[_Union[DeviceStatus, _Mapping]] = ..., config_response: _Optional[_Union[ConfigResponse, _Mapping]] = ...) -> None: ...

class ClientMessage(_message.Message):
    __slots__ = ("device_status", "config_response")
    DEVICE_STATUS_FIELD_NUMBER: _ClassVar[int]
    CONFIG_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    device_status: DeviceStatus
    config_response: ConfigResponse
    def __init__(self, device_status: _Optional[_Union[DeviceStatus, _Mapping]] = ..., config_response: _Optional[_Union[ConfigResponse, _Mapping]] = ...) -> None: ...

class ConfigResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...
