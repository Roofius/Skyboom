from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Optional, Literal, Tuple, List

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Bank(str, Enum):
    leobank = "Leobank"
    unibank = "Unibank"


class Department(str, Enum):
    call_center = "Кол-центр"
    meo = "МЭО"
    hd = "HD"
    sd = "SD"
    soft = "SOFT"
    hard = "HARD"
    underwriting = "Underwriting"
    nps = "NPS"
    telemarketing = "Telemarketing"


class Direction(str, Enum):
    chat = "Чат"
    call = "Звонок"


class RequestStatus(str, Enum):
    waiting = "Ожидание"
    approved = "Одобрено"
    active = "Активно"
    finished = "Завершено"
    rejected = "Отклонено"


class OperatorBinding(BaseModel):
    bank: Bank
    department: Department
    direction: Optional[Direction] = None


class OnboardingRequest(BaseModel):
    bank: Bank
    department: Department
    direction: Optional[Direction] = None


class BreakRequestCreate(BaseModel):
    duration_minutes: Literal[5, 7, 10]


class BreakRequestResponse(BaseModel):
    id: str
    status: RequestStatus
    duration_minutes: int
    created_at: datetime
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None


class OperatorStatusResponse(BaseModel):
    operator_id: str
    binding: OperatorBinding
    latest_request: Optional[BreakRequestResponse]
    queue_position: Optional[int] = None
    queue_ahead: Optional[int] = None
    occupied_slots: int = 0
    limit: int = 0
    auto_reject_at: Optional[datetime] = None
    break_end_at: Optional[datetime] = None
    break_overdue_seconds: Optional[int] = None
    cooldown_remaining_seconds: Optional[int] = None


@dataclass
class BreakRequestRecord:
    id: str
    status: RequestStatus
    duration_minutes: int
    created_at: datetime
    operator_id: str
    binding: OperatorBinding
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None


@dataclass
class OperatorRecord:
    operator_id: str
    binding: OperatorBinding
    requests: List[str] = field(default_factory=list)


@dataclass
class UnitSettings:
    auto_approve: bool = False
    limit: int = 1


@dataclass
class UnitState:
    queue: List[str] = field(default_factory=list)


app = FastAPI(title="Skyboom Breaks API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

operators: Dict[str, OperatorRecord] = {}
requests: Dict[str, BreakRequestRecord] = {}
unit_settings: Dict[Tuple[str, str, Optional[str]], UnitSettings] = {}
unit_states: Dict[Tuple[str, str, Optional[str]], UnitState] = {}


def get_operator_id(x_user_id: Optional[str] = Header(default=None)) -> str:
    # TODO: replace this header-based identity with Keycloak auth integration.
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identity header.",
        )
    return x_user_id


def validate_binding(binding: OnboardingRequest) -> None:
    if binding.bank == Bank.unibank and binding.department != Department.call_center:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unibank supports only Call Center department.",
        )
    if binding.department == Department.call_center:
        if binding.direction is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Call Center requires a direction.",
            )
    else:
        if binding.direction is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Direction is only allowed for Call Center.",
            )


def _unit_key(binding: OperatorBinding) -> Tuple[str, str, Optional[str]]:
    return (binding.bank.value, binding.department.value, binding.direction.value if binding.direction else None)


def _get_settings(key: Tuple[str, str, Optional[str]]) -> UnitSettings:
    if key not in unit_settings:
        unit_settings[key] = UnitSettings()
    return unit_settings[key]


def _get_state(key: Tuple[str, str, Optional[str]]) -> UnitState:
    if key not in unit_states:
        unit_states[key] = UnitState()
    return unit_states[key]


def _occupied_slots(key: Tuple[str, str, Optional[str]]) -> int:
    return sum(
        1
        for request in requests.values()
        if _unit_key(request.binding) == key
        and request.status in {RequestStatus.approved, RequestStatus.active}
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _last_end_time(operator: OperatorRecord) -> Optional[datetime]:
    for request_id in reversed(operator.requests):
        record = requests.get(request_id)
        if not record:
            continue
        if record.status == RequestStatus.finished and record.finished_at:
            return record.finished_at
        if record.status == RequestStatus.rejected and record.rejected_at:
            return record.rejected_at
    return None


def _process_timeouts(key: Tuple[str, str, Optional[str]]) -> None:
    now = _now()
    for request in list(requests.values()):
        if _unit_key(request.binding) != key:
            continue
        if request.status == RequestStatus.approved and request.approved_at:
            if now - request.approved_at >= timedelta(minutes=5):
                request.status = RequestStatus.rejected
                request.rejected_at = now
    _process_auto_approve(key)


def _process_auto_approve(key: Tuple[str, str, Optional[str]]) -> None:
    settings = _get_settings(key)
    if not settings.auto_approve:
        return
    state = _get_state(key)
    while state.queue and _occupied_slots(key) < settings.limit:
        request_id = state.queue.pop(0)
        request = requests.get(request_id)
        if not request or request.status != RequestStatus.waiting:
            continue
        request.status = RequestStatus.approved
        request.approved_at = _now()


@app.post("/api/onboarding", response_model=OperatorStatusResponse)
async def onboarding(
    payload: OnboardingRequest,
    operator_id: str = Depends(get_operator_id),
) -> OperatorStatusResponse:
    validate_binding(payload)
    if operator_id in operators:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Onboarding already completed.",
        )
    binding = OperatorBinding(**payload.model_dump())
    operators[operator_id] = OperatorRecord(operator_id=operator_id, binding=binding)
    return OperatorStatusResponse(operator_id=operator_id, binding=binding, latest_request=None)


@app.get("/api/operator", response_model=OperatorStatusResponse)
async def operator_status(operator_id: str = Depends(get_operator_id)) -> OperatorStatusResponse:
    operator = operators.get(operator_id)
    if not operator:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operator not onboarded.")
    latest_request = requests.get(operator.requests[-1]) if operator.requests else None
    key = _unit_key(operator.binding)
    _process_timeouts(key)
    state = _get_state(key)
    settings = _get_settings(key)
    queue_position = None
    queue_ahead = None
    if latest_request and latest_request.status == RequestStatus.waiting:
        if latest_request.id in state.queue:
            queue_position = state.queue.index(latest_request.id) + 1
            queue_ahead = queue_position - 1
    now = _now()
    auto_reject_at = None
    break_end_at = None
    break_overdue_seconds = None
    if latest_request and latest_request.status == RequestStatus.approved and latest_request.approved_at:
        auto_reject_at = latest_request.approved_at + timedelta(minutes=5)
    if latest_request and latest_request.status == RequestStatus.active and latest_request.started_at:
        break_end_at = latest_request.started_at + timedelta(minutes=latest_request.duration_minutes)
        if now > break_end_at:
            break_overdue_seconds = int((now - break_end_at).total_seconds())
    cooldown_remaining = None
    last_end = _last_end_time(operator)
    if last_end:
        cooldown_end = last_end + timedelta(minutes=3)
        if now < cooldown_end:
            cooldown_remaining = int((cooldown_end - now).total_seconds())
    return OperatorStatusResponse(
        operator_id=operator.operator_id,
        binding=operator.binding,
        latest_request=_to_response(latest_request),
        queue_position=queue_position,
        queue_ahead=queue_ahead,
        occupied_slots=_occupied_slots(key),
        limit=settings.limit,
        auto_reject_at=auto_reject_at,
        break_end_at=break_end_at,
        break_overdue_seconds=break_overdue_seconds,
        cooldown_remaining_seconds=cooldown_remaining,
    )


@app.post("/api/requests", response_model=BreakRequestResponse)
async def create_request(
    payload: BreakRequestCreate,
    operator_id: str = Depends(get_operator_id),
) -> BreakRequestResponse:
    operator = operators.get(operator_id)
    if not operator:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operator not onboarded.")
    key = _unit_key(operator.binding)
    _process_timeouts(key)
    last_end = _last_end_time(operator)
    if last_end and _now() - last_end < timedelta(minutes=3):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cooldown period between breaks has not elapsed.",
        )
    latest_request = operator.requests[-1] if operator.requests else None
    latest_request_record = requests.get(latest_request) if latest_request else None
    if latest_request_record and latest_request_record.status in {
        RequestStatus.waiting,
        RequestStatus.approved,
        RequestStatus.active,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Active break request already exists.",
        )
    request_id = f"{operator_id}-req-{len(operator.requests) + 1}"
    created_at = _now()
    new_request = BreakRequestRecord(
        id=request_id,
        status=RequestStatus.waiting,
        duration_minutes=payload.duration_minutes,
        created_at=created_at,
        operator_id=operator_id,
        binding=operator.binding,
    )
    requests[request_id] = new_request
    operator.requests.append(request_id)
    state = _get_state(key)
    state.queue.append(request_id)
    _process_auto_approve(key)
    return _to_response(requests[request_id])


@app.post("/api/requests/{request_id}/approve", response_model=BreakRequestResponse)
async def approve_request(
    request_id: str,
    operator_id: str = Depends(get_operator_id),
) -> BreakRequestResponse:
    request = _get_request(operator_id, request_id)
    key = _unit_key(request.binding)
    _process_timeouts(key)
    if request.status != RequestStatus.waiting:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only waiting requests can be approved.",
        )
    request.status = RequestStatus.approved
    request.approved_at = _now()
    state = _get_state(key)
    if request.id in state.queue:
        state.queue.remove(request.id)
    return _to_response(request)


@app.post("/api/requests/{request_id}/start", response_model=BreakRequestResponse)
async def start_break(
    request_id: str,
    operator_id: str = Depends(get_operator_id),
) -> BreakRequestResponse:
    request = _get_request(operator_id, request_id)
    key = _unit_key(request.binding)
    _process_timeouts(key)
    if request.status != RequestStatus.approved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only approved requests can be started.",
        )
    request.status = RequestStatus.active
    request.started_at = _now()
    return _to_response(request)


@app.post("/api/requests/{request_id}/finish", response_model=BreakRequestResponse)
async def finish_break(
    request_id: str,
    operator_id: str = Depends(get_operator_id),
) -> BreakRequestResponse:
    request = _get_request(operator_id, request_id)
    key = _unit_key(request.binding)
    _process_timeouts(key)
    if request.status != RequestStatus.active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only active requests can be finished.",
        )
    request.status = RequestStatus.finished
    request.finished_at = _now()
    _process_auto_approve(key)
    return _to_response(request)


@app.post("/api/requests/{request_id}/reject", response_model=BreakRequestResponse)
async def reject_request(
    request_id: str,
    operator_id: str = Depends(get_operator_id),
) -> BreakRequestResponse:
    request = _get_request(operator_id, request_id)
    key = _unit_key(request.binding)
    _process_timeouts(key)
    if request.status not in {RequestStatus.waiting, RequestStatus.approved}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only waiting or approved requests can be rejected.",
        )
    request.status = RequestStatus.rejected
    request.rejected_at = _now()
    state = _get_state(key)
    if request.id in state.queue:
        state.queue.remove(request.id)
    _process_auto_approve(key)
    return _to_response(request)


class UnitSettingsPayload(BaseModel):
    auto_approve: bool
    limit: int = 1


@app.get("/api/admin/unit-settings", response_model=UnitSettingsPayload)
async def get_unit_settings(
    bank: Bank,
    department: Department,
    direction: Optional[Direction] = None,
) -> UnitSettingsPayload:
    validate_binding(OnboardingRequest(bank=bank, department=department, direction=direction))
    key = (bank.value, department.value, direction.value if direction else None)
    settings = _get_settings(key)
    return UnitSettingsPayload(auto_approve=settings.auto_approve, limit=settings.limit)


@app.put("/api/admin/unit-settings", response_model=UnitSettingsPayload)
async def update_unit_settings(
    payload: UnitSettingsPayload,
    bank: Bank,
    department: Department,
    direction: Optional[Direction] = None,
) -> UnitSettingsPayload:
    validate_binding(OnboardingRequest(bank=bank, department=department, direction=direction))
    if payload.limit < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Limit must be positive.")
    key = (bank.value, department.value, direction.value if direction else None)
    settings = _get_settings(key)
    settings.auto_approve = payload.auto_approve
    settings.limit = payload.limit
    _process_auto_approve(key)
    return UnitSettingsPayload(auto_approve=settings.auto_approve, limit=settings.limit)


def _get_request(operator_id: str, request_id: str) -> BreakRequestRecord:
    operator = operators.get(operator_id)
    if not operator:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operator not onboarded.")
    if request_id in operator.requests:
        request = requests.get(request_id)
        if request:
            return request
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found.")


def _to_response(request: Optional[BreakRequestRecord]) -> Optional[BreakRequestResponse]:
    if not request:
        return None
    return BreakRequestResponse(
        id=request.id,
        status=request.status,
        duration_minutes=request.duration_minutes,
        created_at=request.created_at,
        approved_at=request.approved_at,
        started_at=request.started_at,
        finished_at=request.finished_at,
        rejected_at=request.rejected_at,
    )
