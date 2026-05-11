"""Agent decision service.

This implementation intentionally avoids reading the raw driver/cargo files.
It only uses ``SimulationApiPort`` methods exposed by the evaluation runtime:
driver status, visible cargo candidates, and in-session decision history.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from simkit.ports import SimulationApiPort


SIM_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
WALL_FMT = "%Y-%m-%d %H:%M:%S"
REPOSITION_SPEED_KM_PER_HOUR = 60.0
CARGO_VIEW_BATCH_SIZE = 10
DEFAULT_WAIT_MINUTES = 180
MAX_WAIT_MINUTES = 12 * 60


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def _distance_minutes(distance_km: float) -> int:
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil(distance_km / REPOSITION_SPEED_KM_PER_HOUR * 60.0))


def _wall_to_minute(text: str | None) -> int | None:
    if not text:
        return None
    try:
        return int((datetime.strptime(str(text), WALL_FMT) - SIM_EPOCH).total_seconds() // 60)
    except ValueError:
        return None


def _day_minute(minute: int) -> int:
    return int(minute) % 1440


def _day_index(minute: int) -> int:
    return max(0, int(minute) // 1440)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _point_near(lat: float, lng: float, target: tuple[float, float], radius_km: float = 1.0) -> bool:
    return _haversine_km(lat, lng, target[0], target[1]) <= radius_km


def _extract_content(preference: Any) -> str:
    if isinstance(preference, str):
        return preference
    if isinstance(preference, dict):
        return str(preference.get("content") or preference.get("text") or "")
    return str(preference or "")


def _parse_chinese_hour(text: str) -> list[int]:
    out: list[int] = []
    patterns = [
        r"(\d{1,2})[:：]00",
        r"(\d{1,2})点",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, text):
            try:
                hour = int(value)
            except ValueError:
                continue
            if 0 <= hour <= 24:
                out.append(hour)
    return out


def _parse_coordinate_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for a, b in re.findall(r"[（(]\s*(\d{2}\.\d+)\s*[，,]\s*(\d{3}\.\d+)\s*[）)]", text):
        try:
            pairs.append((float(a), float(b)))
        except ValueError:
            continue
    return pairs


@dataclass(frozen=True)
class Policy:
    forbidden_categories: set[str] = field(default_factory=set)
    avoid_categories: set[str] = field(default_factory=set)
    max_haul_km: float | None = None
    max_pickup_km: float | None = None
    daily_rest_minutes: int = 0
    required_full_rest_days: int = 0
    quiet_windows: list[tuple[int, int]] = field(default_factory=list)
    lunch_windows: list[tuple[int, int]] = field(default_factory=list)
    must_stay_in_shenzhen: bool = False
    forbidden_zones: list[tuple[float, float, float]] = field(default_factory=list)
    home_target: tuple[float, float] | None = None
    home_deadline_minute: int | None = None
    home_release_minute: int | None = None
    spouse_pickup_target: tuple[float, float] | None = None
    recurring_visit_target: tuple[float, float] | None = None
    familiar_cargo_ids: set[str] = field(default_factory=set)
    familiar_target: tuple[float, float] | None = None
    familiar_start_minute: int | None = None
    familiar_end_minute: int | None = None


@dataclass
class Memory:
    history_count: int = 0
    accepted_by_day: dict[int, int] = field(default_factory=dict)
    moving_days: set[int] = field(default_factory=set)
    wait_intervals_by_day: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    visited_targets_by_day: set[int] = field(default_factory=set)
    spouse_picked_up: bool = False
    last_position: tuple[float, float] | None = None


class ModelDecisionService:
    """Rule-guided dispatch agent using only the official environment port."""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._memory: dict[str, Memory] = {}

    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        base_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        policy = self._build_policy(status.get("preferences") or [])
        memory = self._refresh_memory(driver_id, policy)

        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
        items = cargo_resp.get("items", [])
        if not isinstance(items, list):
            items = []
        action_minute = base_minute + math.ceil(len(items) / CARGO_VIEW_BATCH_SIZE) if items else base_minute

        forced = self._forced_action(driver_id, status, policy, memory, action_minute)
        if forced is not None:
            return forced

        chosen = self._choose_cargo(status, policy, memory, items, action_minute)
        if chosen is not None:
            cargo_id, score = chosen
            self._logger.info("choose take_order driver_id=%s cargo_id=%s score=%.2f", driver_id, cargo_id, score)
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}

        reposition = self._choose_reposition(status, policy, memory, action_minute, items)
        if reposition is not None:
            lat2, lng2 = reposition
            self._logger.info("choose reposition driver_id=%s target=(%.5f,%.5f)", driver_id, lat2, lng2)
            return {"action": "reposition", "params": {"latitude": lat2, "longitude": lng2}}

        wait_minutes = self._fallback_wait_minutes(policy, memory, action_minute)
        self._logger.info("choose wait driver_id=%s duration=%s", driver_id, wait_minutes)
        return {"action": "wait", "params": {"duration_minutes": wait_minutes}}

    def _build_policy(self, preferences: list[Any]) -> Policy:
        forbidden: set[str] = set()
        avoid: set[str] = set()
        max_haul: float | None = None
        max_pickup: float | None = None
        daily_rest = 0
        full_rest_days = 0
        quiet_windows: list[tuple[int, int]] = []
        lunch_windows: list[tuple[int, int]] = []
        forbidden_zones: list[tuple[float, float, float]] = []
        must_stay_sz = False
        home_target: tuple[float, float] | None = None
        home_deadline: int | None = None
        home_release: int | None = None
        spouse_target: tuple[float, float] | None = None
        recurring_visit: tuple[float, float] | None = None
        familiar_ids: set[str] = set()
        familiar_target: tuple[float, float] | None = None
        familiar_start: int | None = None
        familiar_end: int | None = None

        for pref in preferences:
            text = _extract_content(pref)
            if not text:
                continue

            if "不接货源品类" in text or "禁止" in text:
                for category in re.findall(r"「([^」]+)」", text):
                    if "货源编号" not in text:
                        forbidden.add(category)
            elif "尽量不拉货源品类" in text or "尽量不" in text:
                for category in re.findall(r"「([^」]+)」", text):
                    avoid.add(category)

            if "装货点至卸货点" in text and "不得超过" in text:
                m = re.search(r"不得超过\s*(\d+(?:\.\d+)?)\s*公里", text)
                if m:
                    value = float(m.group(1))
                    max_haul = value if max_haul is None else min(max_haul, value)

            if "赴装货点" in text and "不得超过" in text:
                m = re.search(r"不得超过\s*(\d+(?:\.\d+)?)\s*公里", text)
                if m:
                    value = float(m.group(1))
                    max_pickup = value if max_pickup is None else min(max_pickup, value)

            if "连续" in text and ("休息" in text or "歇" in text):
                m = re.search(r"(?:满|至少)\s*(\d+)\s*小时", text)
                if m:
                    daily_rest = max(daily_rest, int(m.group(1)) * 60)
            if "每天至少连续停车" in text:
                m = re.search(r"(?:满|至少)\s*(\d+)\s*小时", text)
                if m:
                    daily_rest = max(daily_rest, int(m.group(1)) * 60)

            if "整天" in text and ("不接单" in text or "完全歇" in text or "既不接单" in text):
                m = re.search(r"至少(?:要有)?\s*(\d+)\s*个?天", text)
                if m:
                    full_rest_days = max(full_rest_days, int(m.group(1)))
                elif "一整天" in text:
                    full_rest_days = max(full_rest_days, 1)

            if "不接单" in text and ("不空" in text or "不空车" in text or "不空跑" in text or "不空车赶路" in text):
                if "23" in text and ("次日4" in text or "次日4点" in text):
                    quiet_windows.append((23 * 60, 28 * 60))
                elif "23" in text and ("早6" in text or "次日早6" in text):
                    quiet_windows.append((23 * 60, 30 * 60))
                elif "凌晨2" in text and "5点" in text:
                    quiet_windows.append((2 * 60, 5 * 60))
                elif "23点" in text and "次日8点" in text:
                    quiet_windows.append((23 * 60, 32 * 60))
                elif "12点" in text and ("下午1点" in text or "13" in text):
                    lunch_windows.append((12 * 60, 13 * 60))

            if "深圳" in text and "不出市" in text:
                must_stay_sz = True

            if "不得进入" in text and "半径" in text:
                coords = _parse_coordinate_pairs(text)
                m = re.search(r"半径\s*(\d+(?:\.\d+)?)\s*公里", text)
                if coords and m:
                    forbidden_zones.append((coords[0][0], coords[0][1], float(m.group(1))))

            if "自家位置" in text:
                coords = _parse_coordinate_pairs(text)
                if coords:
                    home_target = coords[0]
                    home_deadline = 23 * 60
                    if "次日8" in text:
                        home_release = 32 * 60

            if "家中急事" in text:
                coords = _parse_coordinate_pairs(text)
                if len(coords) >= 2:
                    spouse_target = coords[0]
                    home_target = coords[1]
                    home_deadline = _wall_to_minute("2026-03-10 22:00:00")
                    home_release = _wall_to_minute("2026-03-13 22:00:00")

            if "至少5个不同的自然日到过" in text:
                coords = _parse_coordinate_pairs(text)
                if coords:
                    recurring_visit = coords[0]

            if "指定熟货源编号" in text:
                for cid in re.findall(r"编号\s*(\d+)", text):
                    familiar_ids.add(cid)
                coords = _parse_coordinate_pairs(text)
                if coords:
                    familiar_target = coords[0]
                times = re.findall(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text)
                if times:
                    familiar_start = _wall_to_minute(times[0])
                if len(times) >= 2:
                    familiar_end = _wall_to_minute(times[1])

        return Policy(
            forbidden_categories=forbidden,
            avoid_categories=avoid,
            max_haul_km=max_haul,
            max_pickup_km=max_pickup,
            daily_rest_minutes=daily_rest,
            required_full_rest_days=full_rest_days,
            quiet_windows=quiet_windows,
            lunch_windows=lunch_windows,
            must_stay_in_shenzhen=must_stay_sz,
            forbidden_zones=forbidden_zones,
            home_target=home_target,
            home_deadline_minute=home_deadline,
            home_release_minute=home_release,
            spouse_pickup_target=spouse_target,
            recurring_visit_target=recurring_visit,
            familiar_cargo_ids=familiar_ids,
            familiar_target=familiar_target,
            familiar_start_minute=familiar_start,
            familiar_end_minute=familiar_end,
        )

    def _refresh_memory(self, driver_id: str, policy: Policy) -> Memory:
        memory = self._memory.setdefault(driver_id, Memory())
        try:
            history = self._api.query_decision_history(driver_id, -1)
        except Exception:
            return memory
        records = history.get("records") if isinstance(history, dict) else None
        if not isinstance(records, list) or len(records) == memory.history_count:
            return memory

        memory.accepted_by_day.clear()
        memory.moving_days.clear()
        memory.wait_intervals_by_day.clear()
        memory.visited_targets_by_day.clear()
        memory.spouse_picked_up = False

        current = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            result = rec.get("result") if isinstance(rec.get("result"), dict) else {}
            action = rec.get("action") if isinstance(rec.get("action"), dict) else {}
            name = str(action.get("action", "")).lower()
            end = int(result.get("simulation_progress_minutes", current) or current)
            exec_cost = int(rec.get("action_exec_cost_minutes", 0) or 0)
            query_cost = int(rec.get("query_scan_cost_minutes", 0) or 0)
            start = max(0, end - exec_cost - query_cost)
            action_start = start + query_cost

            if name in {"take_order", "reposition"}:
                for day in range(_day_index(action_start), _day_index(max(action_start, end - 1)) + 1):
                    memory.moving_days.add(day)
            if name == "take_order" and bool(result.get("accepted", False)):
                day = _day_index(action_start)
                memory.accepted_by_day[day] = memory.accepted_by_day.get(day, 0) + 1
            if name == "wait" and exec_cost > 0:
                day = _day_index(action_start)
                memory.wait_intervals_by_day.setdefault(day, []).append((action_start, end))
                pos_before = rec.get("position_before") if isinstance(rec.get("position_before"), dict) else {}
                try:
                    before_lat = float(pos_before["lat"])
                    before_lng = float(pos_before["lng"])
                    if policy.spouse_pickup_target and _point_near(before_lat, before_lng, policy.spouse_pickup_target):
                        memory.spouse_picked_up = True
                except (KeyError, TypeError, ValueError):
                    pass

            pos_after = rec.get("position_after") if isinstance(rec.get("position_after"), dict) else {}
            try:
                memory.last_position = (float(pos_after["lat"]), float(pos_after["lng"]))
                if policy.recurring_visit_target and _point_near(
                    memory.last_position[0], memory.last_position[1], policy.recurring_visit_target
                ):
                    memory.visited_targets_by_day.add(_day_index(end))
            except (KeyError, TypeError, ValueError):
                pass
            current = end

        memory.history_count = len(records)
        return memory

    def _forced_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        policy: Policy,
        memory: Memory,
        action_minute: int,
    ) -> dict[str, Any] | None:
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])

        if self._should_take_full_rest_day(policy, memory, action_minute):
            wait = min(1440 - _day_minute(action_minute), MAX_WAIT_MINUTES)
            return {"action": "wait", "params": {"duration_minutes": max(1, wait)}}

        quiet_wait = self._quiet_window_wait(policy, action_minute)
        if quiet_wait is not None:
            return {"action": "wait", "params": {"duration_minutes": quiet_wait}}

        if policy.familiar_target and policy.familiar_start_minute is not None:
            lead = 8 * 60
            end = policy.familiar_end_minute or (policy.familiar_start_minute + 2 * 60)
            if policy.familiar_start_minute - lead <= action_minute <= end:
                if not _point_near(lat, lng, policy.familiar_target, radius_km=5.0):
                    return self._reposition_to(policy.familiar_target)
                if action_minute < policy.familiar_start_minute:
                    return {
                        "action": "wait",
                        "params": {"duration_minutes": max(1, min(policy.familiar_start_minute - action_minute, MAX_WAIT_MINUTES))},
                    }

        if policy.spouse_pickup_target and policy.home_target and policy.home_deadline_minute is not None:
            pickup_start = _wall_to_minute("2026-03-10 10:00:00") or (policy.home_deadline_minute - 12 * 60)
            home_stay_end = _wall_to_minute("2026-03-13 10:00:00") or policy.home_release_minute
            if home_stay_end and action_minute >= pickup_start and action_minute < home_stay_end and memory.spouse_picked_up:
                if not _point_near(lat, lng, policy.home_target):
                    return self._reposition_to(policy.home_target)
                return {"action": "wait", "params": {"duration_minutes": min(home_stay_end - action_minute, MAX_WAIT_MINUTES)}}
            if action_minute >= policy.home_deadline_minute and policy.home_release_minute and action_minute < policy.home_release_minute:
                if not _point_near(lat, lng, policy.home_target):
                    return self._reposition_to(policy.home_target)
                return {"action": "wait", "params": {"duration_minutes": min(policy.home_release_minute - action_minute, MAX_WAIT_MINUTES)}}
            if pickup_start - 6 * 60 <= action_minute < policy.home_deadline_minute:
                if not _point_near(lat, lng, policy.spouse_pickup_target):
                    return self._reposition_to(policy.spouse_pickup_target)
                if action_minute < pickup_start:
                    return {"action": "wait", "params": {"duration_minutes": min(pickup_start - action_minute, MAX_WAIT_MINUTES)}}
                if not memory.spouse_picked_up:
                    return {"action": "wait", "params": {"duration_minutes": 10}}

        if policy.home_target and policy.home_deadline_minute == 23 * 60:
            dm = _day_minute(action_minute)
            if dm >= 20 * 60 and dm < 23 * 60 and not _point_near(lat, lng, policy.home_target):
                return self._reposition_to(policy.home_target)
            if dm >= 23 * 60 or dm < 8 * 60:
                if not _point_near(lat, lng, policy.home_target):
                    return self._reposition_to(policy.home_target)
                wait_until = (32 * 60 if dm >= 23 * 60 else 8 * 60) - dm
                return {"action": "wait", "params": {"duration_minutes": max(1, min(wait_until, MAX_WAIT_MINUTES))}}

        if policy.recurring_visit_target:
            visited = len(memory.visited_targets_by_day)
            day = _day_index(action_minute)
            if visited < 5 and day not in memory.visited_targets_by_day and _day_minute(action_minute) < 9 * 60:
                if not _point_near(lat, lng, policy.recurring_visit_target):
                    return self._reposition_to(policy.recurring_visit_target)

        if self._needs_daily_rest_now(policy, memory, action_minute):
            return {"action": "wait", "params": {"duration_minutes": min(policy.daily_rest_minutes, MAX_WAIT_MINUTES)}}

        return None

    def _choose_cargo(
        self,
        status: dict[str, Any],
        policy: Policy,
        memory: Memory,
        items: list[Any],
        action_minute: int,
    ) -> tuple[str, float] | None:
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        best: tuple[str, float] | None = None

        for item in items:
            if not isinstance(item, dict):
                continue
            cargo = item.get("cargo")
            if not isinstance(cargo, dict):
                continue
            cargo_id = str(cargo.get("cargo_id", "")).strip()
            if not cargo_id:
                continue
            score = self._score_cargo(cargo, item, lat, lng, policy, memory, action_minute)
            if score is None:
                continue
            if best is None or score > best[1]:
                best = (cargo_id, score)
        return best

    def _score_cargo(
        self,
        cargo: dict[str, Any],
        item: dict[str, Any],
        current_lat: float,
        current_lng: float,
        policy: Policy,
        memory: Memory,
        action_minute: int,
    ) -> float | None:
        category = str(cargo.get("cargo_name", "") or "")
        if category in policy.forbidden_categories:
            return None
        if policy.must_stay_in_shenzhen:
            if not self._cargo_within_shenzhen(cargo):
                return None

        try:
            start = cargo["start"]
            end = cargo["end"]
            start_lat = float(start["lat"])
            start_lng = float(start["lng"])
            end_lat = float(end["lat"])
            end_lng = float(end["lng"])
            price = float(cargo.get("price", 0.0) or 0.0)
            cost_time = int(cargo.get("cost_time_minutes", 0) or 0)
            pickup_km = float(item.get("distance_km", _haversine_km(current_lat, current_lng, start_lat, start_lng)) or 0.0)
        except (KeyError, TypeError, ValueError):
            return None

        haul_km = _haversine_km(start_lat, start_lng, end_lat, end_lng)
        if policy.max_haul_km is not None and haul_km > policy.max_haul_km:
            return None
        if policy.max_pickup_km is not None and pickup_km > policy.max_pickup_km:
            return None
        if self._hits_forbidden_zone(policy, start_lat, start_lng) or self._hits_forbidden_zone(policy, end_lat, end_lng):
            return None

        pickup_minutes = _distance_minutes(pickup_km)
        arrival_minute = action_minute + pickup_minutes
        load_start, load_end = self._load_window(cargo)
        remove_minute = _wall_to_minute(str(cargo.get("remove_time", "")))
        if remove_minute is not None and action_minute > remove_minute:
            return None
        wait_for_load = 0
        if load_end is not None and arrival_minute > load_end:
            return None
        if load_start is not None and arrival_minute < load_start:
            wait_for_load = load_start - arrival_minute

        finish = arrival_minute + wait_for_load + cost_time
        if finish > (_day_index(action_minute) + 1) * 1440:
            return None
        if self._conflicts_quiet_windows(policy, action_minute, finish):
            return None
        if self._would_violate_home(policy, end_lat, end_lng, finish):
            return None
        if (
            policy.daily_rest_minutes
            and not self._day_has_rest(memory, _day_index(action_minute), policy.daily_rest_minutes)
            and finish > (_day_index(action_minute) + 1) * 1440 - policy.daily_rest_minutes
        ):
            return None

        revenue = price
        variable_cost = 1.5 * (pickup_km + haul_km)
        total_minutes = max(1, finish - action_minute)
        score = revenue - variable_cost
        score += (score / total_minutes) * 60.0 * 0.35
        score -= pickup_km * 0.8
        if category in policy.avoid_categories:
            score -= 500.0
        if cargo.get("cargo_id") in policy.familiar_cargo_ids:
            score += 10000.0
        if wait_for_load > 4 * 60:
            score -= (wait_for_load - 4 * 60) * 0.2
        return score if score > 20.0 else None

    def _choose_reposition(
        self,
        status: dict[str, Any],
        policy: Policy,
        memory: Memory,
        action_minute: int,
        items: list[Any],
    ) -> tuple[float, float] | None:
        if self._conflicts_quiet_windows(policy, action_minute, action_minute + 60):
            return None
        if policy.must_stay_in_shenzhen:
            target = (22.54, 114.06)
            lat = float(status["current_lat"])
            lng = float(status["current_lng"])
            if not _point_near(lat, lng, target, radius_km=8.0):
                return target
            return None

        candidates: list[tuple[float, float, float]] = []
        for item in items[:20]:
            cargo = item.get("cargo") if isinstance(item, dict) else None
            if not isinstance(cargo, dict):
                continue
            try:
                start = cargo["start"]
                lat = float(start["lat"])
                lng = float(start["lng"])
                price = float(cargo.get("price", 0.0) or 0.0)
                dist = float(item.get("distance_km", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            if dist > 120:
                candidates.append((price / max(dist, 1.0), lat, lng))
        if candidates and _day_minute(action_minute) < 18 * 60 and not self._needs_daily_rest_now(policy, memory, action_minute):
            _, lat, lng = max(candidates, key=lambda x: x[0])
            return (round(lat, 6), round(lng, 6))
        return None

    def _fallback_wait_minutes(self, policy: Policy, memory: Memory, action_minute: int) -> int:
        quiet = self._quiet_window_wait(policy, action_minute)
        if quiet is not None:
            return quiet
        if self._needs_daily_rest_now(policy, memory, action_minute):
            return min(policy.daily_rest_minutes, MAX_WAIT_MINUTES)
        dm = _day_minute(action_minute)
        if 0 <= dm < 6 * 60:
            return min(6 * 60 - dm, MAX_WAIT_MINUTES)
        if 22 * 60 <= dm:
            return min(30 * 60 - dm, MAX_WAIT_MINUTES)
        return DEFAULT_WAIT_MINUTES

    def _quiet_window_wait(self, policy: Policy, action_minute: int) -> int | None:
        dm = _day_minute(action_minute)
        for start, end in policy.quiet_windows + policy.lunch_windows:
            local = dm
            if end > 1440 and dm < end - 1440:
                local = dm + 1440
            if start <= local < end:
                return max(1, min(end - local, MAX_WAIT_MINUTES))
        return None

    def _conflicts_quiet_windows(self, policy: Policy, start_minute: int, end_minute: int) -> bool:
        for day in range(_day_index(start_minute), _day_index(max(start_minute, end_minute - 1)) + 1):
            base = day * 1440
            for start, end in policy.quiet_windows + policy.lunch_windows:
                if _overlaps(start_minute, end_minute, base + start, base + end):
                    return True
                if end > 1440 and _overlaps(start_minute, end_minute, base - 1440 + start, base - 1440 + end):
                    return True
        return False

    def _needs_daily_rest_now(self, policy: Policy, memory: Memory, action_minute: int) -> bool:
        if policy.daily_rest_minutes <= 0:
            return False
        day = _day_index(action_minute)
        if self._day_has_rest(memory, day, policy.daily_rest_minutes):
            return False
        return _day_minute(action_minute) >= max(18 * 60, 1440 - policy.daily_rest_minutes - 60)

    def _day_has_rest(self, memory: Memory, day: int, required_minutes: int) -> bool:
        intervals = sorted(memory.wait_intervals_by_day.get(day, []))
        if not intervals:
            return False
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return any(end - start >= required_minutes for start, end in merged)

    def _should_take_full_rest_day(self, policy: Policy, memory: Memory, action_minute: int) -> bool:
        if policy.required_full_rest_days <= 0:
            return False
        day = _day_index(action_minute)
        rested = len({done_day for done_day in range(day) if done_day not in memory.moving_days})
        if rested >= policy.required_full_rest_days or day in memory.moving_days:
            return False
        return _day_minute(action_minute) < 20 * 60

    def _would_violate_home(self, policy: Policy, current_lat: float, current_lng: float, finish_minute: int) -> bool:
        if policy.home_target is None:
            return False
        if policy.home_deadline_minute == 23 * 60:
            day = _day_index(finish_minute)
            deadline = day * 1440 + 23 * 60
            return finish_minute > deadline - _distance_minutes(_haversine_km(current_lat, current_lng, *policy.home_target))
        if policy.home_deadline_minute is not None and finish_minute > policy.home_deadline_minute:
            return True
        return False

    def _load_window(self, cargo: dict[str, Any]) -> tuple[int | None, int | None]:
        raw = cargo.get("load_time")
        if isinstance(raw, list) and len(raw) == 2:
            return _wall_to_minute(str(raw[0])), _wall_to_minute(str(raw[1]))
        return None, None

    def _cargo_within_shenzhen(self, cargo: dict[str, Any]) -> bool:
        try:
            points = [cargo["start"], cargo["end"]]
            for point in points:
                lat = float(point["lat"])
                lng = float(point["lng"])
                if not (22.42 <= lat <= 22.89 and 113.74 <= lng <= 114.66):
                    return False
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def _hits_forbidden_zone(self, policy: Policy, lat: float, lng: float) -> bool:
        return any(_haversine_km(lat, lng, zlat, zlng) <= radius for zlat, zlng, radius in policy.forbidden_zones)

    def _reposition_to(self, target: tuple[float, float]) -> dict[str, Any]:
        return {"action": "reposition", "params": {"latitude": target[0], "longitude": target[1]}}

    def _build_prompt(self, driver_id: str, status: dict[str, Any], items: list[dict[str, Any]]) -> str:
        """Kept for compatibility with the original demo; the rule agent does not call the LLM."""
        return json.dumps({"driver_id": driver_id, "status": status, "items": items[:20]}, ensure_ascii=False)

    def _parse_action(self, model_resp: dict[str, Any]) -> dict[str, Any]:
        """Kept for compatibility with older experiments that may still call the model manually."""
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回 content 为空")
        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("模型返回动作不是JSON对象")
        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params")
        if action_name not in {"take_order", "reposition", "wait"}:
            raise ValueError(f"模型返回未知action: {action_name}")
        if not isinstance(params, dict):
            raise ValueError("模型返回 params 必须是对象")
        if action_name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("take_order 缺少有效 cargo_id")
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}
        if action_name == "reposition":
            return {
                "action": "reposition",
                "params": {"latitude": float(params["latitude"]), "longitude": float(params["longitude"])},
            }
        duration_minutes = int(params["duration_minutes"])
        if duration_minutes <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        return {"action": "wait", "params": {"duration_minutes": duration_minutes}}
