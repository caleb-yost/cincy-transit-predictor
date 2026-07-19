"""Deterministic tests for GTFS-realtime parsing (no network — feed built in-memory)."""

from __future__ import annotations

from google.transit import gtfs_realtime_pb2

import ingestion.fetch_realtime as fr


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content


def _trip_update_feed() -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    ent = feed.entity.add()
    ent.id = "1"
    tu = ent.trip_update
    tu.trip.trip_id = "T1"
    tu.trip.route_id = "R1"
    tu.trip.start_date = "20260719"
    tu.vehicle.id = "V1"
    stu = tu.stop_time_update.add()
    stu.stop_id = "S1"
    stu.stop_sequence = 5
    stu.departure.time = 1784439210
    return feed.SerializeToString()


def _vehicle_feed() -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    ent = feed.entity.add()
    ent.id = "v"
    v = ent.vehicle
    v.vehicle.id = "V1"
    v.trip.route_id = "R1"
    v.position.latitude = 39.10
    v.position.longitude = -84.51
    return feed.SerializeToString()


def test_trip_updates_parse(monkeypatch):
    monkeypatch.setattr(fr, "_get", lambda url, **kw: _FakeResp(_trip_update_feed()))
    df = fr.fetch_trip_updates("http://x", 999)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["trip_id"] == "T1"
    assert row["route_id"] == "R1"
    assert row["stop_sequence"] == 5
    assert row["predicted_departure"] == 1784439210
    assert row["snapshot_ts"] == 999


def test_vehicle_positions_parse(monkeypatch):
    monkeypatch.setattr(fr, "_get", lambda url, **kw: _FakeResp(_vehicle_feed()))
    df = fr.fetch_vehicle_positions("http://x", 42)
    assert len(df) == 1
    assert df.iloc[0]["vehicle_id"] == "V1"
    assert round(df.iloc[0]["latitude"], 2) == 39.10
