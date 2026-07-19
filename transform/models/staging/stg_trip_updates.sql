-- Realtime predicted stop times. SORTA populates departure.time far more reliably than
-- arrival.time, so we coalesce the two into a single predicted_time (epoch seconds, UTC).
with src as (
    select * from {{ source('raw', 'trip_updates') }}
)
select
    snapshot_ts,
    to_timestamp(snapshot_ts)                          as snapshot_at,
    trip_id,
    route_id,
    start_date,
    vehicle_id,
    stop_id,
    cast(stop_sequence as integer)                     as stop_sequence,
    coalesce(predicted_arrival, predicted_departure)   as predicted_time,
    cast(schedule_relationship as integer)             as schedule_relationship
from src
where trip_id is not null
  and stop_id is not null
  and coalesce(predicted_arrival, predicted_departure) is not null
