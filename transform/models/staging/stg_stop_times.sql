-- Scheduled stop times. GTFS times can exceed 24:00:00 (trips after midnight), so we parse
-- "HH:MM:SS" into seconds-after-service-midnight rather than casting to TIME.
with src as (
    select * from {{ source('reference', 'stop_times') }}
)
select
    trip_id,
    stop_id,
    cast(stop_sequence as integer) as stop_sequence,
    arrival_time                   as scheduled_arrival_str,
    departure_time                 as scheduled_departure_str,
    coalesce(
        try_cast(split_part(arrival_time, ':', 1) as integer) * 3600
        + try_cast(split_part(arrival_time, ':', 2) as integer) * 60
        + try_cast(split_part(arrival_time, ':', 3) as integer),
        try_cast(split_part(departure_time, ':', 1) as integer) * 3600
        + try_cast(split_part(departure_time, ':', 2) as integer) * 60
        + try_cast(split_part(departure_time, ':', 3) as integer)
    ) as scheduled_secs
from src
