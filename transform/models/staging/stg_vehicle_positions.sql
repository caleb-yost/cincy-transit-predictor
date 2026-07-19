select
    snapshot_ts,
    to_timestamp(snapshot_ts)              as snapshot_at,
    vehicle_id,
    trip_id,
    route_id,
    cast(latitude as double)               as latitude,
    cast(longitude as double)              as longitude,
    cast(bearing as double)                as bearing,
    cast(speed as double)                  as speed,
    cast(current_stop_sequence as integer) as current_stop_sequence,
    cast(current_status as integer)        as current_status,
    vehicle_ts
from {{ source('raw', 'vehicle_positions') }}
where latitude is not null and longitude is not null
