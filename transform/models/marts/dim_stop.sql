select
    stop_id,
    stop_name,
    try_cast(stop_lat as double) as stop_lat,
    try_cast(stop_lon as double) as stop_lon
from {{ ref('stg_stops') }}
