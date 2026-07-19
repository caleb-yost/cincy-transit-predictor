select
    route_id,
    route_short_name,
    route_long_name,
    try_cast(route_type as integer) as route_type
from {{ ref('stg_routes') }}
