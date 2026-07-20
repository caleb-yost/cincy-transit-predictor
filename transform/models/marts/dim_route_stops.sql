-- Distinct stops served by each route, with a representative order and coordinates.
-- Powers the predictor's stop dropdown (pick a real stop by name) and the map pin. A route runs
-- many trip patterns, so we take the min stop_sequence per (route, stop) as a representative order.
with route_stop as (
    select distinct
        t.route_id,
        st.stop_id,
        st.stop_sequence
    from {{ ref('stg_trips') }} t
    join {{ ref('stg_stop_times') }} st on t.trip_id = st.trip_id
),
ordered as (
    select
        route_id,
        stop_id,
        min(stop_sequence) as stop_sequence
    from route_stop
    group by route_id, stop_id
)
select
    md5(concat_ws('|', o.route_id, o.stop_id)) as route_stop_key,
    o.route_id,
    dr.route_short_name,
    o.stop_id,
    ds.stop_name,
    ds.stop_lat,
    ds.stop_lon,
    o.stop_sequence
from ordered o
left join {{ ref('dim_route') }} dr on o.route_id = dr.route_id
left join {{ ref('dim_stop') }} ds on o.stop_id = ds.stop_id
