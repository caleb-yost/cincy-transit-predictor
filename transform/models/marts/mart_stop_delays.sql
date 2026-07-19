-- The labeled dataset for ML + reliability: one row per scheduled stop-arrival, taking the
-- LAST prediction we observed for it (a bus stops being predicted once it passes, so the final
-- prediction is a proxy for the actual arrival). Weather from the same snapshot is joined on.
with ranked as (
    select
        *,
        row_number() over (
            partition by start_date, trip_id, stop_id, stop_sequence
            order by snapshot_ts desc
        ) as rn
    from {{ ref('fct_arrivals') }}
),
final_pred as (
    select * from ranked where rn = 1
),
wx as (
    select
        snapshot_ts,
        avg(temperature_f)    as temperature_f,
        avg(precipitation_in) as precipitation_in,
        avg(snowfall_in)      as snowfall_in,
        avg(wind_speed_mph)   as wind_speed_mph,
        max(weather_code)     as weather_code
    from {{ ref('stg_weather') }}
    group by snapshot_ts
)
select
    md5(concat_ws('|', fp.start_date, fp.trip_id, fp.stop_id, fp.stop_sequence)) as stop_delay_key,
    fp.start_date,
    fp.trip_id,
    fp.route_id,
    fp.stop_id,
    fp.stop_sequence,
    fp.vehicle_id,
    fp.scheduled_at,
    fp.predicted_at,
    fp.delay_seconds,
    fp.delay_minutes,
    -- local-time features
    extract(hour from timezone('America/New_York', fp.scheduled_at))         as sched_hour,
    extract(dow  from timezone('America/New_York', fp.scheduled_at))         as sched_dow,
    (extract(dow from timezone('America/New_York', fp.scheduled_at)) in (0, 6)) as is_weekend,
    case
        when fp.delay_seconds between {{ var('on_time_early_secs') }} and {{ var('on_time_late_secs') }}
        then 1 else 0
    end as is_on_time,
    case when fp.delay_seconds > {{ var('on_time_late_secs') }} then 1 else 0 end as is_late,
    wx.temperature_f,
    wx.precipitation_in,
    wx.snowfall_in,
    wx.wind_speed_mph,
    wx.weather_code
from final_pred fp
left join wx on fp.snapshot_ts = wx.snapshot_ts
