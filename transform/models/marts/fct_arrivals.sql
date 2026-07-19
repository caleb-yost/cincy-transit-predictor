-- One row per (snapshot, trip, stop): the predicted stop time joined to the schedule,
-- with delay = predicted - scheduled.
--
-- SORTA's realtime `start_date` is unreliable for after-midnight trips: it reports the
-- physical calendar date (e.g. 20260719) while the schedule still encodes those stops with
-- >24:00 clock times (e.g. 24:55:00), so naively combining them double-counts the day
-- rollover (a flat +24h error). Instead we take the schedule's clock time-of-day and anchor
-- it to the calendar day (-1/0/+1) whose instant is closest to the observed prediction. A bus
-- is never more than a few hours off schedule, so the nearest occurrence is unambiguous.
with rt as (
    select * from {{ ref('stg_trip_updates') }}
),
sched as (
    select trip_id, stop_id, stop_sequence, scheduled_secs
    from {{ ref('stg_stop_times') }}
),
joined as (
    select
        rt.snapshot_ts,
        rt.snapshot_at,
        rt.trip_id,
        rt.route_id,
        rt.stop_id,
        rt.stop_sequence,
        rt.vehicle_id,
        rt.start_date,
        rt.predicted_time,
        (sched.scheduled_secs % 86400) as sched_tod_secs,
        timezone('America/New_York', to_timestamp(rt.predicted_time))::date as pred_local_date
    from rt
    join sched
        on  rt.trip_id = sched.trip_id
        and rt.stop_id = sched.stop_id
        and rt.stop_sequence = sched.stop_sequence
    where sched.scheduled_secs is not null
),
candidates as (
    select j.*, unnest([-1, 0, 1]) as day_off
    from joined j
),
scored as (
    select
        *,
        epoch(
            timezone(
                'America/New_York',
                (pred_local_date + day_off)::timestamp + to_seconds(sched_tod_secs)
            )
        ) as cand_scheduled_epoch
    from candidates
),
best as (
    select
        *,
        row_number() over (
            partition by snapshot_ts, trip_id, stop_id, stop_sequence
            order by abs(predicted_time - cand_scheduled_epoch)
        ) as rn
    from scored
)
select
    md5(concat_ws('|', snapshot_ts, trip_id, stop_id, stop_sequence)) as arrival_key,
    snapshot_ts,
    snapshot_at,
    trip_id,
    route_id,
    stop_id,
    stop_sequence,
    vehicle_id,
    start_date,
    to_timestamp(cand_scheduled_epoch)                       as scheduled_at,
    to_timestamp(predicted_time)                             as predicted_at,
    (predicted_time - cand_scheduled_epoch)                  as delay_seconds,
    round((predicted_time - cand_scheduled_epoch) / 60.0, 2) as delay_minutes
from best
where rn = 1
