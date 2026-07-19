-- Rolling reliability by route x scheduled-hour x day-of-week. Powers the dashboard leaderboard.
select
    d.route_id,
    dr.route_short_name,
    dr.route_long_name,
    d.sched_hour,
    d.sched_dow,
    count(*)                            as n_arrivals,
    round(avg(d.delay_minutes), 2)      as avg_delay_minutes,
    round(median(d.delay_minutes), 2)   as median_delay_minutes,
    round(100.0 * avg(d.is_on_time), 1) as on_time_pct,
    round(max(d.delay_minutes), 2)      as worst_delay_minutes
from {{ ref('mart_stop_delays') }} d
left join {{ ref('dim_route') }} dr on d.route_id = dr.route_id
group by 1, 2, 3, 4, 5
