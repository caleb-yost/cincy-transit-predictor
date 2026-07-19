select
    snapshot_ts,
    to_timestamp(snapshot_ts)          as snapshot_at,
    cast(temperature_f as double)      as temperature_f,
    cast(precipitation_in as double)   as precipitation_in,
    cast(rain_in as double)            as rain_in,
    cast(snowfall_in as double)        as snowfall_in,
    cast(wind_speed_mph as double)     as wind_speed_mph,
    cast(weather_code as integer)      as weather_code
from {{ source('raw', 'weather') }}
