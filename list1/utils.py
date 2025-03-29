import math
from typing import Dict


def time_to_minutes(time_str):
    """Converts time in HH:MM:SS format to minutes after midnight"""
    h, m, s = map(int, time_str.split(':'))
    return h * 60 + m + s / 60


def minutes_to_time(minutes):
    """Converts minutes from midnight to time in HH:MM:SS format"""
    minutes = minutes % (24 * 60)
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance (in km) between two points on Earth"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    r = 6371
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    hav_theta = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(hav_theta))


def print_path(path):
    prev_end_stop, prev_connection = None, None
    for current_stop, next_stop, connection in path:
        if prev_end_stop is None and prev_connection is None:
            print(f"Take {connection['line']} at {current_stop} stop at {connection['departure_time']}")
        elif prev_connection["line"] != connection["line"]:
            print(f"Get off {prev_connection['line']} at {prev_end_stop} stop at {prev_connection['arrival_time']}")
            print(f"Take {connection['line']} at {current_stop} stop at {connection['departure_time']}")
        prev_end_stop = next_stop
        prev_connection = connection
    print(f"Get off {prev_connection['line']} at {prev_end_stop} stop at {prev_connection['arrival_time']}")

def print_route(route: Dict):
    """Prints details of the route"""
    if not route:
        print("Path not found.")
        return

    prev_end_stop, prev_line, prev_arrival_time = None, None, None
    for step in route['path']:
        if prev_end_stop is None and prev_line is None:
            print(f"Take {step[2]} at {step[0]} stop at {step[3]}")
        elif prev_line != step[2]:
            print(f"Get off {prev_line} at {prev_end_stop} stop at {prev_arrival_time}")
            print(f"Take {step[2]} at {step[0]} stop at {step[3]}")
        prev_end_stop = step[1]
        prev_line = step[2]
        prev_arrival_time = step[4]
    print(f"Get off {prev_line} at {prev_end_stop} stop at {prev_arrival_time}")

    print(f"Total path time: {route['total_time']} minutes")
    print(f"Total changes: {route['transfers']}")
    # for step in route['path']:
    #     print(f"{step[0]} → {step[1]} (Linia: {step[2]}, Odjazd: {step[3]}, Przyjazd: {step[4]})")
