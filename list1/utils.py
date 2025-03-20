from datetime import datetime


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


def convert_time(time_str):
    """Converts time in HH:MM:SS format with hours exceeding 23"""
    time_str = str(int(time_str[:2]) % 24) + time_str[2:]
    return datetime.strptime(time_str, "%H:%M:%S").time()
