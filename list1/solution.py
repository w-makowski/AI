import heapq
import time
import math

import pandas as pd

import utils as ut

AVG_TRANSPORT_SPEED = 20    # Average speed of public transport (in km/h) used to estimate travel time
TRANSFER_PENALTY = 30   # Penalty for line changes (in min)


class PublicTransportGraph:
    def __init__(self, csv_file):
        self.df = self.adjust_csv_file(csv_file)
        self.stops = self.extract_stops()
        self.connections = self.build_connections()
        self.stop_coordinates = self.extract_stop_coordinates()

    def adjust_csv_file(self, csv_file):
        df = pd.read_csv(csv_file)

        def time_adjustment(time_str):
            return str(int(time_str[:2]) % 24) + time_str[2:]

        df["departure_time"] = df["departure_time"].apply(time_adjustment)
        df["arrival_time"] = df["arrival_time"].apply(time_adjustment)
        return df

    def extract_stops(self):
        """Extracts stops from csv file"""
        start_stops = set(self.df['start_stop'])
        end_stops = set(self.df['end_stop'])
        return list(start_stops.union(end_stops))

    def extract_stop_coordinates(self):
        """Extracts stops coordinates"""
        stop_coords = {}
        for _, row in self.df.iterrows():
            stop_coords[row["start_stop"]] = (row["start_stop_lat"], row["start_stop_lon"])
            stop_coords[row["end_stop"]] = (row["end_stop_lat"], row["end_stop_lon"])
        return stop_coords

    def build_connections(self):
        """Builds graph of connections between bus stops"""
        connections = {}

        for _, row in self.df.iterrows():
            start, end = row["start_stop"], row["end_stop"]
            dep_time = ut.time_to_minutes(row["departure_time"])
            arr_time = ut.time_to_minutes(row["arrival_time"])

            if arr_time < dep_time:
                arr_time += 24 * 60

            travel_time = arr_time - dep_time

            connection_info = {
                "line": str(row["line"]),
                "company": row["company"],
                "departure_time": row["departure_time"],
                "arrival_time": row["arrival_time"],
                "travel_time": travel_time,
                "dep_minutes": dep_time,
                "arr_minutes": arr_time
            }

            if start not in connections:
                connections[start] = {}
            if end not in connections[start]:
                connections[start][end] = []
            connections[start][end].append(connection_info)

        return connections

    def get_best_connection(self, start, end, current_time_minutes, current_line, change_penalty=0.01):
        """Returns the best connection in terms of travel time + waiting time between two stops"""
        if start not in self.stops or end not in self.connections[start]:
            return None, float("inf"), float("inf")

        connections = self.connections[start][end]
        best_connection = None
        min_total_time = float("inf")
        min_wait_time = float("inf")
        penalty = 0

        for conn in connections:
            dep_time = conn["dep_minutes"]

            # Checks if departure is in future and ads 24h if necessary
            if dep_time < current_time_minutes and (current_time_minutes - dep_time) < 1440:
                dep_time += 1440

            tmp_penalty = change_penalty if current_line != conn["line"] else 0
            wait_time = dep_time - current_time_minutes
            total_time = wait_time + conn["travel_time"] + tmp_penalty

            if total_time < min_total_time:
                min_total_time = total_time
                penalty = tmp_penalty
                min_wait_time = wait_time
                best_connection = conn

        return best_connection, min_total_time - penalty, min_wait_time

    def dijkstra(self, start_stop, end_stop, start_time_minutes):
        """Dijkstra algorithm to finds the best connection (by time) between two stops"""

        # Initializing
        distances = {stop: float("inf") for stop in self.stops}
        distances[start_stop] = 0
        priority_queue = [(0, start_stop, start_time_minutes, None, None)]
        visited = set()
        path = {}

        while priority_queue:
            current_distance, current_stop, current_time, prev_line, prev_conn = heapq.heappop(priority_queue)
            if current_stop in visited and current_distance > distances[current_stop]:
                continue
            visited.add(current_stop)
            if current_stop == end_stop:
                break
            if current_stop not in self.connections:
                continue
            for next_stop in self.connections[current_stop]:
                if next_stop in visited:
                    continue
                next_conn, total_time, wait_time = self.get_best_connection(current_stop, next_stop, current_time, prev_line)
                if next_conn is None or total_time == float("inf"):
                    continue

                if current_distance + total_time < distances[next_stop]:
                    distances[next_stop] = current_distance + total_time
                    next_arrival_time = current_time + wait_time + next_conn["travel_time"]
                    next_arrival_time %= 1440
                    path[next_stop] = (current_stop, next_conn, current_time)
                    heapq.heappush(priority_queue,
                                   (distances[next_stop], next_stop, next_arrival_time, next_conn["line"], next_conn))

        if end_stop not in path:
            return None, float("inf")

        result = []
        current = end_stop

        while current != start_stop:
            prev_stop, connection, prev_time = path[current]
            result.append((prev_stop, current, connection))
            current = prev_stop

        result.reverse()
        return result, distances[end_stop]

    def a_star_time(self, start_stop, end_stop, start_time_minutes):
        """A* algorithm to finds shortest connection between two stops with time optimization"""

        g_score = {stop: float("inf") for stop in self.stops}
        g_score[start_stop] = 0

        f_score = {stop: float("inf") for stop in self.stops}
        f_score[start_stop] = self.estimate_time(start_stop, end_stop)

        open_set = [(f_score[start_stop], start_stop, start_time_minutes, None, None)]
        closed_set = set()
        path = {}

        while open_set:
            _, current_stop, current_time, prev_line, prev_conn = heapq.heappop(open_set)

            if current_stop in closed_set:
                continue
            if current_stop == end_stop:
                break

            closed_set.add(current_stop)

            if current_stop not in self.connections:
                continue

            for next_stop in self.connections[current_stop]:
                if next_stop in closed_set:
                    continue

                next_conn, total_time, wait_time = self.get_best_connection(current_stop, next_stop, current_time, prev_line)

                if next_conn is None:
                    continue

                tentative_g_score = g_score[current_stop] + total_time

                if tentative_g_score < g_score[next_stop]:
                    next_arrival_time = current_time + wait_time + next_conn["travel_time"]
                    path[next_stop] = (current_stop, next_conn, current_time)
                    g_score[next_stop] = tentative_g_score
                    f_score[next_stop] = g_score[next_stop] + self.estimate_time(next_stop, end_stop)
                    heapq.heappush(open_set,
                                   (f_score[next_stop], next_stop, next_arrival_time, next_conn["line"], next_conn))

        if end_stop not in path:
            return None, float("inf")

        result = []
        current = end_stop

        while current != start_stop:
            prev_stop, connection, prev_time = path[current]
            result.append((prev_stop, current, connection))
            current = prev_stop

        result.reverse()
        return result, g_score[end_stop]

    def estimate_time(self, stop1, stop2):
        """Heuristic function for A* to estimate time needed to get from stop1 to stop2. Returns time in minutes"""
        if stop1 not in self.stop_coordinates or stop2 not in self.stop_coordinates:
            return 0

        lat1, lon1 = self.stop_coordinates[stop1]
        lat2, lon2 = self.stop_coordinates[stop2]

        distance = self.haversine_distance(lat1, lon1, lat2, lon2)
        estimated_time = (distance / AVG_TRANSPORT_SPEED) * 60
        return estimated_time

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance (in km) between two points on Earth"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        r = 6371
        d_lon = lon2 - lon1
        d_lat = lat2 - lat1
        hav_theta = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        return 2 * r * math.asin(math.sqrt(hav_theta))

    def a_star_transfers(self, start_stop, end_stop, start_time):
        """A* algorithm to finds shortest connection between two stops with optimalization of changes (and subsequently - time)"""
        open_set = []
        heapq.heappush(open_set, (0, start_stop, start_time, start_time, [], None))

        g_score = {(start_stop, None): 0}
        closed_set = set()

        heuristic_cache = {}

        def get_heuristic(stop):
            if stop in heuristic_cache:
                return heuristic_cache[stop]
            else:
                h_val = (self.haversine_distance(*self.stop_coordinates[stop], *self.stop_coordinates[
                    end_stop]) / AVG_TRANSPORT_SPEED) * 60
                heuristic_cache[stop] = h_val
                return h_val

        while open_set:
            current_f, current_stop, current_cost, current_time, current_path, prev_line = heapq.heappop(open_set)

            if current_stop == end_stop:
                return {
                    'path': current_path,
                    'total_time': current_time - start_time if current_time > start_time else 1440 - start_time + current_time,
                    'transfers': len(set(path[2] for path in current_path)) - 1
                }

            key = (current_stop, prev_line)
            if key in closed_set:
                continue

            closed_set.add(key)

            if current_stop not in self.connections.keys():
                continue

            for next_stop, connections in self.connections[current_stop].items():
                for connection in connections:
                    if (connection['dep_minutes'] < current_time and
                            connection['dep_minutes'] + 1440 < current_time):
                        continue

                    dep_time = connection['dep_minutes']
                    if dep_time < current_time:
                        dep_time += 1440

                    waiting_time = dep_time - current_time
                    travel_time = connection['travel_time']
                    line = connection['line']

                    t_penalty = TRANSFER_PENALTY if current_time > 240 and current_time < 1400 else TRANSFER_PENALTY * 5

                    transfer_penalty = t_penalty if prev_line and prev_line != line else 0

                    g_cost = current_cost + waiting_time + travel_time + transfer_penalty
                    if (next_stop, line) not in g_score or g_cost < g_score[(next_stop, line)]:
                        new_time = connection['arr_minutes']
                        g_score[(next_stop, line)] = g_cost

                        h_cost = get_heuristic(next_stop)

                        f_cost = g_cost + h_cost

                        new_path = current_path + [
                            (current_stop, next_stop, line, connection['departure_time'], connection['arrival_time'])]

                        heapq.heappush(open_set, (f_cost, next_stop, g_cost, new_time, new_path, line))

        return None


def find_fastest_connection(graph, start_stop, end_stop, start_time):
    """Finds fastest connection between two stops and given departure time in format HH:MM:SS using
    PublicTransportGraph's method and prints results"""
    start_time_minutes = ut.time_to_minutes(start_time)
    st1 = time.time()
    results, distances = graph.dijkstra(start_stop, end_stop, start_time_minutes)
    et1 = time.time()
    print(f"Dijkstra algorithm {et1 - st1}:")
    ut.print_path(results)
    print(f"total time: {ut.minutes_to_time(distances)}")

    st2 = time.time()
    res, g_score = graph.a_star_time(start_stop, end_stop, start_time_minutes)
    et2 = time.time()
    print(f"\nA* algorithm {et2 - st2}:")
    ut.print_path(res)
    print(f"total time: {g_score}")


def find_connection_with_least_changes(graph, start_stop, end_stop, start_time):
    """Finds best connection in terms of the least changes between two stops and given departure time in format HH:MM:SS
     using PublicTransportGraph's method and prints results"""
    start_time_minutes = ut.time_to_minutes(start_time)
    st1 = time.time()
    results = graph.a_star_transfers(start_stop, end_stop, start_time_minutes)
    et1 = time.time()
    print(f"A* transfers ({et1 - st1})")
    ut.print_route(results)


def main():
    graph = PublicTransportGraph('DANE - lista 1.csv')

    exit_flag = False

    while not exit_flag:
        try:
            start_stop = input("Enter departure stop:")
            end_stop = input("Enter arrival stop:")
            start_time = input("Enter departure time (in format HH:MM:SS):")
            condition = input("Optimize by time (t) or changes (c)?")
            if condition == 't':
                find_fastest_connection(graph, start_stop, end_stop, start_time)
            elif condition == 'c':
                find_connection_with_least_changes(graph, start_stop, end_stop, start_time)
            else:
                break
        except KeyboardInterrupt:
            print("Exiting program...")
            exit_flag = True


if __name__ == "__main__":
    main()
