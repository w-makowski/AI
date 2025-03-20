import pandas as pd
from datetime import datetime
import networkx as nx
import utils as ut
import heapq


class PublicTransportGraph:
    def __init__(self, csv_file):
        self.df = self.adjust_csv_file(csv_file)
        self.stops = self.extract_stops()
        self.connections = self.build_connections()

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

    def build_connections(self):
        """Build graph of connections between bus stops"""
        connections = {}

        for _, row in self.df.iterrows():
            start, end = row["start_stop"], row["end_stop"]
            dep_time = ut.time_to_minutes(row["departure_time"])
            arr_time = ut.time_to_minutes(row["arrival_time"])

            if arr_time < dep_time:
                arr_time += 24 * 60

            travel_time = arr_time - dep_time

            connection_info = {
                "line": row["line"],
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

    def get_available_connection(self, start, end, current_time_minutes):
        """Returns closest (by time) connection"""
        if start not in self.stops or end not in self.connections[start]:
            return None, float("inf")

        connections = self.connections[start][end]
        next_connection = None
        min_wait_time = float("inf")

        for conn in connections:
            dep_time = conn["dep_minutes"]

            # Checks if departure is in future and ads 24h if necessary
            if dep_time < current_time_minutes and (current_time_minutes - dep_time) < 1440:
                dep_time += 1440

            wait_time = dep_time - current_time_minutes

            if wait_time < min_wait_time:
                min_wait_time = wait_time
                next_connection = conn

        return next_connection, min_wait_time

    def dijkstra(self, start_stop, end_stop, start_time):
        """Implemented Dijkstra algorithm to finds the best connection (by time) between two stops"""
        start_time_minutes = ut.time_to_minutes(start_time)

        # Initializing
        distances = {stop: float("inf") for stop in self.stops}
        distances[start_stop] = 0
        priority_queue = [(0, start_stop, start_time_minutes, None, None)]
        visited = set()
        path = {}

        while priority_queue:
            current_distance, current_stop, current_time, prev_line, prev_conn = heapq.heappop(priority_queue)
            if current_stop in visited:
                continue
            visited.add(current_stop)
            if current_stop == end_stop:
                break
            if current_stop not in self.connections:
                continue
            for next_stop in self.connections[current_stop]:
                if next_stop in visited:
                    continue
                next_conn, wait_time = self.get_available_connection(current_stop, next_stop, current_time)
                if next_conn is None:
                    continue
                total_time = wait_time + next_conn["travel_time"]

                if current_distance + total_time < distances[next_stop]:
                    # print(f"current time: {current_distance}")
                    distances[next_stop] = current_distance + total_time
                    next_arrival_time = current_time + wait_time + next_conn["travel_time"]
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


def find_fastest_connection(graph, start_stop, end_stop, start_time):
    """Finds fastest connection between two stops and given departure time in format HH:MM:SS using
    PublicTransportGraph's method and prints results"""
    results, distances = graph.dijkstra(start_stop, end_stop, start_time)
    for result in results:
        current_stop, next_stop, connection = result
        print(f"{current_stop} -> {next_stop} ({connection['line']}) departure: {connection['departure_time']} "
              f"arrival: {connection['arrival_time']} travel time: {connection['travel_time']}")
    print(f"total time: {ut.minutes_to_time(distances)}")


def main():
    graph = PublicTransportGraph('DANE - lista 1.csv')

    exit_flag = False

    while not exit_flag:
        try:
            start_stop = input("Enter departure stop:")
            end_stop = input("Enter arrival stop:")
            start_time = input("Enter departure time (in format HH:MM:SS):")
            condition = input("Enter condition by which the connection will be optimized (t for time, c for changes)")
            if condition == 't':
                find_fastest_connection(graph, start_stop, end_stop, start_time)
            elif condition == 'c':
                pass
            else:
                break
        except KeyboardInterrupt:
            print("Exiting program...")
            exit_flag = True


if __name__ == "__main__":
    main()
