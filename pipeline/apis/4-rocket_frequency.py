#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket using the SpaceX API.
Results are sorted by launch count (descending) and rocket name
(alphabetically).
"""
import requests
from collections import defaultdict


def get_rocket_launches():
    """
    Fetches and returns the frequency of launches per rocket from SpaceX API.

    Returns:
        list: List of tuples containing (rocket_name, launch_count),
              sorted by count (descending) and name (alphabetically)
    """
    try:
        # Get all launches (past and upcoming)
        past_launches = requests.get(
            "https://api.spacexdata.com/v4/launches/past").json()
        upcoming_launches = requests.get(
            "https://api.spacexdata.com/v4/launches/upcoming").json()

        # Combine all launches
        all_launches = past_launches + upcoming_launches

        # Get all rocket IDs and their frequency
        rocket_freq = defaultdict(int)
        rocket_ids = set()

        for launch in all_launches:
            rocket_id = launch.get('rocket')
            if rocket_id:
                rocket_freq[rocket_id] += 1
                rocket_ids.add(rocket_id)

        # Get rocket names
        rocket_names = {}
        for rocket_id in rocket_ids:
            response = requests.get(
                "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id))
            rocket_data = response.json()
            rocket_names[rocket_id] = rocket_data.get('name')

        # Create list of (name, count) tuples
        launch_data = [(rocket_names.get(rid, 'Unknown'), count)
                       for rid, count in rocket_freq.items()]

        # Sort by count (descending) and name (alphabetically)
        launch_data.sort(key=lambda x: (-x[1], x[0]))

        return launch_data

    except requests.exceptions.RequestException as e:
        print("Error fetching launch data: {}".format(str(e)))
        return []
    except (KeyError, ValueError) as e:
        print("Error processing launch data: {}".format(str(e)))
        return []


if __name__ == '__main__':
    rocket_launches = get_rocket_launches()
    for rocket_name, launch_count in rocket_launches:
        print("{}: {}".format(rocket_name, launch_count))
        