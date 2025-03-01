#!/usr/bin/env python3
"""
This script fetches SpaceX launch data from the public API and displays 
the number of launches per rocket.

- All launches are counted.
- Rockets are sorted by number of launches (descending).
- If two rockets have the same count, they are sorted alphabetically.
"""

import requests

def fetch_launches():
    """
    Fetch all SpaceX launches from the API.
    Returns:
        list: A list of launch data (JSON objects).
    """
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Error: Unable to fetch launch data")

    return response.json()

def fetch_rockets():
    """
    Fetch all SpaceX rockets from the API.
    Returns:
        dict: A dictionary mapping rocket IDs to their names.
    """
    url = "https://api.spacexdata.com/v4/rockets"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Error: Unable to fetch rocket data")

    rockets_data = response.json()
    return {rocket["id"]: rocket["name"] for rocket in rockets_data}

def count_launches(launches):
    """
    Count the number of launches for each rocket.

    Args:
        launches (list): A list of launch JSON objects.

    Returns:
        dict: A dictionary mapping rocket IDs to launch counts.
    """
    rocket_launch_count = {}

    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_launch_count[rocket_id] = rocket_launch_count.get(rocket_id, 0) + 1

    return rocket_launch_count

def display_launch_counts(rocket_launch_count, rocket_names):
    """
    Print the number of launches per rocket in the required format.

    Args:
        rocket_launch_count (dict): Rocket ID -> Number of launches
        rocket_names (dict): Rocket ID -> Rocket Name
    """
    sorted_rockets = sorted(
        rocket_launch_count.items(),
        key=lambda item: (-item[1], rocket_names.get(item[0], ""))
    )

    for rocket_id, count in sorted_rockets:
        print("{}: {}".format(rocket_names.get(rocket_id, "Unknown Rocket"), count))

if __name__ == '__main__':
    """
    Main execution block.
    Fetches launches, counts them per rocket, and displays the sorted results.
    """
    try:
        launches = fetch_launches()
        rocket_names = fetch_rockets()
        rocket_launch_count = count_launches(launches)
        display_launch_counts(rocket_launch_count, rocket_names)
    except Exception as e:
        print(e)
