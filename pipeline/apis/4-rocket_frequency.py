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
        print("Error: Unable to fetch launch data")
        return []  # Return an empty list to prevent crashes

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
        print("Error: Unable to fetch rocket data")
        return {}  # Return an empty dictionary to prevent crashes

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
    # Ensure we only use known rocket names
    valid_counts = {rocket_names.get(rid, "Unknown Rocket"): count for rid, count in rocket_launch_count.items() if rid in rocket_names}

    # Sort by launch count (descending), then by name (ascending)
    sorted_rockets = sorted(valid_counts.items(), key=lambda item: (-item[1], item[0]))

    # Print results
    for name, count in sorted_rockets:
        print("{}: {}".format(name, count))

if __name__ == '__main__':
    """
    Main execution block.
    Fetches launches, counts them per rocket, and displays the sorted results.
    """
    launches = fetch_launches()
    rocket_names = fetch_rockets()

    # Debug: Check if API data was fetched properly
    if not launches:
        print("Error: No launch data retrieved.")
    elif not rocket_names:
        print("Error: No rocket data retrieved.")
    else:
        rocket_launch_count = count_launches(launches)
        display_launch_counts(rocket_launch_count, rocket_names)
