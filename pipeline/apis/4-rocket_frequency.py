#!/usr/bin/env python3
"""
Fetches SpaceX launch data and displays the number of launches per rocket.

- Counts all launches
- Sorts by number of launches (descending)
- If two rockets have the same count, sorts alphabetically (A-Z)
"""

import requests

def fetch_launches():
    """
    Fetch SpaceX launches from the API.
    
    Returns:
        list: List of launch data (JSON objects) or empty list on failure.
    """
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error: Unable to fetch launch data")
        return []
    
    return response.json()  # Convert response to Python list

def fetch_rockets():
    """
    Fetch SpaceX rocket details from the API.
    
    Returns:
        dict: Mapping of rocket IDs to their names.
    """
    url = "https://api.spacexdata.com/v4/rockets"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error: Unable to fetch rocket data")
        return {}
    
    rockets_data = response.json()
    return {rocket["id"]: rocket["name"] for rocket in rockets_data}  # Dictionary of {id: name}

def count_launches(launches):
    """
    Count how many times each rocket was used for a launch.

    Args:
        launches (list): List of launch JSON objects.

    Returns:
        dict: Mapping of rocket IDs to launch counts.
    """
    rocket_launch_count = {}

    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            if rocket_id in rocket_launch_count:
                rocket_launch_count[rocket_id] += 1
            else:
                rocket_launch_count[rocket_id] = 1

    return rocket_launch_count

def display_launch_counts(rocket_launch_count, rocket_names):
    """
    Print the number of launches per rocket in the required format.

    Args:
        rocket_launch_count (dict): Rocket ID -> Number of launches.
        rocket_names (dict): Rocket ID -> Rocket Name.
    """
    valid_counts = {
        rocket_names.get(rid, "Unknown Rocket"): count
        for rid, count in rocket_launch_count.items()
        if rid in rocket_names
    }

    # Sort by launch count (descending), then by name (ascending)
    sorted_rockets = sorted(valid_counts.items(), key=lambda item: (-item[1], item[0]))

    # Print results in "Rocket Name: Count" format
    for name, count in sorted_rockets:
        print("{}: {}".format(name, count))

if __name__ == '__main__':
    """
    Main execution block.
    Fetches launches, counts them per rocket, and displays the sorted results.
    """
    launches = fetch_launches()
    rocket_names = fetch_rockets()

    if not launches:
        print("Error: No launch data retrieved.")
    elif not rocket_names:
        print("Error: No rocket data retrieved.")
    else:
        rocket_launch_count = count_launches(launches)
        display_launch_counts(rocket_launch_count, rocket_names)
