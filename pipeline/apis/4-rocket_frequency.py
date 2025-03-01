#!/usr/bin/env python3
"""Displays the number of launches per rocket using the SpaceX API"""
import requests

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error: Unable to fetch launch data")
        exit(1)

    launches = response.json()
    rocket_launch_count = {}

    # Count launches per rocket ID
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_launch_count[rocket_id] = rocket_launch_count.get(rocket_id, 0) + 1

    # Fetch rocket names
    url_rockets = "https://api.spacexdata.com/v4/rockets"
    rockets_response = requests.get(url_rockets)

    if rockets_response.status_code != 200:
        print("Error: Unable to fetch rocket data")
        exit(1)

    rockets_data = rockets_response.json()
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets_data}

    # Prepare sorted list
    sorted_rockets = sorted(
        rocket_launch_count.items(),
        key=lambda item: (-item[1], rocket_names.get(item[0], ""))  # Sort by launches desc, then name asc
    )

    # Print results
    for rocket_id, count in sorted_rockets:
        print(f"{rocket_names.get(rocket_id, 'Unknown Rocket')}: {count}")
