#!/usr/bin/env python3
"""Pipeline API - Fetches the next SpaceX launch"""

import requests
from datetime import datetime

if __name__ == '__main__':
    """Fetches the next upcoming launch"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error: Unable to fetch launch data")
        exit(1)

    launches = response.json()
    if not launches:
        print("No upcoming launches found.")
        exit(1)

    # Find the earliest upcoming launch
    next_launch = min(launches, key=lambda x: x["date_unix"])

    # Extract launch details
    launch_name = next_launch["name"]
    date = next_launch["date_local"]

    # Fetch rocket details
    rocket_url = "https://api.spacexdata.com/v4/rockets/" + next_launch["rocket"]
    rocket_response = requests.get(rocket_url)
    if rocket_response.status_code != 200:
        print("Error fetching rocket data")
        exit(1)
    rocket_name = rocket_response.json().get("name", "Unknown Rocket")

    # Fetch launchpad details
    launchpad_url = "https://api.spacexdata.com/v4/launchpads/" + next_launch["launchpad"]
    launchpad_response = requests.get(launchpad_url)
    if launchpad_response.status_code != 200:
        print("Error fetching launchpad data")
        exit(1)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data.get("name", "Unknown Launchpad")
    launchpad_locality = launchpad_data.get("locality", "Unknown Location")

    # Print result in expected format
    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name, launchpad_name, launchpad_locality))
