#!/usr/bin/env python3
"""
Script that displays information about the next upcoming SpaceX launch
using the SpaceX API.
"""
import requests
from datetime import datetime
import time
from datetime import timezone, timedelta


def get_upcoming_launch():
    """
    Fetches and returns formatted information about the next upcoming SpaceX
    launch.
    Returns:
        str: Formatted string containing launch information
    """
    api_url = "https://api.spacexdata.com/v4/launches/upcoming"
    try:
        # Get upcoming launches
        response = requests.get(api_url)
        response.raise_for_status()
        launches = response.json()
        if not launches:
            return "No upcoming launches found"

        # Sort launches by date_unix and get the soonest one
        launches.sort(key=lambda x: x.get('date_unix', float('inf')))
        next_launch = launches[0]

        # Get rocket information
        rocket_response = requests.get(
            "https://api.spacexdata.com/v4/rockets/{}".format(
                next_launch['rocket']))
        rocket_response.raise_for_status()
        rocket_data = rocket_response.json()

        # Get launchpad information
        launchpad_response = requests.get(
            "https://api.spacexdata.com/v4/launchpads/{}".format(
                next_launch['launchpad']))
        launchpad_response.raise_for_status()
        launchpad_data = launchpad_response.json()

        # Convert UTC timestamp to Eastern Time
        utc_time = datetime.fromtimestamp(
            next_launch['date_unix'], timezone.utc)

        # Using EDT (UTC-4) as per the example
        edt_offset = timedelta(hours=-4)
        launch_time = utc_time + edt_offset

        # Format with EDT offset (-04:00)
        launch_date = launch_time.strftime('%Y-%m-%dT%H:%M:%S-04:00')

        # Format the output string
        return "{} ({}) {} - {} ({})".format(
            next_launch['name'],
            launch_date,
            rocket_data['name'],
            launchpad_data['name'],
            launchpad_data['locality']
        )
    except requests.exceptions.RequestException as e:
        return "Error fetching launch data: {}".format(str(e))
    except (KeyError, ValueError) as e:
        return "Error processing launch data: {}".format(str(e))


if __name__ == '__main__':
    result = get_upcoming_launch()
    print(result)