#!/usr/bin/env python3
"""Pipeline Api"""
import requests

if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)

    if r.status_code != 200:
        print("Failed to fetch launches")
        exit(1)

    rocket_dict = {}

    for launch in r.json():
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_dict[rocket_id] = rocket_dict.get(rocket_id, 0) + 1

    for key, value in sorted(rocket_dict.items(), key=lambda kv: kv[1], reverse=True):
        rurl = f"https://api.spacexdata.com/v4/rockets/{key}"
        req = requests.get(rurl)

        if req.status_code == 200:
            rocket_data = req.json()
            rocket_name = rocket_data.get("name", "Unknown Rocket")
            print(f"{rocket_name}: {value}")
        else:
            print(f"Failed to fetch rocket details for ID: {key}")
