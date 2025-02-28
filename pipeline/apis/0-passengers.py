#!/usr/bin/env python3
"""
This module contains a function to fetch ships from the Swapi API
that can hold a given number of passengers.
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of ships that can hold a given number of passengers.

    Args:
        passengerCount (int): The number of passengers to accommodate.

    Returns:
        list: A list of ship names that can hold the given number of
        passengers.
    """
    base_url = "https://swapi.dev/api/starships/"
    ships = []
    page = 1

    while True:
        response = requests.get("{}?page={}".format(base_url, page))
        data = response.json()

        for ship in data['results']:
            if ship['passengers'] != 'n/a' and ship['passengers'] != 'unknown':
                try:
                    if int(ship['passengers'].replace(',', '')
                           ) >= passengerCount:
                        ships.append(ship['name'])
                except ValueError:
                    continue

        if data['next'] is None:
            break
        page += 1

    return ships