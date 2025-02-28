#!/usr/bin/env python3
"""
This module contains a function to fetch home planets of sentient species
from the Swapi API.
"""
import requests


def sentientPlanets():
    """
    Returns a list of names of home planets of all sentient species.

    Returns:
        list: A list of planet names where sentient species originate from.
    """
    base_url = "https://swapi.dev/api/species/"
    planets = []
    page = 1

    while True:
        response = requests.get("{}?page={}".format(base_url, page))
        data = response.json()

        for species in data['results']:
            # Check if species is sentient
            if (species.get('designation') == 'sentient' or
                    species.get('classification') == 'sentient'):
                # Get homeworld if it exists
                homeworld = species.get('homeworld')
                if homeworld:
                    # Fetch planet name from homeworld URL
                    planet_response = requests.get(homeworld)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets.append(planet_data['name'])
                else:
                    planets.append('unknown')

        if data['next'] is None:
            break
        page += 1

    # Remove duplicates while maintaining order
    seen = set()
    unique_planets = []
    for planet in planets:
        if planet not in seen:
            seen.add(planet)
            unique_planets.append(planet)

    return unique_planets