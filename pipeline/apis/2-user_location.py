#!/usr/bin/env python3
"""
Script that prints the location of a specific GitHub user using the GitHub API.
Handles rate limiting and user not found cases.
"""
import sys
import requests
import time


def get_user_location(api_url):
    """
    Fetches and returns the location of a GitHub user.

    Args:
        api_url (str): The full GitHub API URL for the user

    Returns:
        str: User's location or appropriate error message
    """
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            user_data = response.json()
            return user_data.get('location') or 'Not found'
        elif response.status_code == 404:
            return 'Not found'
        elif response.status_code == 403:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            current_time = int(time.time())
            minutes_remaining = max(0, (reset_time - current_time) // 60)
            return 'Reset in {} min'.format(minutes_remaining)
        else:
            return 'Error: {}'.format(response.status_code)

    except requests.exceptions.RequestException:
        return 'Error: Unable to connect to GitHub API'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <api_url>'.format(sys.argv[0]))
        sys.exit(1)

    api_url = sys.argv[1]
    result = get_user_location(api_url)
    print(result)