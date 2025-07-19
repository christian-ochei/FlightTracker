
# airlines = get_flights_by_airport(flight_numbers)


import os

import streamlit as st
from datetime import datetime, timedelta
import pytz
import time

# --- DATA (As provided in the prompt) ---
# In a real app, this would come from an API call
null = None
from datetime import datetime, timedelta
import pytz

# --- MOCK DATA (As provided in the original script) ---
# In a real app, this would come from an API call
import streamlit as st
st.set_page_config(page_title="First Love Church - Live Flight Tracker", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
import re
from typing import Dict, List, Any
import json

# Sample flight data (your existing data)
flight_data = {
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 2,
        "total": 2
    },
    "data": [{'flight_date': '2025-07-15', 'flight_status': 'active', 'departure': {'airport': 'John F Kennedy International', 'timezone': 'America/New_York', 'iata': 'JFK', 'icao': 'KJFK', 'terminal': '8', 'gate': None, 'delay': 26, 'scheduled': '2025-07-15T01:20:00+00:00', 'estimated': '2025-07-15T01:20:00+00:00', 'actual': '2025-07-15T01:45:00+00:00', 'estimated_runway': '2025-07-15T01:45:00+00:00', 'actual_runway': '2025-07-15T01:45:00+00:00'}, 'arrival': {'airport': 'Doha International', 'timezone': 'Asia/Qatar', 'iata': 'DOH', 'icao': 'OTHH', 'terminal': 'HIA', 'gate': None, 'baggage': '7', 'scheduled': '2025-07-15T20:45:00+00:00', 'delay': None, 'estimated': None, 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'airline': {'name': 'Qatar Airways', 'iata': 'QR', 'icao': 'QTR'}, 'flight': {'number': '706', 'iata': 'QR706', 'icao': 'QTR706', 'codeshared': None}, 'aircraft': None, 'live': None}, {'flight_date': '2025-07-16', 'flight_status': 'scheduled', 'departure': {'airport': 'Bole International', 'timezone': 'Africa/Addis_Ababa', 'iata': 'ADD', 'icao': 'HAAB', 'terminal': '2', 'gate': 'A10', 'delay': None, 'scheduled': '2025-07-16T08:40:00+00:00', 'estimated': '2025-07-16T08:40:00+00:00', 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'arrival': {'airport': 'Kotoka', 'timezone': 'Africa/Accra', 'iata': 'ACC', 'icao': 'DGAA', 'terminal': '3', 'gate': None, 'baggage': None, 'scheduled': '2025-07-16T11:20:00+00:00', 'delay': None, 'estimated': None, 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'airline': {'name': 'Ethiopian Airlines', 'iata': 'ET', 'icao': 'ETH'}, 'flight': {'number': '921', 'iata': 'ET921', 'icao': 'ETH921', 'codeshared': None}, 'aircraft': None, 'live': None}, {'flight_date': '2025-07-15', 'flight_status': 'active', 'departure': {'airport': 'Bole International', 'timezone': 'Africa/Addis_Ababa', 'iata': 'ADD', 'icao': 'HAAB', 'terminal': '2', 'gate': 'A10', 'delay': 24, 'scheduled': '2025-07-15T08:40:00+00:00', 'estimated': '2025-07-15T08:40:00+00:00', 'actual': '2025-07-15T09:03:00+00:00', 'estimated_runway': '2025-07-15T09:03:00+00:00', 'actual_runway': '2025-07-15T09:03:00+00:00'}, 'arrival': {'airport': 'Kotoka', 'timezone': 'Africa/Accra', 'iata': 'ACC', 'icao': 'DGAA', 'terminal': '3', 'gate': None, 'baggage': None, 'scheduled': '2025-07-15T11:20:00+00:00', 'delay': None, 'estimated': None, 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'airline': {'name': 'Ethiopian Airlines', 'iata': 'ET', 'icao': 'ETH'}, 'flight': {'number': '921', 'iata': 'ET921', 'icao': 'ETH921', 'codeshared': None}, 'aircraft': None, 'live': None}, {'flight_date': '2025-07-16', 'flight_status': 'scheduled', 'departure': {'airport': 'Hartsfield-jackson Atlanta International', 'timezone': 'America/New_York', 'iata': 'ATL', 'icao': 'KATL', 'terminal': 'I', 'gate': None, 'delay': None, 'scheduled': '2025-07-16T10:35:00+00:00', 'estimated': '2025-07-16T10:35:00+00:00', 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'arrival': {'airport': 'Bole International', 'timezone': 'Africa/Addis_Ababa', 'iata': 'ADD', 'icao': 'HAAB', 'terminal': '2', 'gate': None, 'baggage': None, 'scheduled': '2025-07-17T07:50:00+00:00', 'delay': None, 'estimated': None, 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'airline': {'name': 'Ethiopian Airlines', 'iata': 'ET', 'icao': 'ETH'}, 'flight': {'number': '519', 'iata': 'ET519', 'icao': 'ETH519', 'codeshared': None}, 'aircraft': None, 'live': None}, {'flight_date': '2025-07-16', 'flight_status': 'scheduled', 'departure': {'airport': 'Doha International', 'timezone': 'Asia/Qatar', 'iata': 'DOH', 'icao': 'OTHH', 'terminal': None, 'gate': None, 'delay': None, 'scheduled': '2025-07-16T08:40:00+00:00', 'estimated': '2025-07-16T08:40:00+00:00', 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'arrival': {'airport': 'Kotoka', 'timezone': 'Africa/Accra', 'iata': 'ACC', 'icao': 'DGAA', 'terminal': '3', 'gate': None, 'baggage': None, 'scheduled': '2025-07-16T13:40:00+00:00', 'delay': None, 'estimated': None, 'actual': None, 'estimated_runway': None, 'actual_runway': None}, 'airline': {'name': 'Qatar Airways', 'iata': 'QR', 'icao': 'QTR'}, 'flight': {'number': '1423', 'iata': 'QR1423', 'icao': 'QTR1423', 'codeshared': None}, 'aircraft': None, 'live': None}]
}


@st.cache_data
def load_excel_data():
    """Load and process the Excel file containing passenger information."""
    # Try to read the Excel file
    df = pd.read_excel('sample - FLNA Travel Info Data.xlsx')

    # Clean column names (remove extra spaces, standardize)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    # Expected columns (adjust based on your actual Excel structure)
    # ['name', 'flight_number', 'departure_date', 'seat', 'class', 'phone', 'email']
    print(df)

    # If the actual columns are different, map them here
    # For example, if your Excel has different column names:
    column_mapping = {
        'passenger_name': 'name',
        'flight_no': 'flight_number',
        'travel_date': 'departure_date',
        'seat_number': 'seat',
        'travel_class': 'class',
        'phone_number': 'phone',
        'email_address': 'email'
    }

    # Apply column mapping if needed
    df = df.rename(columns=column_mapping)

    # Ensure flight_number is string and clean it
    if 'flight_number' in df.columns:
        df['flight_number'] = df['flight_number'].astype(str).str.strip().str.upper()

    # Parse dates if they exist
    # if 'departure_date' in df.columns:
    #     df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')
    df['arrival_datetime'] = pd.to_datetime(df['arrival_date'].str.split('T').str[0] + ' ' + df['arrival_time'])
    df['departure_datetime'] = pd.to_datetime(df['departure_date'].str.split('T').str[0] + ' ' + df['departure_time'])
    return df

def copy_to_clipboard(text):

    """
    Copy string to clipboard using pyperclip (cross-platform).

    Args:
        text (str): String to copy to clipboard

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pyperclip
        import subprocess
        import sys
        pyperclip.copy(text)
        print(f"Copied {len(text)} characters to clipboard")
        return True
    except Exception as e:
        print(f"Error copying to clipboard: {e}")
        return False

def excel_to_json_string():
    df = load_excel_data()
    # Convert to JSON string with no indentation or newlines
    json_string = df.to_json(orient='records', date_format='iso')

    # Print the raw JSON
    print(json_string)
    copy_to_clipboard(json_string.replace('"', '\\"'))
    return json_string


def load_excel_from_env():
    """
    Load Excel data from EXCEL_SHEET environment variable.

    Returns:
        pd.DataFrame: DataFrame loaded from the environment variable JSON string

    Raises:
        ValueError: If EXCEL_SHEET environment variable is not set or empty
        json.JSONDecodeError: If the JSON string is invalid
    """
    # Get the JSON string from environment variable
    json_string = os.getenv('EXCEL_SHEET')

    if not json_string:
        raise ValueError("EXCEL_SHEET environment variable is not set or is empty")

    try:
        # Parse JSON string to Python objects
        data = json.loads(json_string)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # If departure_date exists, convert it back to datetime
        # if 'departure_date' in df.columns:
        #     df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')

        print(f"Successfully loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        df['arrival_datetime'] = pd.to_datetime(df['arrival_date'].str.split('T').str[0] + ' ' + df['arrival_time'])
        df['departure_datetime'] = pd.to_datetime(df['departure_date'].str.split('T').str[0] + ' ' + df['departure_time'])
        return df

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in EXCEL_SHEET environment variable: {e}")


def json_string_to_dataframe(json_string):
    """
    Convert JSON string to pandas DataFrame.

    Args:
        json_string (str): JSON string representation of data

    Returns:
        pd.DataFrame: DataFrame loaded from JSON string
    """
    # Parse JSON string to Python objects
    data = json.loads(json_string)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # If departure_date exists, convert it back to datetime
    if 'departure_date' in df.columns:
        df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')

    return df

import os
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
import pytz
import re
from dateutil.parser import parse as date_parse
from typing import Dict, List, Any, Tuple

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FLC Live Flight & Passenger Tracker",
    layout="wide",
    initial_sidebar_state="collapsed"
)

STOP_WORDS = {'in', 'the', 'going', 'to', 'from', 'of', 'at', 'a', 'on', 'for', 'with'}
IATA_COUNTRY_MAP = {'ACC': 'Ghana', 'JFK': 'USA', 'DOH': 'Qatar', 'LHR': 'UK', 'ADD': 'Ethiopia', 'CDG': 'France',
                    'IAD': 'USA', 'LOS': 'Nigeria', 'CMN': 'Morocco'}


def process_data(passenger_df: pd.DataFrame, flight_api_data: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
    people_list = []
    flight_api_data = flight_api_data['data']
    flight_list = []

    for _, row in passenger_df.iterrows():
        arrival_flights = [f.strip() for f in row['arrival_info'].replace(' ', '').split(',')]
        person = {
            'type': 'person', 'full_name': row['full_name'], 'campus': row.get('campus', ''),
            'arrival_datetime_utc': row['arrival_datetime'].tz_localize('UTC'),
            'departure_datetime_utc': row['departure_datetime'].tz_localize('UTC'),
            'arrival_flights': arrival_flights,
            'departure_flights': [f.strip() for f in row['departure_info'].replace(' ', '').split(',')],
            'search_text': f"{row['full_name']} {row.get('campus', '')} {' '.join(arrival_flights)}".lower(), 'match_tip': ''
        }
        people_list.append(person)

    for flight_details in flight_api_data:
        flight_iata = flight_details['flight']['iata']
        passengers_on_flight = [p for p in people_list if flight_iata in p['arrival_flights']]
        search_text = f"{flight_details['airline']['name']} {flight_details['flight']['iata']} {flight_details['departure']['airport']} {flight_details['arrival']['airport']} {flight_details['arrival']['iata']} {flight_details['flight_status']}  {flight_details['flight']['iata']} {' '.join([p['full_name'] for p in passengers_on_flight])}".lower()
        flight = {
            'type': 'flight', 'details': flight_details, 'passengers': passengers_on_flight,
            'arrival_datetime_utc': datetime.fromisoformat(flight_details['arrival']['scheduled']).replace(
                tzinfo=timezone.utc),
            'search_text': search_text, 'match_tip': ''
        }
        flight_list.append(flight)

    return people_list, flight_list


import re
from typing import Dict, List, Any, Tuple
from dateutil.parser import parse as date_parse

# Define stop words and date indicators outside the function for efficiency
STOP_WORDS = {'in', 'the', 'going', 'to', 'from', 'of', 'at', 'a', 'on', 'for', 'with'}
DATE_INDICATORS = {
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'
}


def parse_query(query: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parses a search query to separate search terms from special keywords.
    This version uses a more robust method to detect dates, preventing
    flight numbers and other codes from being misinterpreted.
    """
    query_lower = query.lower()
    tokens = [word for word in re.split(r'\s+', query_lower) if word and word not in STOP_WORDS]

    # --- 1. Extract standard keywords ---
    keywords = {
        'person': 'person' in tokens or 'people' in tokens,
        'flight': 'flight' in tokens or 'plane' in tokens,
        'landed': 'landed' in tokens,
        'landing': 'landing' in tokens,
        'delayed': 'delayed' in tokens,
        'on_time': 'on time' in query_lower,
        'today': 'today' in tokens,
        'yesterday': 'yesterday' in tokens
    }

    # --- 2. Separate potential search terms from keywords ---
    potential_search_terms = [t for t in tokens if t not in keywords]

    # --- 3. Intelligently find and parse date parts ---
    date_parts = []
    # Keep track of terms that are definitely not part of a date
    final_search_terms = []

    for term in potential_search_terms:
        # A term is likely a date part if it's a known date word or a number (e.g., "4", "24th")
        cleaned_term = re.sub(r'(st|nd|rd|th)$', '', term)
        if term in DATE_INDICATORS or cleaned_term.isdigit():
            date_parts.append(term)
        else:
            final_search_terms.append(term)

    # If we found any date-related words, try to parse them
    if date_parts:
        try:
            # Parse only the collected date parts for higher accuracy
            parsed_date = date_parse(' '.join(date_parts))
            keywords['date'] = parsed_date.date()
        except (ValueError, TypeError):
            # If parsing fails, it wasn't a valid date. Add the parts back to the search terms.
            final_search_terms.extend(date_parts)

    # Return the search terms that were NOT identified as part of a date
    return final_search_terms, keywords


def search_and_filter(query: str, all_people: List[Dict], all_flights: List[Dict]):
    if not query.strip(): return sorted(all_flights, key=lambda x: abs(
        (x['arrival_datetime_utc'] - datetime.now(timezone.utc)).total_seconds()))
    terms, keywords = parse_query(query)
    show_people, show_flights = keywords['person'] or not keywords['flight'], keywords['flight'] or not keywords[
        'person']
    potential_results = []
    if show_people: potential_results.extend(all_people)
    if show_flights: potential_results.extend(all_flights)
    results = []
    now = datetime.now(timezone.utc)
    for item in potential_results:
        item['match_tip'] = ''
        is_match = True
        status = item['details']['flight_status'] if item['type'] == 'flight' else 'N/A'
        if keywords['landed'] and status != 'landed': is_match = False
        if keywords['landing'] and status != 'active': is_match = False
        if keywords['delayed'] and (
                item.get('details', {}).get('departure', {}).get('delay') or 0) == 0: is_match = False
        if keywords['on_time'] and (
                item.get('details', {}).get('departure', {}).get('delay') or 0) > 0: is_match = False
        item_date = item['arrival_datetime_utc'].date()
        if keywords.get('date') and item_date != keywords['date']: is_match = False
        if keywords['today'] and item_date != now.date(): is_match = False
        if keywords['yesterday'] and item_date != (now.date() - timedelta(days=1)): is_match = False
        if terms:
            text_to_search = item['search_text']
            if item['type'] == 'flight':
                dest_iata = item['details']['arrival']['iata']
                text_to_search += f" {IATA_COUNTRY_MAP.get(dest_iata, '').lower()}"
            elif item['type'] == 'person':
                final_flight_iata = item['arrival_flights'][-1]
                final_flight = next((f for f in all_flights if f['details']['flight']['iata'] in final_flight_iata),
                                    None)
                if final_flight: text_to_search += f" {IATA_COUNTRY_MAP.get(final_flight['details']['arrival']['iata'], '').lower()}"
            if not all(term in text_to_search for term in terms):
                is_match = False

        if is_match:
            if terms:
                if item['type'] == 'flight':
                    for p in item['passengers']:
                        if any(term in p['full_name'].lower() for term in terms):
                            item['match_tip'] = f"✓ {p['full_name']} found"
                            break
                elif item['type'] == 'person':
                    if any(term.upper() in item['arrival_flights'] for term in terms):
                        item['match_tip'] = f"✓ Flight(s) found"
            results.append(item)
    return sorted(results, key=lambda x: abs((x['arrival_datetime_utc'] - now).total_seconds()))


# --- UI DISPLAY COMPONENTS ---

def get_time_ago_string(dt_utc: datetime) -> str:
    now = datetime.now(timezone.utc)
    delta = now - dt_utc
    seconds = abs(delta.total_seconds())
    tense = "ago" if delta.total_seconds() > 0 else "from now"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0: return f"{int(days)} {'day' if int(days) == 1 else 'days'} {tense}"
    if hours > 0: return f"{int(hours)} {'hour' if int(hours) == 1 else 'hours'} {tense}"
    if minutes > 0: return f"{int(minutes)} {'minute' if int(minutes) == 1 else 'minutes'} {tense}"
    return "just now"


def get_dynamic_status_color(arrival_utc: datetime, status: str) -> str:
    """Returns a color based on time to landing."""
    now = datetime.now(timezone.utc)
    delta_seconds = (arrival_utc - now).total_seconds()

    # Within 2 hours (before or after) -> Critical window
    if abs(delta_seconds) <= 7200:
        return "#ff4b4b"  # Red
    # Landed more than 2 hours ago -> Safe
    if status == 'landed':
        return "#00c3ff"  # Blue
    # Arriving between 2 and 24 hours from now -> Approaching
    if 7200 < delta_seconds <= 86400:
        return "#ffad5a"  # Orange
    # More than 24 hours away -> Scheduled
    return "#808080"  # Gray


def display_statistics(people: List[Dict], flights: List[Dict]):
    st.image("FirstLove.png", width=60)
    st.title("First Love - Flight Tracker")
    st.html(
        """
        <style>
        [data-testid="stMetricValue"] > div {
            font-size: 2rem; /* Adjust the font size as needed */
            font-weight: 600
        }
        
        small {
            font-size: 18px;
        }
        </style>
        """,
    )
    with st.expander("Show/Hide Flight & Passenger Summary", expanded=False):
        cols = st.columns(4)
        sorted_flights = sorted(flights, key=lambda x: x['details']['flight']['iata'])
        for i, flight in enumerate(sorted_flights):
            num_pax = len(flight['passengers'])
            dep = flight['details']['departure']['iata']
            arr = flight['details']['arrival']['iata']
            # cols[i % 4].metric(
            #     value=f"✈ {flight['details']['flight']['iata']} ({dep} → {arr})",
            #     label=f"{num_pax} {'Passenger' if num_pax == 1 else 'Passengers'}"
            # )
            cols[i % 4].html(
                f"""
                <div style="padding: 10px; border: 1px solid #e0e0e011; border-radius: 5px;">
                    <div style="font-size: 13.3px; color: #666; margin-bottom: 5px;">
                        {num_pax} {'Passenger' if num_pax == 1 else 'Passengers'}
                    </div>
                    <div style="font-size: 15.4px; font-weight: bold;">
                        ✈ {flight['details']['flight']['iata']} ({dep} → {arr})
                    </div>
                </div>
                """
            )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Passengers", f"{len(people)}")
    c2.metric("Passengers Landed",
              f"{sum(len(f['passengers']) for f in flights if f['details']['flight_status'] == 'landed')}")
    c3.metric("Total Flights", f"{len(flights)}")
    c4.metric("Flights Landed", f"{sum(1 for f in flights if f['details']['flight_status'] == 'landed')}")
    c5.metric("Flights Delayed", f"{sum(1 for f in flights if f['details']['departure'].get('delay', 0) and f['details']['departure'].get('delay', 0) > 0)}")


def display_person_card(person: Dict, local_tz, flights):
    arrival_dt_local = person['arrival_datetime_utc'].astimezone(local_tz)
    status = "Landed" if person['arrival_datetime_utc'] < datetime.now(timezone.utc) else "En Route"
    status_color = get_dynamic_status_color(person['arrival_datetime_utc'], status)

    # 1. Build the HTML for each clickable flight card in the grid
    flight_cards_html = []
    for flight in flights:
        details = flight['details']
        flight_status = details['flight_status']
        flight_status_color = get_dynamic_status_color(flight['arrival_datetime_utc'], flight_status)
        arr_dt_local = datetime.fromisoformat(details['arrival']['scheduled']).astimezone(local_tz)

        # The destination IATA code to be used in the search query
        destination_iata = details['flight']['iata']
        encoded_iata = urllib.parse.quote(destination_iata)

        # The entire card is now a clickable link (<a> tag) with the class "itinerary-card"
        flight_card = f"""
        <a href="?search={encoded_iata}" target="_self" class="itinerary-card" style="border-left-color: {flight_status_color};">
            <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 5px;">
                ✈ {details['flight']['iata']}
                <span style="float: right; font-weight: bold; color: {flight_status_color}; font-size: 0.9em;">{flight_status.upper()}</span>
            </div>
            <div style="font-size: 1em; color: #ccc; margin-bottom: 10px;">
                {details['departure']['iata']} → <strong>{details['arrival']['iata']}</strong>
            </div>
            <div style="font-size: 0.85em; color: #aaa;">
                Arrives: {arr_dt_local.strftime('%b %d, %I:%M %p')}
                {f" | <span style='color: #ffad5a;'>Delayed</span>" if details['departure'].get('delay') else ""}
            </div>
        </a>
        """
        flight_cards_html.append(flight_card)

    # 2. Combine the flight cards into a grid container
    flights_grid = f"""
    <div style="margin-top: 1.5em;">
        <strong style="color: #ddd; display: block; margin-bottom: 10px;">Flight Itinerary ({len(flights)} {'leg' if len(flights) == 1 else 'legs'})</strong>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            {''.join(flight_cards_html)}
        </div>
    </div>
    """

    st.html(f"""
    <style>
        .itinerary-card {{
            display: block;
            border: 1px solid #333;
            border-left-width: 5px;
            border-radius: 8px;
            padding: 15px;
            background-color: #262626;
            text-decoration: none;
            color: inherit;
            transition: background-color 0.2s ease;
        }}
        .itinerary-card:hover {{
            background-color: #3a3a3a; /* Softly lighter background on hover */
            text-decoration: none;
            color: white;
        }}
    </style>
    <div style="border: 1px solid #333; border-left: 10px solid {status_color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: #1a1a1a;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0;">👤 {person['full_name']}</h3>
            {f'<span style="background-color: #4CAF50; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em;">{person["match_tip"]}</span>' if person["match_tip"] else ''}
        </div>
        <p style="color: #aaa; margin: 0 0 1em 0;">{person['campus']}</p>
        <hr style="border-color: #333; margin-top: 0; margin-bottom: 1em;">

        {flights_grid}
    </div>
    """)

def create_maps_link(airport_name, terminal, gate):
    """Create a Google Maps link for the airport gate."""
    if gate and gate.lower().strip() != 'n/a':
        query = f"{airport_name} airport terminal {terminal} gate {gate}"
    else:
        query = f"{airport_name} airport terminal {terminal}"

    # URL encode the query
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    return f"https://www.google.com/maps/search/?api=1&query={encoded_query}"


def display_flight_card(flight: Dict, local_tz):
    details = flight['details']
    status = details['flight_status']
    status_color = get_dynamic_status_color(flight['arrival_datetime_utc'], status)
    dep_dt_local = datetime.fromisoformat(details['departure']['scheduled']).astimezone(local_tz)
    arr_dt_local = datetime.fromisoformat(details['arrival']['scheduled']).astimezone(local_tz)

    # --- New: Generate clickable passenger tags ---
    passenger_tags = []
    for p in flight['passengers']:
        encoded_name = urllib.parse.quote(p['full_name'])
        tag = f"""
        <a href="?search={encoded_name}" target="_self" class="passenger-tag">
            {p['full_name']}
        </a>
        """
        passenger_tags.append(tag)

    passengers_html = f"""
    <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px;">
        {''.join(passenger_tags)}
    </div>
    """ if passenger_tags else ""

    st.html(f"""
    <style>
        .passenger-tag {{
            background-color: #333;
            color: #ddd;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            text-decoration: none;
            transition: background-color 0.2s ease;
        }}
        .passenger-tag:hover {{
            background-color: #007bff;
            color: white;
            text-decoration: none;
        }}
    </style>
    <div style="border: 1px solid #333; border-left: 10px solid {status_color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: #1a1a1a;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0;">✈ <a style="opacity: 0.9">{details['airline']['name']}</a> <code>{details['flight']['iata']}</code></h3>
                <span style="font-weight: bold; color: {status_color};">{status.upper()}</span>
            </div>
            {f'<span style="background-color: #4CAF50; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em;">{flight["match_tip"]}</span>' if flight["match_tip"] else ''}
        </div>

        {passengers_html}

        <hr style="border-color: #333;">
        <div style="display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; text-align: center; gap: 1em;">
            <div style="text-align: left;">
                <strong>Departing from {details['departure']['iata']}</strong><br>
                <small>{details['departure']['airport']}</small><br>
                <small>{dep_dt_local.strftime('%b %d, %I:%M %p')}</small>
            </div>
            <div>→</div>
            <div style="text-align: right;">
                <strong>Arriving to <b>{details['arrival']['iata']}</b></strong><br>
                <small>{details['arrival']['airport']}</small><br>
                <small>{arr_dt_local.strftime('%b %d, %I:%M %p')}</small>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <strong>Arriving {get_time_ago_string(flight['arrival_datetime_utc'])}</strong> 
            {(f" | <span style='color: #ffad5a;'>Delayed by {details['departure']['delay']} minutes</span>") if details['departure'].get('delay') else f"| <span style='color: #44ff6699;'>On Time"}
        </div>
        {maps_link(details)}
    </div>
    """)
def maps_link(details):
    maps_url = create_maps_link(
        details['departure']['airport'],
        details['departure']['terminal'],
        details['departure']['gate']
    )
    gate_info = ''
    if details['departure']['gate'] and details['departure']['gate'].lower().strip() != 'n/a':
        gate_info = f"Gate: {details['departure']['gate'] or 'N/A'}&nbsp;&nbsp;&nbsp;"
    return f'<br>{gate_info}<a href="{maps_url}" target="_blank" class="maps-button" style="background-color: #00000022; color: #ffffff99; font-weight: 600; text-decoration:none; padding: 2px; padding-right: 10px; border-radius: 4px">⌖ Open in Maps</a>'


# --- MAIN APP LOGIC ---
passenger_df = load_excel_from_env()
all_people, all_flights = process_data(passenger_df, flight_data)
display_statistics(all_people, all_flights)
st.markdown("---")
st.markdown(
    "**Search Examples:** `John Doe`, `flight ET921`, `people landed yesterday`, `delayed`, `FLC Dallas`")
# search_query = st.text_input("Search for flights or people...",
#                              placeholder="Search by name, flight, date, status, campus...",
#                              label_visibility="collapsed")
# Check for search query params to enable clickable passenger tags
query_params = st.query_params.to_dict()
# Use the URL param as the default for the text input, otherwise default to empty
print(query_params, "query_params")
search_from_url = query_params.get("search", "")

search_query = st.text_input("Search for flights or people...",
                             value=search_from_url, # Set the value from the URL
                             placeholder="Search by name, flight, date, status, campus...",
                             label_visibility="collapsed")
results = search_and_filter(search_query, all_people, all_flights)

# Use a timezone relevant to your location in Texas
local_timezone = pytz.timezone("America/Chicago")

if not results:
    if search_query:
        _, keywords = parse_query(search_query)
        status_part = ""
        if keywords['delayed']: status_part = "delayed "
        if keywords['landed']: status_part = "landed "
        type_part = "result"
        if keywords['person'] and not keywords['flight']: type_part = "person"
        if keywords['flight'] and not keywords['person']: type_part = "flight"
        st.warning(f"Found no {status_part}{type_part} for '{search_query}'")
    else:
        st.info("Showing all flights by default. Start typing to search for people and flights.")
else:
    st.markdown(f"**Found {len(results)} result{'s' if len(results) != 1 else ''}**")
    for item in results:
        if item['type'] == 'person':
            flights = [a for a in all_flights if a['type'] == 'flight' and item in a['passengers']]
            display_person_card(item, local_timezone, flights=flights)
        elif item['type'] == 'flight':
            display_flight_card(item, local_timezone)
