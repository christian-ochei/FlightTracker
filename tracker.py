
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
    if 'departure_date' in df.columns:
        df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')
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
        if 'departure_date' in df.columns:
            df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')

        print(f"Successfully loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

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

def get_flight_numbers_set(passenger_df):
    """Create a set of unique flight numbers from the passenger data."""
    if 'flight_number' in passenger_df.columns:
        return set(passenger_df['flight_number'].dropna().unique())
    return set()


def get_passengers_for_flight(flight_number, flight_date, passenger_df):
    """Get all passengers for a specific flight and date."""
    if passenger_df.empty:
        return []

    mask = passenger_df['arrival_info'].str.replace(' ', '').str.contains(flight_number, na=False)
    filtered = passenger_df[mask]
    return filtered.to_dict('records')


def create_maps_link(airport_name, terminal, gate):
    """Create a Google Maps link for the airport gate."""
    if gate and gate != 'N/A':
        query = f"{airport_name} airport terminal {terminal} gate {gate}"
    else:
        query = f"{airport_name} airport terminal {terminal}"

    # URL encode the query
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    return f"https://www.google.com/maps/search/?api=1&query={encoded_query}"


def search_flights(query, flights, passenger_df):
    """Enhanced search function that searches through multiple fields."""
    if not query:
        return flights

    query = query.lower().strip()
    filtered_flights = []

    for flight in flights:
        # Search in flight data
        flight_text = " ".join([
            flight['flight']['iata'],
            flight['flight']['number'],
            flight['airline']['name'],
            flight['departure']['airport'],
            flight['arrival']['airport'],
            flight['departure']['iata'],
            flight['arrival']['iata'],
            flight['flight_status'],
            flight['flight_date']
        ]).lower()

        # Search in passenger data for this flight
        passengers = get_passengers_for_flight(
            flight['flight']['iata'],
            flight['flight_date'],
            passenger_df
        )

        passenger_text = ""
        if passengers:
            passenger_text = " ".join([
                                          str(p.get('name', '')) + ' ' +
                                          str(p.get('seat', '')) + ' ' +
                                          str(p.get('class', '')) + ' ' +
                                          str(p.get('phone', '')) + ' ' +
                                          str(p.get('email', '')) + ' '                                    for p in passengers
            ]).lower()

        # Date-based search
        try:
            flight_date = datetime.fromisoformat(flight['flight_date'])
            date_text = " ".join([
                flight_date.strftime('%B'),  # Month name
                flight_date.strftime('%b'),  # Short month
                flight_date.strftime('%A'),  # Day of week
                flight_date.strftime('%a'),  # Short day
                str(flight_date.day),
                str(flight_date.year)
            ]).lower()
        except:
            date_text = ""

        # Combine all searchable text
        searchable_text = f"{flight_text} {passenger_text} {date_text}"

        # Check if query matches any part of the searchable text
        if query in searchable_text:
            filtered_flights.append(flight)

    return filtered_flights


def get_flight_progress(flight):
    """Calculate flight progress percentage."""
    status = flight['flight_status']
    if status == 'landed':
        return 100
    if status != 'active' or not flight['departure']['actual'] or not flight['arrival']['scheduled']:
        return 0

    now_utc = datetime.now(pytz.utc)
    start_time = datetime.fromisoformat(flight['departure']['actual'])
    end_time = datetime.fromisoformat(flight['arrival']['scheduled'])

    total_duration = (end_time - start_time).total_seconds()
    elapsed_duration = (now_utc - start_time).total_seconds()

    if total_duration <= 0:
        return 0
    progress = (elapsed_duration / total_duration) * 100
    return min(max(progress, 0), 100)


def get_precise_landing_status(flight):
    """Calculate the precise landing status string using local time zone."""
    status = flight['flight_status']
    local_tz = datetime.now().astimezone().tzinfo
    now_local = datetime.now(local_tz)

    if status == 'landed' and flight['arrival']['actual']:
        landed_time_utc = datetime.fromisoformat(flight['arrival']['actual'])
        if landed_time_utc.tzinfo is None:
            landed_time_utc = landed_time_utc.replace(tzinfo=timezone.utc)

        landed_time_local = landed_time_utc.astimezone(local_tz)
        time_since_landing = now_local - landed_time_local

        if time_since_landing.total_seconds() < 0:
            return "Landing time inconsistent"

        # Fix: Use total_seconds() and properly calculate days, hours, minutes
        total_seconds = int(time_since_landing.total_seconds())
        days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
        hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
        minutes, _ = divmod(remainder, 60)  # 60 seconds in a minute

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days > 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

        if not parts:
            return "**Landed:** just now"

        return f"**Landed:** {', '.join(parts)} ago"

    elif flight['arrival']['scheduled']:
        scheduled_time_utc = datetime.fromisoformat(flight['arrival']['scheduled'])
        if scheduled_time_utc.tzinfo is None:
            scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)

        scheduled_time_local = scheduled_time_utc.astimezone(local_tz)
        time_until_landing = scheduled_time_local - now_local

        if time_until_landing.total_seconds() < 0:
            time_overdue = now_local - scheduled_time_local

            # Fix: Use total_seconds() and properly calculate days, hours, minutes
            total_seconds = int(time_overdue.total_seconds())
            days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
            hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
            minutes, _ = divmod(remainder, 60)  # 60 seconds in a minute

            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days > 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

            overdue_text = ', '.join(parts) if parts else "just now"
            return f"**Overdue:** {overdue_text} past scheduled landing"

        # Fix: Use total_seconds() and properly calculate days, hours, minutes
        total_seconds = int(time_until_landing.total_seconds())
        days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
        hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
        minutes, _ = divmod(remainder, 60)  # 60 seconds in a minute

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days > 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

        if not parts:
            time_text = "landing now"
        else:
            time_text = f"landing in {', '.join(parts)}"

        local_time_str = scheduled_time_local.strftime('%I:%M %p')
        local_date_str = scheduled_time_local.strftime('%b %d, %Y')

        return f"**Expected:** {time_text} at {local_time_str} on {local_date_str}"

    return "Landing time not available"


def get_formatted_time_string(dt_str, prefix=""):
    """Format an ISO datetime string into a full date/time with relative description."""
    if not dt_str:
        return "N/A"

    flight_time = datetime.fromisoformat(dt_str)
    now = datetime.now(pytz.utc)
    delta = now - flight_time

    absolute_time = flight_time.strftime('%d %b %Y %I:%M %p')

    tense = "ago" if delta.total_seconds() > 0 else "from now"
    total_seconds = abs(delta.total_seconds())

    # Fix: Properly calculate days, hours, minutes to avoid overflow
    if total_seconds < 60:
        value = int(total_seconds)
        unit = "second"
    elif total_seconds < 3600:
        value = int(total_seconds / 60)
        unit = "minute"
    elif total_seconds < 86400:
        value = int(total_seconds / 3600)
        unit = "hour"
    else:
        value = int(total_seconds / 86400)
        unit = "day"

    relative_time = f"{value} {unit}{'s' if value != 1 else ''} {tense}"
    return f"{prefix}{absolute_time} ({relative_time})"

def display_flight_card(flight, passenger_df):
    """Display a flight information card with passenger details."""
    status = flight['flight_status']
    passengers = get_passengers_for_flight(
        flight['flight']['iata'],
        flight['flight_date'],
        passenger_df
    )

    status_color = "gray"
    glowing_css = ""

    if status == 'active':
        status_color = "red"
    elif status == 'landed':
        status_color = "#00c3ff"
        glowing_css = """
            animation: glowing 3s infinite;
            text-shadow: 0 0 5px #00c3ff, 0 0 10px #00c3ff, 0 0 15px #00c3ff;
        """
        landed_time = datetime.fromisoformat(flight['arrival']['actual'])
        if datetime.now(pytz.utc) - landed_time > timedelta(days=1):
            status_color = "green"
            glowing_css = ""

    st.markdown(f"""
    <style>
        @keyframes glowing {{
            0% {{ box-shadow: 0 0 3px {status_color}; }}
            50% {{ box-shadow: 0 0 15px {status_color}; }}
            100% {{ box-shadow: 0 0 3px {status_color}; }}
        }}
        .card-container {{
            border: 1px solid #333;
            border-left: 10px solid {status_color};
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            background-color: #1a1a1a;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: all 0.3s ease-in-out;
            {glowing_css.replace('text-shadow', 'box-shadow')}
        }}
        .status-text {{
            color: {status_color};
            font-weight: bold;
            font-size: 1.2em;
            {glowing_css}
        }}
        .passenger-hover {{
            position: relative;
            display: inline-block;
        }}
        .passenger-hover .passenger-list {{
            visibility: hidden;
            width: 300px;
            background-color: #2a2a2a;
            color: white;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid #444;
        }}
        .passenger-hover:hover .passenger-list {{
            visibility: visible;
            opacity: 1;
        }}
        .maps-button {{
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            text-decoration: none;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        .maps-button:hover {{
            background-color: #45a049;
            text-decoration: none;
            color: white;
        }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)

        # Header with airline name as main title
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title(f"{flight['airline']['name']}")
            st.markdown(f"**Flight:** `{flight['flight']['iata']}` | **Date:** {flight['flight_date']}")
        with col2:
            st.markdown(f'<p class="status-text">{status.upper()}</p>', unsafe_allow_html=True)

        # Passenger count and hover menu
        if passengers:
            # passenger_names = [p.get('name', 'Unknown') for p in passengers]
            passenger_details = ""
            for p in passengers:
                passenger_details += f"• {p.get('full_name', 'Unknown')} - Seat {p.get('seat', 'N/A')} ({p.get('class', 'N/A')})<br>"

            st.markdown(f"""
            <div class="passenger-hover">
                <span style="color: #4CAF50; font-weight: bold; cursor: pointer;">
                    👥 {len(passengers)} Passenger{'s' if len(passengers) > 1 else ''} (hover for details)
                </span>
                <div class="passenger-list">
                    <strong>Passengers on this flight:</strong><br><br>
                    {passenger_details}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("👥 No passengers from your list on this flight")

        st.markdown("---")
        st.warning(get_precise_landing_status(flight))

        # Flight route with maps buttons
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"**From:** `{flight['departure']['iata']}`")
            st.write(flight['departure']['airport'])
            gate_info = f"Gate: {flight['departure']['gate'] or 'N/A'}"

            if flight['departure']['gate'] and flight['departure']['gate'] != 'N/A':
                maps_url = create_maps_link(
                    flight['departure']['airport'],
                    flight['departure']['terminal'],
                    flight['departure']['gate']
                )
                st.markdown(f'{gate_info} <a href="{maps_url}" target="_blank" class="maps-button" style="background-color: #000; color: #fff; font-weight: bold">📍 Open Maps</a>',
                            unsafe_allow_html=True)
            else:
                st.caption(gate_info)

        with col2:
            st.markdown("✈️", unsafe_allow_html=True)

        with col3:
            st.markdown(f"**To:** `{flight['arrival']['iata']}`")
            st.write(flight['arrival']['airport'])
            gate_info = f"Gate: {flight['arrival']['gate'] or 'N/A'}"

            if flight['arrival']['gate'] and flight['arrival']['gate'] != 'N/A':
                maps_url = create_maps_link(
                    flight['arrival']['airport'],
                    flight['arrival']['terminal'],
                    flight['arrival']['gate']
                )
                st.markdown(f'{gate_info} <a href="{maps_url}" target="_blank" class="maps-button" style="background-color: #000; color: #fff; font-weight: bold; margin-bottom: 9px;">📍 Open Maps</a>',
                            unsafe_allow_html=True)
            else:
                st.caption(gate_info)

        # Progress bar
        progress = get_flight_progress(flight)
        st.progress(int(progress))
        st.markdown(f"**Flight Progress:** {int(progress)}%")

        # Time information
        takeoff_time = get_formatted_time_string(flight['departure']['actual'])
        landing_time = get_formatted_time_string(
            flight['arrival']['actual'] or flight['arrival']['scheduled'],
            prefix="Scheduled: " if not flight['arrival']['actual'] else ""
        )
        st.info(f"**Take Off:** {takeoff_time} | **Landing:** {landing_time}")

        # Delay information
        delay = flight['departure']['delay']
        if delay and delay > 0:
            st.error(f":red[This flight was **delayed** by {delay} minutes.]")
        else:
            st.success("This flight appears to be **on time**.")

        st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APP ---

# Load passenger data
# passenger_df = load_excel_data()
passenger_df = load_excel_from_env()

# Create set of flight numbers (as requested)
flight_numbers_set = get_flight_numbers_set(passenger_df)

# --- MAIN PAGE CONTENT ---
st.title("First Love Church - Live Flight Tracker")

st.title("📊 Flight Statistics")

# Enhanced search with examples
st.markdown(
    "**Search Examples:** Try searching for names, flight numbers (ET921), dates (July 11, Monday), airports (ADD, ACC), or status (active, landed)")
search_query = st.text_input(
    "Search flights:",
    placeholder="Search by passenger name, flight number, date, airport, etc."
)

# Filter flights to only show those with passengers from Excel
relevant_flights = []
for flight in flight_data['data']:
    passengers = get_passengers_for_flight(
        flight['flight']['iata'],
        flight['flight_date'],
        passenger_df
    )
    if passengers:  # Only include flights with passengers from our Excel
        relevant_flights.append(flight)

print(relevant_flights, 'relevant_flights')

# Apply search filter
filtered_flights = search_flights(search_query, relevant_flights, passenger_df)
if not passenger_df.empty:
    # Left-aligned columns with metrics
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.markdown(f"<div style='text-align: left; color: #ffffff66'><strong>Total Passengers:</strong> {len(passenger_df)}</div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align: left; color: #ffffff66'><strong>Unique Flights:</strong> {len(flight_numbers_set)}</div>",
                    unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"<div style='text-align: left; color: #ffffff66'><strong>Active Flights:</strong> {len([f for f in relevant_flights if f['flight_status'] == 'active'])}</div>",
            unsafe_allow_html=True)
    with col4:
        st.markdown(
            f"<div style='text-align: left; color: #ffffff66'><strong>Landed Flights:</strong> {len([f for f in relevant_flights if f['flight_status'] == 'landed'])}</div>",
            unsafe_allow_html=True)

    # Show passenger distribution with searchable dropdown
    if 'arrival_info' in passenger_df.columns:
        st.markdown("### Passengers per Flight")

        # Get flight counts
        flight_counts = passenger_df['arrival_info'].str.split(',').explode().str.strip().value_counts()

        # Create searchable dropdown for flight selection
        flight_options = ['All Flights'] + list(flight_counts.index)
        selected_flight = st.selectbox(
            "Search and select a flight:",
            options=flight_options,
            index=0,
            help="Type to search for a specific flight number"
        )

        # Display results based on selection
        if selected_flight == 'All Flights':
            # Show all flights in a nicely formatted table
            st.markdown("**All Flight Passenger Counts:**")

            # Create a DataFrame for better display
            flight_data = []
            for flight, count in flight_counts.items():
                flight_data.append({
                    'Flight Number': flight,
                    'Passengers': count,
                    # 'Status': 'Multiple' if count > 1 else 'Single'
                })

            flight_df = pd.DataFrame(flight_data)

            # Display as a styled table
            st.dataframe(
                flight_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Flight Number': st.column_config.TextColumn(
                        'Flight Number',
                        width='medium'
                    ),
                    'Passengers': st.column_config.NumberColumn(
                        'Passengers',
                        width='small'
                    ),
                    'Status': st.column_config.TextColumn(
                        'Status',
                        width='small'
                    )
                }
            )
        else:
            # Show details for selected flight
            passenger_count = flight_counts[selected_flight]
            st.markdown(f"**{selected_flight}:** {passenger_count} passenger{'s' if passenger_count > 1 else ''}")

            # Show passenger details for this flight
            flight_passengers = passenger_df[passenger_df['arrival_info'].str.contains(selected_flight, na=False)]
            if not flight_passengers.empty:
                st.markdown("**Passenger Details:**")

                # Display available columns (excluding arrival_info since it's already shown)
                display_columns = [col for col in flight_passengers.columns if col != 'arrival_info']

                if display_columns:
                    st.dataframe(
                        flight_passengers[display_columns],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.dataframe(
                        flight_passengers,
                        use_container_width=True,
                        hide_index=True
                    )
# Display results
if not filtered_flights:
    if search_query:
        st.warning("No flights found matching your search criteria.")
    else:
        st.warning("No flights found with passengers from your Excel file.")
else:
    st.markdown(f"**Found {len(filtered_flights)} flight{'s' if len(filtered_flights) > 1 else ''}**")

    for flight in filtered_flights:
        display_flight_card(flight, passenger_df)
