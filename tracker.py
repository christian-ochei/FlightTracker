

import streamlit as st
from datetime import datetime, timedelta
import pytz
import time

# --- DATA (As provided in the prompt) ---
# In a real app, this would come from an API call
null = None
import streamlit as st
from datetime import datetime, timedelta
import pytz

# --- MOCK DATA (As provided in the original script) ---
# In a real app, this would come from an API call
flight_data = {
    "pagination": {
        "limit": 100,
        "offset": 0,
        "count": 2,
        "total": 2
    },
    "data": [
        {
            "flight_date": "2025-07-11",
            "flight_status": "active",
            "departure": {
                "airport": "Bole International",
                "timezone": "Africa/Addis_Ababa",
                "iata": "ADD",
                "icao": "HAAB",
                "terminal": "2",
                "gate": "A10",
                "delay": 42,
                "scheduled": "2025-07-11T08:40:00+00:00",
                "estimated": "2025-07-11T08:40:00+00:00",
                "actual": "2025-07-11T09:22:00+00:00",
                "estimated_runway": "2025-07-11T09:22:00+00:00",
                "actual_runway": "2025-07-11T09:22:00+00:00"
            },
            "arrival": {
                "airport": "Kotoka",
                "timezone": "Africa/Accra",
                "iata": "ACC",
                "icao": "DGAA",
                "terminal": "3",
                "gate": null,
                "baggage": null,
                "scheduled": "2025-07-11T11:20:00+00:00",
                "delay": null,
                "estimated": "2025-07-11T11:09:00+00:00",
                "actual": null,
                "estimated_runway": null,
                "actual_runway": null
            },
            "airline": {
                "name": "Ethiopian Airlines",
                "iata": "ET",
                "icao": "ETH"
            },
            "flight": {
                "number": "921",
                "iata": "ET921",
                "icao": "ETH921",
                "codeshared": null
            },
            "aircraft": null,
            "live": null
        },
        {
            "flight_date": "2025-07-10",
            "flight_status": "landed",
            "departure": {
                "airport": "Bole International",
                "timezone": "Africa/Addis_Ababa",
                "iata": "ADD",
                "icao": "HAAB",
                "terminal": "2",
                "gate": "A10",
                "delay": 27,
                "scheduled": "2025-07-10T08:40:00+00:00",
                "estimated": "2025-07-10T08:40:00+00:00",
                "actual": "2025-07-10T09:06:00+00:00",
                "estimated_runway": "2025-07-10T09:06:00+00:00",
                "actual_runway": "2025-07-10T09:06:00+00:00"
            },
            "arrival": {
                "airport": "Kotoka",
                "timezone": "Africa/Accra",
                "iata": "ACC",
                "icao": "DGAA",
                "terminal": "3",
                "gate": null,
                "baggage": null,
                "scheduled": "2025-07-10T11:20:00+00:00",
                "delay": null,
                "estimated": "2025-07-10T10:54:00+00:00",
                "actual": "2025-07-10T10:55:00+00:00",
                "estimated_runway": "2025-07-10T10:55:00+00:00",
                "actual_runway": "2025-07-10T10:55:00+00:00"
            },
            "airline": {
                "name": "Ethiopian Airlines",
                "iata": "ET",
                "icao": "ETH"
            },
            "flight": {
                "number": "921",
                "iata": "ET921",
                "icao": "ETH921",
                "codeshared": null
            },
            "aircraft": null,
            "live": null
        }
    ]
}
#
# user_info = {
#     'QTR706': {'name': 'John Does'}
# }


# --- SESSION STATE INITIALIZATION ---
if 'flight_statuses' not in st.session_state:
    st.session_state.flight_statuses = {flight['flight']['iata']: flight['flight_status'] for flight in flight_data['data']}
if 'notifications' not in st.session_state:
    st.session_state.notifications = []

# --- UI & HELPER FUNCTIONS ---

def check_for_status_changes():
    """
    Monitors flight statuses and generates toast notifications on change.
    In a real app, you'd periodically fetch new data. Here, we simulate a change.
    """
    for flight in flight_data['data']:
        try:
            current_status = flight['flight_status']
            previous_status = st.session_state.flight_statuses[flight['flight']['iata']]

            if current_status != previous_status and flight['flight']['iata'] in st.session_state.get('subscribed_flights', []):
                user_name = flight['arriva']['airport']
                message = f"🚨 Status Change for {user_name}: Flight {flight['flight']['iata']} is now {current_status.upper()}!"
                st.toast(message, icon='✈️')
                st.session_state.notifications.append(f"{datetime.now().strftime('%H:%M:%S')}: {message}")
                st.session_state.flight_statuses[flight['flight']['iata']] = current_status
        except (IndexError, KeyError) as e:
            st.error(f"Error checking status changes: {e}")


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


from datetime import datetime, timezone
import pytz


def get_precise_landing_status(flight):
    """
    Calculates the precise landing status string using local time zone.
    Shows time until landing for future flights, or time since landing for completed flights.
    """
    status = flight['flight_status']

    # Get the current time in local timezone
    local_tz = datetime.now().astimezone().tzinfo
    now_local = datetime.now(local_tz)

    # If the flight has landed, calculate how long ago
    if status == 'landed' and flight['arrival']['actual']:
        landed_time_utc = datetime.fromisoformat(flight['arrival']['actual'])

        # Convert to local timezone if it's in UTC
        if landed_time_utc.tzinfo is None:
            landed_time_utc = landed_time_utc.replace(tzinfo=timezone.utc)

        landed_time_local = landed_time_utc.astimezone(local_tz)
        time_since_landing = now_local - landed_time_local

        # Handle negative time (shouldn't happen for landed flights, but just in case)
        if time_since_landing.total_seconds() < 0:
            return "Landing time inconsistent"

        # Breakdown the timedelta into days, hours, and minutes
        days = time_since_landing.days
        hours, remainder = divmod(int(time_since_landing.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)

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

    # If the flight hasn't landed yet, show time until landing
    elif flight['arrival']['scheduled']:
        scheduled_time_utc = datetime.fromisoformat(flight['arrival']['scheduled'])

        # Convert to local timezone if it's in UTC
        if scheduled_time_utc.tzinfo is None:
            scheduled_time_utc = scheduled_time_utc.replace(tzinfo=timezone.utc)

        scheduled_time_local = scheduled_time_utc.astimezone(local_tz)
        time_until_landing = scheduled_time_local - now_local

        # If the scheduled time has passed but flight hasn't landed
        if time_until_landing.total_seconds() < 0:
            time_overdue = now_local - scheduled_time_local
            days = time_overdue.days
            hours, remainder = divmod(int(time_overdue.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)

            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days > 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

            overdue_text = ', '.join(parts) if parts else "just now"
            return f"**Overdue:** {overdue_text} past scheduled landing"

        # Calculate time until landing
        days = time_until_landing.days
        hours, remainder = divmod(int(time_until_landing.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)

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

        # Show local time and date
        local_time_str = scheduled_time_local.strftime('%I:%M %p')
        local_date_str = scheduled_time_local.strftime('%b %d, %Y')

        return f"**Expected:** {time_text} at {local_time_str} on {local_date_str}"

    return "Landing time not available"


import streamlit as st
from datetime import datetime, timedelta
import pytz  # Required for timezone-aware datetime objects


def display_flight_card(flight, user_name):
    """
    Displays a single flight information card using Streamlit's native elements.
    """
    status = flight['flight_status']
    iata = flight['flight']['iata']
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
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(user_name)
            st.markdown(f"**{flight['airline']['name']}** - `{flight['flight']['iata']}`")
        with col2:
            st.markdown(f'<p class="status-text">{status.upper()}</p>', unsafe_allow_html=True)

        st.markdown("---")
        st.warning(get_precise_landing_status(flight))  # Assuming this helper function exists

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"**From:** `{flight['departure']['iata']}`")
            st.write(flight['departure']['airport'])
            st.caption(f"Gate: {flight['departure']['gate'] or 'N/A'}")
        with col2:
            st.image("travel_24dp_FFFFFF_FILL0_wght200_GRAD-25_opsz20.svg", width=30)
        with col3:
            st.markdown(f"**To:** `{flight['arrival']['iata']}`", )
            st.write(flight['arrival']['airport'])
            st.caption(f"Gate: {flight['arrival']['gate'] or 'N/A'}")

        progress = get_flight_progress(flight)  # Assuming this helper function exists
        st.progress(int(progress))
        st.markdown(f"**Flight Progress:** {int(progress)}%")

        # --- MODIFICATION START ---

        def get_formatted_time_string(dt_str, prefix=""):
            """
            Formats an ISO datetime string into a full date/time with a relative description.
            Example: "16 May 2028 05:30 PM (2 hours ago)"
            """
            if not dt_str:
                return "N/A"

            flight_time = datetime.fromisoformat(dt_str)
            now = datetime.now(pytz.utc)
            delta = now - flight_time

            # Format the absolute time, e.g., "16 May 2028 05:30 PM"
            absolute_time = flight_time.strftime('%d %b %Y %I:%M %p')

            # Determine the relative time
            tense = "ago" if delta.total_seconds() > 0 else "from now"
            seconds = abs(delta.total_seconds())

            if seconds < 3600:  # Less than an hour
                value = int(seconds / 60)
                unit = "minute"
            elif seconds < 86400:  # Less than a day
                value = int(seconds / 3600)
                unit = "hour"
            else:  # Days
                value = int(seconds / 86400)
                unit = "day"

            relative_time = f"{value} {unit}{'s' if value != 1 else ''} {tense}"

            return f"{prefix}{absolute_time} ({relative_time})"

        # Generate the formatted strings for takeoff and landing
        takeoff_time = get_formatted_time_string(flight['departure']['actual'])

        landing_time = get_formatted_time_string(
            flight['arrival']['actual'] or flight['arrival']['scheduled'],
            prefix="Scheduled: " if not flight['arrival']['actual'] else ""
        )

        st.info(f"**Take Off:** {takeoff_time} | **Landing:** {landing_time}")

        # --- MODIFICATION END ---

        delay = flight['departure']['delay']
        if delay and delay > 0:
            st.error(f":red[This flight was **delayed** by {delay} minutes.]")
        else:
            st.success("This flight appears to be **on time**.")

        st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APP ---
st.set_page_config(page_title="Live Flight Tracker", layout="wide", initial_sidebar_state="expanded")

# --- CHECK FOR NOTIFICATIONS ---
check_for_status_changes()

# --- SIDEBAR ---
# with st.sidebar:
#     st.header("Notification Manager")
#     st.write("Enable real-time alerts for status changes.")
#     subscribed_flights = st.multiselect(
#         "Subscribe to flights:",
#         options=list(user_info.keys()),
#         format_func=lambda iata: f"{user_info[iata]['name']} ({iata})",
#         key='subscribed_flights'
#     )
#     st.success(f"You are subscribed to {len(subscribed_flights)} flight(s).")
#
#     st.markdown("---")
#     st.subheader("Notification Log")
#     if st.session_state.notifications:
#         log_container = st.container()
#         for notif in reversed(st.session_state.notifications):
#             log_container.write(notif)
#     else:
#         st.write("No new notifications.")

# --- MAIN PAGE CONTENT ---
st.title("Live Flight Tracker 🚀")
search_query = st.text_input("Search for a user by name:", placeholder="Search by airport, delayed flights, names and so on")

# --- RENDER FLIGHT CARDS ---
# filtered_users = {iata: data for iata, data in user_info.items() if search_query.lower() in data['name'].lower()}


filtered_flights = [a for a in flight_data['data']]
if not filtered_flights:
    st.warning("No users found matching your search.")
else:
    # for iata_code, user_data in filtered_users.items():
        # flight = next((f for f in flight_data['data'] if f['flight']['iata'] == iata_code), None)
    for flight in filtered_flights:
        display_flight_card(flight, flight['flight']['iata'])
