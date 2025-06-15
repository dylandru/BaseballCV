import streamlit as st
import datetime

PITCH_TYPES = {
    "Four-Seam Fastball": "FF", "Sinker": "SI", "Cutter": "FC", "Curveball": "CU", 
    "Slider": "SL", "Changeup": "CH", "Split-Finger": "FS", "Knuckleball": "KN"
}
PA_RESULTS = {
    "Single": "single", "Double": "double", "Triple": "triple", "Home Run": "home_run",
    "Walk": "walk", "Strikeout": "strikeout", "Field Out": "field_out", 
    "Hit By Pitch": "hit_by_pitch", "Sac Fly": "sac_fly", "Sac Bunt": "sac_bunt",
    "Fielders Choice": "fielders_choice", "Double Play": "double_play"
}
TEAMS = {
    'Angels': '108', 'Astros': '117', 'Athletics': '133', 'Blue Jays': '141',
    'Braves': '144', 'Brewers': '158', 'Cardinals': '138', 'Cubs': '112',
    'Diamondbacks': '109', 'Dodgers': '119', 'Giants': '137', 'Guardians': '114',
    'Mariners': '136', 'Marlins': '146', 'Mets': '121', 'Nationals': '120',
    'Orioles': '110', 'Padres': '135', 'Phillies': '143', 'Pirates': '134',
    'Rangers': '140', 'Rays': '139', 'Red Sox': '111', 'Reds': '113',
    'Rockies': '115', 'Royals': '118', 'Tigers': '116', 'Twins': '142',
    'White Sox': '145', 'Yankees': '147'
}
METRIC_FILTERS = {
    "Exit Velocity (mph)": {"param": "api_h_launch_speed", "min": 0, "max": 125},
    "Launch Angle (°)": {"param": "api_h_launch_angle", "min": -90, "max": 90},
    "Distance Projected (ft)": {"param": "api_h_distance_projected", "min": 0, "max": 550},
    "Bat Speed (mph)": {"param": "sweetspot_speed_mph", "min": 0, "max": 100},
    "Arm Angle (°)": {"param": "arm_angle", "min": -90, "max": 90},
    "IVB (in)": {"param": "api_break_z_induced", "min": -30, "max": 30},
    "HB - Glove Side (in)": {"param": "api_break_x_glove", "min": -30, "max": 30}
}

def display_search_interface(player_df):
    st.sidebar.header("Search Options")
    query_mode = st.sidebar.radio("Query Mode", ('Search by Filters', 'Search by Specific Play ID'))
    if query_mode == 'Search by Filters':
        return 'filters', display_search_filters(player_df)
    else:
        return 'play_id', display_play_id_search()

def display_search_filters(player_df):
    params = {}
    player_names = sorted(player_df['name'].unique()) if not player_df.empty else []

    st.sidebar.markdown("##### Date Range")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=2)
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)
    params['game_date_gt'] = [start_date.strftime('%Y-%m-%d')]
    params['game_date_lt'] = [end_date.strftime('%Y-%m-%d')]
    
    st.sidebar.markdown("##### Player & Team")
    selected_pitchers = st.sidebar.multiselect("Pitcher(s)", player_names)
    selected_batters = st.sidebar.multiselect("Batter(s)", player_names)
    pitcher_ids = [int(player_df[player_df['name'] == name].iloc[0]['id']) for name in selected_pitchers]
    batter_ids = [int(player_df[player_df['name'] == name].iloc[0]['id']) for name in selected_batters]
    if pitcher_ids:
        params['pitchers_lookup[]'] = pitcher_ids
    if batter_ids:
        params['batters_lookup[]'] = batter_ids

    st.sidebar.markdown("##### Pitch, PA & Team")
    params['hfPT'] = [PITCH_TYPES[p] for p in st.sidebar.multiselect("Pitch Type(s)", list(PITCH_TYPES.keys()))]
    params['hfAB'] = [PA_RESULTS[p] for p in st.sidebar.multiselect("PA Result(s)", list(PA_RESULTS.keys()))]
    params['hfTeam'] = [TEAMS[t] for t in st.sidebar.multiselect("Team(s)", list(TEAMS.keys()))]
    
    st.sidebar.markdown("##### Advanced Metric Filters")
    selected_metrics = st.sidebar.multiselect("Select up to 6 metrics", options=list(METRIC_FILTERS.keys()), max_selections=6, key="metric_selector")
    
    metric_counter = 1
    for metric_name in selected_metrics:
        metric_info = METRIC_FILTERS[metric_name]
        min_val, max_val = metric_info["min"], metric_info["max"]
        val_range = st.sidebar.slider(f"{metric_name}", min_value=min_val, max_value=max_val, value=(min_val, max_val), key=f"slider_{metric_info['param']}")
        if val_range != (min_val, max_val):
            params[f'metric_{metric_counter}'] = [metric_info['param']]
            params[f'metric_{metric_counter}_gt'] = [val_range[0]]
            params[f'metric_{metric_counter}_lt'] = [val_range[1]]
            metric_counter += 1

    st.sidebar.markdown("##### Other")
    params['player_type'] = [st.sidebar.selectbox("Primary Player Type", ["pitcher", "batter"], index=0)]
    max_results = st.sidebar.slider("Max Results to Fetch", 1, 500, 50)

    return params, max_results, start_date, end_date

def display_play_id_search():
    st.sidebar.markdown("##### Enter Play Identifiers")
    game_pk = st.sidebar.number_input("Game PK", step=1, value=None)
    at_bat_number = st.sidebar.number_input("At Bat Number", step=1, value=None)
    pitch_number = st.sidebar.number_input("Pitch Number", step=1, value=None)
    return game_pk, at_bat_number, pitch_number, None, None