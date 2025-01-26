import streamlit as st
import random
import pandas as pd

df = pd.read_excel("Demo-Hygine Data.xlsx")
df["Catalog Score"] = (df['Ratings'] >= 4).astype(int) + (df['Title Length'] >= 180).astype(int) + (df['Bullet Point Count'] > 5).astype(int) + (df['Images Count'] >= 7).astype(int) + (df['A+'] == 'Yes').astype(int)
df["Catalog Score"] = df["Catalog Score"]*20
df["Date"] = pd.to_datetime(df["Date"])

df['Total Ratings'] = (
    df['Total Ratings']
      .fillna('')                 # So NaN becomes an empty string
      .astype(str)                # Convert numbers to string
      .str.extract(r'(\d+)')[0]   # Extract only digits
      .astype(float)              # Convert back to float
)

def calculate_hygiene_metrics(df):
    """
    Calculate hygiene metrics for each row in the dataset
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the required columns
    
    Returns:
    pandas.DataFrame: Original DataFrame with additional hygiene metrics columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # 1. Activation Hygiene (100% if both validations are True, 0% if either is False)
    result_df['Activation_Hygiene'] = ((result_df['SNS Validation'] == True) & 
                                     (result_df['BXGY Validation'] == True)) * 100
    
    result_df['Price_Hygiene'] = (result_df['Price Validation'] == True) * 100
    
    # 2. EDD Hygiene (100% if EDD ≤ 2, 0% if EDD > 2)
    result_df['EDD_Hygiene'] = (result_df['EDD'] <= 2) * 100
    
    # 3. Catalog Hygiene (use raw Catalog Score for each row)
    result_df['Catalog_Hygiene'] = result_df['Catalog Score']
    
    # 4. Rating Hygiene (divide by 5 and multiply by 100 for percentage)
    result_df['Rating_Hygiene'] = (result_df['Ratings'] / 5) * 100
    
    # 5. Availability Hygiene (100% if Available, 0% if Not Available)
    result_df['Availability_Hygiene'] = (result_df['Availability'] == 'Yes') * 100
    
    # 6. Deal Hygiene (100% if Coupon Validation is True, 0% if False)
    result_df['Deal_Hygiene'] = result_df['Coupon Validation'] * 100
    
    # 7. Overall Brand Score
    result_df['Overall_Brand_Score'] = (
        (result_df['Price_Hygiene'].fillna(0) * 0.2) +
        (result_df['Activation_Hygiene'].fillna(0) * 0.05) +
        (result_df['Deal_Hygiene'].fillna(0) * 0.05) +
        (result_df['Availability_Hygiene'].fillna(0) * 0.2) +
        (result_df['EDD_Hygiene'].fillna(0) * 0.1) +
        (result_df['Rating_Hygiene'].fillna(0) * 0.2) +
        (result_df['Catalog_Hygiene'].fillna(0) * 0.2)
    )   # Divide by 100 to get final percentage

    return result_df

df['EDD_400013_Score'] = (df['EDD_400013'] <= 2).astype("int")
df['EDD_600005_Score'] = (df['EDD_600005'] <= 2).astype("int")
df['EDD_122102_Score'] = (df['EDD_122102'] <= 2).astype("int")
df['EDD_700016_Score'] = (df['EDD_700016'] <= 2).astype("int")
df['EDD_560068_Score'] = (df['EDD_560068'] <= 2).astype("int")

df['EDD'] = (df['EDD_400013_Score'] + df['EDD_600005_Score'] + df['EDD_122102_Score'] + df['EDD_700016_Score'] + df['EDD_560068_Score'])*20

df = calculate_hygiene_metrics(df)

st.set_page_config(initial_sidebar_state="collapsed")


# Hide menu & footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def force_rerun():
    # Change the URL query parameters to force a rerun
    st.query_params = {"rerun": str(random.random())}

def show_login():
    st.title("Login Page")
    valid_users = {"alice":"password123", "bob":"qwerty"}

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in valid_users and valid_users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["page"] = "brand"
            force_rerun()
        else:
            st.error("Invalid username or password.")

def show_brand():
    if not st.session_state.get("logged_in"):
        st.warning("Please login first.")
        st.session_state["page"] = "login"
        force_rerun()
        return

    st.title("Choose Brand")
    brands = ["Oshea", "Origami", "Harissons"]
    choice = st.selectbox("Select brand:", brands)

    if st.button("Next"):
        st.session_state["selected_brand"] = choice
        st.session_state["page"] = "analytics"
        force_rerun()

def show_analytics():
    # Guard checks
    if not st.session_state.get("logged_in"):
        st.warning("Please log in first.")
        st.session_state["page"] = "login"
        force_rerun()
        return
    if "selected_brand" not in st.session_state:
        st.warning("Please select a brand first.")
        st.session_state["page"] = "brand_selection"
        force_rerun()
        return

    # Show the sidebar only on this page
    st.sidebar.header("Analytics Sections")
    mode = st.sidebar.radio("View mode:", ["Summary", "Drill Down"])

    st.title(f"Analytics for {st.session_state['selected_brand']}")

    if mode == "Summary":
        show_summary_tab()
    else:
        show_drilldown_tab()

def show_summary_tab():
    st.subheader("Summary View")

    # Let the user pick a sub‐category
    categories = df["Sub-category"].dropna().unique()
    selected_cat = st.selectbox("Select a category:", categories)

    # Filter to just that sub‐category
    filtered = df[df["Sub-category"] == selected_cat].copy()

    # Sort the unique dates
    unique_dates = sorted(filtered["Date"].unique())
    if len(unique_dates) == 0:
        st.write("No data available for this category.")
        return

    # Always take the latest date
    latest_date = unique_dates[-1]

    # Take the second-latest date if it exists, otherwise None
    if len(unique_dates) >= 2:
        prev_date = unique_dates[-2]
    else:
        prev_date = None

    # List of hygiene metrics
    metrics = [
        "Activation_Hygiene", "Price_Hygiene", "EDD_Hygiene",
        "Catalog_Hygiene", "Rating_Hygiene", "Availability_Hygiene",
        "Deal_Hygiene", "Overall_Brand_Score"
    ]

    # Average for the latest date
    latest_means = filtered[filtered["Date"] == latest_date][metrics].mean()

    # Average for the previous date (if present)
    prev_means = (filtered[filtered["Date"] == prev_date][metrics].mean()
                  if prev_date else None)

    # Helper: display a single metric in percentage form
    def display_metric(col, label, current_val, prev_val):
        # If current is NaN, treat it as 0 for display
        if pd.isna(current_val):
            current_val = 0

        # If there's no previous date or it's NaN, we show no delta
        if (prev_val is None) or pd.isna(prev_val):
            delta_str = ""
        else:
            diff = current_val - prev_val
            delta_str = f"{diff:.1f}%"

        col.metric(
            label,
            f"{current_val:.1f}%",
            delta_str
        )

    # Layout columns (8 metrics -> 2 rows of 4)
    col1, col2, col3, col4 = st.columns(4)
    display_metric(col1, "Activation",   latest_means["Activation_Hygiene"],  
                   None if prev_means is None else prev_means["Activation_Hygiene"])
    display_metric(col2, "Price",        latest_means["Price_Hygiene"],       
                   None if prev_means is None else prev_means["Price_Hygiene"])
    display_metric(col3, "EDD",          latest_means["EDD_Hygiene"],         
                   None if prev_means is None else prev_means["EDD_Hygiene"])
    display_metric(col4, "Catalog",      latest_means["Catalog_Hygiene"],     
                   None if prev_means is None else prev_means["Catalog_Hygiene"])

    col5, col6, col7, col8 = st.columns(4)
    display_metric(col5, "Rating",       latest_means["Rating_Hygiene"],      
                   None if prev_means is None else prev_means["Rating_Hygiene"])
    display_metric(col6, "Availability", latest_means["Availability_Hygiene"],
                   None if prev_means is None else prev_means["Availability_Hygiene"])
    display_metric(col7, "Deal",         latest_means["Deal_Hygiene"],        
                   None if prev_means is None else prev_means["Deal_Hygiene"])
    display_metric(col8, "Overall Score",latest_means["Overall_Brand_Score"], 
                   None if prev_means is None else prev_means["Overall_Brand_Score"])

    # Finally, plot a trend line of Overall_Brand_Score vs. Date
    st.markdown("### Overall Brand Score Trend")
    trend_df = (
        filtered.groupby("Date", as_index=False)["Overall_Brand_Score"]
                .mean()
    )
    trend_df.set_index("Date", inplace=True)
    st.line_chart(trend_df["Overall_Brand_Score"])

import streamlit as st
import pandas as pd
import numpy as np

def show_drilldown_tab():
    st.subheader("Drill Down View")

    # 8 possible indicators
    indicator_list = [
        "Activation_Hygiene",
        "Price_Hygiene",
        "EDD_Hygiene",
        "Catalog_Hygiene",
        "Rating_Hygiene",
        "Availability_Hygiene",
        "Deal_Hygiene",
        "Overall_Brand_Score"
    ]
    
    # For each indicator, specify which "extra" columns to display
    indicator_columns = {
        "Activation_Hygiene": ['SNS Rule', 'Live SNS',
       'SNS Validation', 'BXGY Rule', 'Live BXGY', 'BXGY Validation'],
        "Price_Hygiene":      ['Price Rule', 'Live Price', 'Price Validation'],
        "EDD_Hygiene":       [ 'EDD_400013', 'EDD_600005', 'EDD_122102',
       'EDD_700016', 'EDD_560068'],
        "Catalog_Hygiene":    ['Ratings', 'Title Length','Bullet Point Count','Images Count', 'A+'],
        "Rating_Hygiene":     ['3 Star Ratings', '2 Star Ratings',
       '1 Star Ratings', 'Total Ratings', 'Ratings'],
        "Availability_Hygiene": ['Availability'],
        "Deal_Hygiene":       ['Coupon Rule',
       'Live Coupon', 'Coupon Validation'],
        "Overall_Brand_Score": ['Activation_Hygiene', 'Price_Hygiene', 'EDD_Hygiene', 'Catalog_Hygiene',
       'Rating_Hygiene', 'Availability_Hygiene', 'Deal_Hygiene']
    }

    # Category selection
    categories = df["Sub-category"].dropna().unique()
    cat_filter = st.selectbox("Category Filter", ["All"] + list(categories))

    # Indicator selection
    chosen_indicator = st.selectbox("Indicator", indicator_list)

    # Date selection (with “All” option)
    all_dates = sorted(df["Date"].dropna().unique())
    date_choice = st.selectbox("Select a Date", ["All"] + list(all_dates))

    # Range slider for the chosen indicator
    valid_vals = df[chosen_indicator].dropna()
    if valid_vals.empty:
        low, high = 0, 100
    else:
        low, high = float(valid_vals.min()), float(valid_vals.max())
    threshold_range = st.slider("Threshold Range", low, high, (low, high))

    # --- Filtering ---
    filtered_df = df.copy()

    if cat_filter != "All":
        filtered_df = filtered_df[filtered_df["Sub-category"] == cat_filter]
    if date_choice != "All":
        filtered_df = filtered_df[filtered_df["Date"] == date_choice]

    # Keep rows whose chosen_indicator is within the threshold range
    mask = filtered_df[chosen_indicator].between(*threshold_range, inclusive="both")
    filtered_df = filtered_df[mask]

    st.markdown("### Filtered Results")

    # Base columns you always want to see:
    base_cols = ["ASIN", "Product", chosen_indicator]
    # Add in the extra columns for the chosen indicator
    extra_cols = indicator_columns.get(chosen_indicator, [])

    # Combine & only keep columns that actually exist in df
    columns_to_show = base_cols + extra_cols
    final_cols = [c for c in columns_to_show if c in filtered_df.columns]

    st.dataframe(filtered_df[final_cols])


# -----------------------
# Main Router
# -----------------------
if "page" not in st.session_state:
    st.session_state["page"] = "login"

page = st.session_state["page"]
if page == "login":
    show_login()
elif page == "brand":
    show_brand()
elif page == "analytics":
    show_analytics()
