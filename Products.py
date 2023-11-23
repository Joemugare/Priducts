import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Product Portfolio Dashboard",
    page_icon="📊",  # You can use an emoji or provide a URL to a favicon.ico
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your Streamlit app content goes here
st.title("Product Portfolio Dashboard")
st.write("Welcome to the Product Portfolio Dashboard.")

# Load your datasets
df_google = pd.read_csv('Google Products.csv')
df_apple = pd.read_csv('Apple Products.csv')
df_amazon = pd.read_csv('Amazon Products.csv')

# Add a column to identify the company
df_google['Company'] = 'Google'
df_apple['Company'] = 'Apple'
df_amazon['Company'] = 'Amazon'

# Concatenate the dataframes vertically
df = pd.concat([df_google, df_apple, df_amazon], ignore_index=True)

# Convert 'Launch Year' to numeric format
df['Launch Year'] = pd.to_datetime(df['Launch Year'], errors='coerce').dt.year.fillna(1900).astype(int)

# Unique values for filtering options
categories = df['What it does?'].unique()  # Replace 'What it does?' with the correct column name

# Sidebar for filtering options
st.sidebar.header("Filter Options")

selected_company = st.sidebar.selectbox("Select Company", df['Company'].unique())
selected_categories = st.sidebar.multiselect("Select Category", categories)
selected_years = st.sidebar.slider("Select Launch Year Range", df['Launch Year'].min(), 2023,
                                   (df['Launch Year'].min(), 2023), 1)  # Update the maximum value to 2023

# Filter the dataframe based on user selections
filtered_df = df[df['Company'] == selected_company]

if selected_categories:
    filtered_df = filtered_df[filtered_df['What it does?'].isin(selected_categories)]

filtered_df = filtered_df[
    (filtered_df['Launch Year'] >= selected_years[0]) & (filtered_df['Launch Year'] <= selected_years[1])
]

# Launch Trends Chart
st.header(f'Launch Trends - {selected_company}')
launch_trends_fig = px.line(
    filtered_df, x='Launch Year', y='Status', color='Product Name',
    title=f'Launch Trends - {selected_company}', hover_name='Product Name',
    labels={'Product Name': 'Product'}
)
launch_trends_fig.update_traces(
    mode='lines+markers',
    hovertemplate='<br>'.join([
        'Product: %{hovertext}',
        'Year: %{x}',
        'Status: %{y}'
    ])
)
st.plotly_chart(launch_trends_fig)

# Success Rates Chart
st.header(f'Success Rates - {selected_company}')
success_rates_fig = px.pie(
    filtered_df, names='Status', title=f'Success Rates - {selected_company}', color='Status',
    color_discrete_map={'Active': 'blue', 'Discontinued': 'red'},
    hover_data=['Company']
)
success_rates_fig.update_traces(
    hoverinfo='label+percent',
    textinfo='value+percent'
)
st.plotly_chart(success_rates_fig)

# Current Status Table
st.header(f'Current Status - {selected_company}')
current_status_fig = px.bar(
    filtered_df, x='Product Name', y='Launch Year', color='Status',
    title=f'Current Status - {selected_company}', labels={'Product Name': 'Product'},
    hover_name='Status'
)
current_status_fig.update_traces(
    hoverinfo='y+name'
)
st.plotly_chart(current_status_fig)
