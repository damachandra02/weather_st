import os
import streamlit as st
import plotly.express as px
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from datetime import datetime

# Set page configuration
st.set_page_config(layout="wide", page_title="Taluk-level Temperature Dashboard")

# Define base directory and file paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get the script's absolute directory path
ROOT_DIR = os.path.join(BASE_DIR, '..')  # Go up one level to the repo root

# Define paths to the data files
shapefile_dir = os.path.join(ROOT_DIR, 'data', 'karnataka_shape_files_taluk_district', 'Taluk')
shapefile_path = os.path.join(shapefile_dir, 'Taluk.shp')
netcdf_path = os.path.join(ROOT_DIR, 'data', 'aifs_forecast_heat_stress.nc')

# Load data
@st.cache_data
def load_data():
    try:
        # Load the dataset and compute daily parameters
        # Try with no engine specified to let xarray choose automatically
        ds = xr.open_dataset(netcdf_path)
        
        # Compute daily aggregates
        min_2t = ds['2t'].resample(time='1D').min()
        max_2t = ds['2t'].resample(time='1D').max()
        avg_2t = ds['2t'].resample(time='1D').mean()
        avg_rh = ds['rh'].resample(time='1D').mean()
        max_hi = ds['hi'].resample(time='1D').max()
        avg_hi = ds['hi'].resample(time='1D').mean()
        dtr = max_2t - min_2t
        
        # Get time, latitude, and longitude arrays
        times = max_2t['time'].values
        lon = max_2t['longitude'].values
        lat = max_2t['latitude'].values
        
        # Create date options
        date_options = [pd.to_datetime(t).strftime('%b%d') for t in times]
        
        # Start date (first date in the dataset)
        start_date = pd.to_datetime(times[0])
        
        # Load taluk shapefile
        tal = gpd.read_file(shapefile_path)
        tal = tal.to_crs(epsg=4326)
        tal['geometry'] = tal['geometry'].simplify(tolerance=0.01)
        
        return {
            'min_2t': min_2t,
            'max_2t': max_2t,
            'avg_2t': avg_2t,
            'avg_rh': avg_rh,
            'max_hi': max_hi,
            'avg_hi': avg_hi,
            'dtr': dtr,
            'times': times,
            'lon': lon,
            'lat': lat,
            'date_options': date_options,
            'taluk_data': tal,
            'start_date': start_date
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"NetCDF path: {netcdf_path}")
        st.error(f"Shapefile path: {shapefile_path}")
        return None

# Main app
def main():
    # Title
    st.title("Taluk-level Aggregated Temperature Dashboard")
    
    # Check if data files exist
    if not os.path.exists(netcdf_path):
        st.error(f"NetCDF file not found at: {netcdf_path}")
        st.info("Please ensure your data files are correctly placed in the repository.")
        return
        
    if not os.path.exists(shapefile_path):
        st.error(f"Shapefile not found at: {shapefile_path}")
        st.info("Please ensure your shapefile directory is correctly placed in the repository.")
        return
    
    # Load data
    with st.spinner("Loading data... This may take a moment."):
        data = load_data()
        
    if data is None:
        return
    
    # Create sidebar for controls
    st.sidebar.header("Controls")
    
    # Date selector
    st.sidebar.subheader("Select Date")
    date_idx = st.sidebar.select_slider(
        "Date",
        options=range(len(data['date_options'])),
        format_func=lambda i: data['date_options'][i]
    )
    
    # Calculate and display lead days
    selected_date = pd.to_datetime(data['times'][date_idx])
    lead_days = (selected_date - data['start_date']).days
    st.sidebar.info(f"Lead Days: {lead_days} days from start date ({data['start_date'].strftime('%b %d, %Y')})")
    
    # Variable selector - changed to radio buttons for single selection
    st.sidebar.subheader("Select Variable")
    variables = {
        'max_2t': "Max Temperature",
        'min_2t': "Min Temperature", 
        'avg_2t': "Avg Temperature",
        'avg_rh': "Avg Relative Humidity",
        'max_hi': "Max Heat Index",
        'avg_hi': "Avg Heat Index",
        'dtr': "Diurnal Temp Range"
    }
    
    selected_var = st.sidebar.radio(
        "Variable",
        options=list(variables.keys()),
        format_func=lambda x: variables[x]
    )
    
    # Main content
    st.subheader(f"{variables[selected_var]} on {data['date_options'][date_idx]}")
    
    # Get data for the selected variable and date
    if selected_var == 'max_2t':
        da = data['max_2t'].isel(time=date_idx)
    elif selected_var == 'min_2t':
        da = data['min_2t'].isel(time=date_idx)
    elif selected_var == 'avg_2t':
        da = data['avg_2t'].isel(time=date_idx)
    elif selected_var == 'avg_rh':
        da = data['avg_rh'].isel(time=date_idx)
    elif selected_var == 'max_hi':
        da = data['max_hi'].isel(time=date_idx)
    elif selected_var == 'avg_hi':
        da = data['avg_hi'].isel(time=date_idx)
    elif selected_var == 'dtr':
        da = data['dtr'].isel(time=date_idx)

    # Create a meshgrid and flatten the data into a DataFrame
    long_grid, lat_grid = np.meshgrid(data['lon'], data['lat'])
    df = pd.DataFrame({
        'lon': long_grid.flatten(),
        'lat': lat_grid.flatten(),
        'value': da.values.flatten()
    }).dropna(subset=['value'])
    
    # Convert the DataFrame to a GeoDataFrame with point geometries
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs='EPSG:4326'
    )
    
    # Spatially join the points with the taluk polygons
    tal = data['taluk_data']
    joined = gpd.sjoin(gdf_points, tal, how='inner', predicate='within')
    agg = joined.groupby('KGISTalukN')['value'].mean()
    tal_agg = tal.merge(agg, left_on='KGISTalukN', right_index=True)
    tal_agg['id'] = tal_agg['KGISTalukN']
    geojson_data = tal_agg.__geo_interface__
    
    # Create a choropleth map using Plotly Express
    fig = px.choropleth(
        tal_agg,
        geojson=geojson_data,
        locations='id',
        color='value',
        featureidkey="properties.KGISTalukN",
        color_continuous_scale='Spectral_r',
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(marker_line_width=0)
    
    # Update layout to make map larger - 1500px by 1500px as requested
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove margins
        autosize=True,  # Automatically adjust size
        height=800,  # Increase height
        width=1400,   # Increase width to match screen width
        font=dict(size=20)
    )

    
    # Update colorbar
    fig.update_coloraxes(colorbar=dict(
        title=variables[selected_var],
        title_font=dict(size=20),
        tickfont=dict(size=18),
        len=0.5  # Make colorbar longer
    ))
    
    # Display the map
    st.plotly_chart(fig, use_container_width=False)  # Changed to use_container_width=False to respect exact dimensions

if __name__ == "__main__":
    main()