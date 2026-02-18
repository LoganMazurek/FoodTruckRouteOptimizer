# Food Truck Route Optimizer

A web-based tool designed to help food trucks and mobile businesses plan optimal routes that maximize street coverage and customer exposure. Given a geographic area, it generates efficient routes that traverse every street or intersection within specified boundaries.

## Features

- **Interactive Map Interface**: Select coverage areas by drawing boundaries on an interactive map
- **Intelligent Route Optimization**: Uses graph theory and routing algorithms to find paths that maximize street coverage  
- **Multiple Export Options**: 
  - Export routes as GPX files for navigation apps (OsmAnd, Organic Maps, Komoot, Garmin)
- **Customizable Settings**: Adjust route parameters and pruning thresholds for optimal coverage
- **Real-time Visualization**: View and edit the street network graph before optimization
- **Comprehensive Navigation Data**: Access distance, duration, and turn-by-turn instructions

## How It Works

1. Enter a location by ZIP code then draw a custom boundary on the map
2. The tool fetches detailed street data using OpenStreetMap
3. A graph is constructed from the street network
4. An optimization algorithm finds the optimal route that covers maximum streets
5. Routes are generated with navigation guidance and can be exported as GPX files

## Technology Stack

- **Backend**: Python Flask web framework
- **Routing**: OSRM (Open Source Routing Machine)
- **Street Data**: OpenStreetMap via Overpass API
- **Geocoding**: Nominatim (OpenStreetMap's free geocoding service)
- **Graph Processing**: NetworkX
- **Frontend**: Leaflet.js for interactive maps with OpenStreetMap tiles
- **Navigation**: GPX file export for mobile navigation apps

## Use Cases

- Food truck route planning for maximum market coverage
- Delivery service optimization
- Street survey and data collection routes

## Installation

### Prerequisites

- Python 3.8 or higher
- OSRM (Open Source Routing Machine) instance running locally or accessible remotely

**No API keys required!** This application uses free, open-source services:
- **Nominatim** for geocoding (converting ZIP codes to coordinates)
- **OpenStreetMap** for map tiles and street data
- **OSRM** for routing calculations

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/LoganMazurek/FoodTruckRouteOptimizer.git
   cd FoodTruckRouteOptimizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure OSRM is running**
   - Set up OSRM with your desired map data (the project includes Illinois data)
   - OSRM should be accessible at `http://localhost:5000` (configure the URL in `osrm_client.py` if using a different address)

5. **Run the application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5001`

### Configuration

- **OSRM Server**: Update the OSRM endpoint in `osrm_client.py` if running on a different host/port
- **Map Data**: The project includes Illinois street data. To use different regions, replace the `.osm.pbf` file in the `data/` directory

### First Run

1. Navigate to `http://localhost:5001` in your browser
2. Enter a ZIP code to start planning a route
3. Draw a boundary on the map to define your coverage area
4. Review the optimized route and export as GPX for navigation apps

## Why No API Keys?

This project is designed to be completely free and open:
- **Nominatim** provides free geocoding (respecting usage limits)
- **OpenStreetMap** provides free map tiles and data
- **Overpass API** provides free street data queries
- **OSRM** is self-hosted for routing

No credit cards, billing accounts, or API keys required! 🎉

## Troubleshooting

### Geocoding Issues (ZIP Code Not Found)

If geocoding fails when entering a ZIP code:

1. **Nominatim Rate Limiting**: The free Nominatim service has usage limits
   - Wait a few seconds between requests
   - For heavy usage, consider running your own Nominatim instance

2. **ZIP Code Format**: 
   - Use standard US ZIP codes (e.g., "60504")
   - For other countries, use city names or postal codes in the format "City, Country"

3. **Network Issues**: 
   - Ensure your server can reach `https://nominatim.openstreetmap.org`
   - Check firewall settings if running on a restricted network

### Overpass API "Request Denied" Errors

If you encounter "request denied" errors from the Overpass API:

1. **Rate Limiting**: The Overpass API has rate limits. The application now includes:
   - Automatic retry logic with exponential backoff
   - Proper User-Agent headers to identify the application
   - Better error handling and logging

2. **Best Practices**:
   - Avoid making too many requests in a short time period
   - Consider setting up your own Overpass API instance for heavy usage
   - Monitor the logs for rate limiting warnings

3. **Check Logs**: The application logs detailed information about API requests. Check your server logs for specific error messages.

4. **Public vs Private Instance**: For production deployments with high traffic, consider running your own Overpass API instance rather than relying on the public endpoint.
