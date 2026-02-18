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
- **Graph Processing**: NetworkX
- **Frontend**: Leaflet.js for interactive maps
- **Navigation**: GPX file export for mobile navigation apps

## Use Cases

- Food truck route planning for maximum market coverage
- Delivery service optimization
- Street survey and data collection routes

## Installation

### Prerequisites

- Python 3.8 or higher
- OSRM (Open Source Routing Machine) instance running locally or accessible remotely
- Google Maps API key (for coordinate lookup by ZIP code)

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

4. **Set environment variables**
   ```bash
   export GOOGLE_MAPS_API_KEY="your_api_key_here"
   ```
   On Windows (PowerShell):
   ```powershell
   $env:GOOGLE_MAPS_API_KEY="your_api_key_here"
   ```

5. **Ensure OSRM is running**
   - Set up OSRM with your desired map data (the project includes Illinois data)
   - OSRM should be accessible at `http://localhost:5000` (configure the URL in `osrm_client.py` if using a different address)

6. **Run the application**
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

## Troubleshooting

### Google Maps API "Request Denied" Errors

If you see `REQUEST_DENIED` errors when entering a ZIP code:

1. **Enable the Geocoding API**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/apis/library)
   - Search for "Geocoding API"
   - Click "Enable"

2. **Check API Key Validity**:
   - Go to [API Credentials](https://console.cloud.google.com/apis/credentials)
   - Verify your API key is active and not expired
   - Check that the key hasn't been accidentally deleted or regenerated

3. **Enable Billing**:
   - Google Maps APIs require a billing account even though they have a free tier
   - Go to [Billing](https://console.cloud.google.com/billing) in Google Cloud Console
   - Attach a billing account to your project

4. **Check API Key Restrictions**:
   - If you've set IP address restrictions, ensure your droplet's IP is allowed
   - For HTTP referrer restrictions, these don't apply to server-side requests
   - Application restrictions should be set to "None" or include your server's IP

5. **Monitor Quota Usage**:
   - Check your [API quotas](https://console.cloud.google.com/apis/api/geocoding-backend.googleapis.com/quotas)
   - The free tier includes $200/month in API credits

⚠️ **Security Note**: Never commit your API key to git. Use environment variables only. If your key is exposed in logs, regenerate it immediately.

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
