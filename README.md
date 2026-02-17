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
