# GitHub Copilot Instructions for Food Truck Route Optimizer

## Project Overview

This is a web-based food truck route optimization tool that helps food trucks and mobile businesses plan optimal routes for maximum street coverage. The application uses graph theory and routing algorithms to find paths that maximize street coverage within user-defined boundaries.

## Technology Stack

- **Backend**: Python 3.x with Flask 2.3.0 web framework
- **Routing Engine**: OSRM (Open Source Routing Machine)
- **Data Source**: OpenStreetMap via Overpass API
- **Graph Processing**: NetworkX 3.1
- **Frontend**: Leaflet.js for interactive mapping
- **Testing**: pytest 7.3.1 with pytest-mock
- **Production Server**: Waitress 2.1.2

## Code Style and Standards

### Python Guidelines

1. **Code Style**: Follow PEP 8 conventions
   - Use 4 spaces for indentation
   - Maximum line length of 100 characters
   - Use snake_case for functions and variables
   - Use descriptive variable names

2. **Imports**: Organize imports in the following order:
   - Standard library imports
   - Third-party library imports
   - Local application imports
   - Separate each group with a blank line

3. **Logging**: Use the Python logging module, not print statements
   - Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
   - Include context in log messages

4. **Error Handling**: 
   - Use try-except blocks for external API calls and file operations
   - Return meaningful error messages to users
   - Log errors with sufficient context for debugging

### Flask Application Conventions

1. **Routes**: Use clear, RESTful route names
2. **Request Handling**: Validate all user inputs
3. **Session Management**: Use Flask sessions for temporary data storage
4. **Environment Variables**: Use `os.getenv()` for configuration (e.g., API keys)
5. **Security**: Never commit secrets or API keys to the repository

### Graph Processing

1. **NetworkX**: Use NetworkX for all graph operations
2. **Node Representation**: Nodes are represented as coordinate tuples (latitude, longitude)
3. **Edge Weights**: Edge weights typically represent distance or traversal cost
4. **Graph Algorithms**: Prefer built-in NetworkX algorithms when available

### Testing Standards

1. **Framework**: Use pytest for all tests
2. **Test Files**: Place tests in the `tests/` directory with `test_` prefix
3. **Fixtures**: Use pytest fixtures for common test setup
4. **Mocking**: Use pytest-mock (mocker fixture) for external dependencies
5. **Coverage**: Write tests for:
   - Flask route handlers
   - Graph processing functions
   - API client functions
   - Utility functions

### File Organization

- `app.py`: Main Flask application with route handlers
- `find_route.py`: Route optimization algorithms
- `get_street_data.py`: OpenStreetMap data fetching
- `osrm_client.py`: OSRM routing service client
- `build_urls.py`: URL generation utilities
- `visualization.py`: Graph visualization functions
- `utils.py`: General utility functions
- `tests/`: All test files
- `templates/`: HTML templates for Flask
- `temp/`: Temporary data storage (not committed to git)

## Development Workflow

1. **Dependencies**: Install dependencies with `pip install -r requirements.txt`
2. **Testing**: Run tests with `pytest` from the repository root
3. **Running Locally**: Execute `python app.py` to start the development server
4. **Production**: Use Waitress for production deployment as shown in `deploy.sh`

## External Services

1. **OpenStreetMap Overpass API**: Used for fetching street network data
   - Be respectful of rate limits
   - Cache data when possible
   - Handle API errors gracefully

2. **OSRM**: Used for route calculation with turn-by-turn instructions
   - Ensure OSRM server is accessible
   - Handle connection errors

3. **Google Maps API**: Used for geocoding and map display
   - API key required via `GOOGLE_MAPS_API_KEY` environment variable
   - Check for API key presence before using

## Security Considerations

1. **Secrets**: Never commit API keys, passwords, or secrets
2. **Input Validation**: Validate all user inputs, especially boundary coordinates
3. **Session Keys**: Use secure, randomly generated session keys
4. **Dependencies**: Keep dependencies updated to avoid security vulnerabilities

## Common Patterns

### Processing User-Defined Boundaries

1. User submits ZIP code → Get coordinates via geocoding
2. User draws boundary on map → Receive 4 corner coordinates
3. Fetch street data from OpenStreetMap within boundaries
4. Build NetworkX graph from street data
5. Apply optimization algorithm to find best route
6. Generate navigation instructions and export options

### Graph Building and Optimization

1. Extract nodes (intersections) and ways (streets) from OSM data
2. Build NetworkX graph with coordinates as nodes
3. Apply simplification to reduce unnecessary nodes
4. Prune nodes using "common sense" rules
5. Run Chinese Postman Problem (CPP) or max coverage algorithm
6. Generate final route with waypoints

## Best Practices

1. **Error Messages**: Provide clear, actionable error messages to users
2. **Performance**: Consider caching for expensive operations (graph building, route optimization)
3. **User Experience**: Provide progress feedback for long-running operations
4. **Data Validation**: Validate coordinate boundaries and settings before processing
5. **Resource Cleanup**: Clean up temporary files after processing
6. **Documentation**: Update README.md when adding major features
7. **Testing**: Write tests for new functionality before considering it complete

## Things to Avoid

1. Don't use `print()` for logging - use the logging module
2. Don't commit temporary files in `temp/` directory
3. Don't hard-code API keys or secrets
4. Don't modify graph algorithms without understanding their mathematical basis
5. Don't break existing tests unless fixing bugs
6. Don't add dependencies without adding them to `requirements.txt`
