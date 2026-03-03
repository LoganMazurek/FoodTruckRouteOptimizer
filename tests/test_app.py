import sys
import os
import pytest
import networkx as nx
from flask import session
import json
import pickle

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test_secret_key'
    with app.test_client() as client:
        with app.app_context():
            yield client

def test_home_get(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome" in response.data  # Assuming your index.html contains "Welcome"

def test_home_post(client, mocker):
    mocker.patch('app.get_coordinates', return_value=(41.8781, -87.6298))
    response = client.post('/', data={'zipcode': '60601'})
    assert response.status_code == 200
    assert b"Submit Boundaries" in response.data  # Check for a unique element or text in the HTML content

def test_home_post_invalid_zipcode(client, mocker):
    mocker.patch('app.get_coordinates', return_value=(None, None))
    response = client.post('/', data={'zipcode': '00000'})
    assert response.status_code == 500
    assert b"Failed to geocode ZIP code" in response.data

def test_process_boundaries_invalid_corners(client):
    response = client.post('/process_boundaries', json={"corners": [{"lat": 41.8781, "lng": -87.6298}]})
    assert response.status_code == 400
    assert b"Please select exactly 4 corners." in response.data

def test_process_boundaries_valid_corners(client, mocker):
    mocker.patch('app.fetch_overpass_data', return_value=mocker.Mock(nodes=[], ways=[]))
    mocker.patch('app.extract_nodes_and_ways', return_value=({}, []))
    mocker.patch('app.simplify_graph', return_value=nx.Graph())
    mocker.patch('app.clean_up_graph', return_value=nx.Graph())
    
    corners = [
        {"lat": 41.8781, "lng": -87.6298},
        {"lat": 41.8795, "lng": -87.6288},
        {"lat": 41.8798, "lng": -87.6292},
        {"lat": 41.8785, "lng": -87.6305}
    ]
    response = client.post('/process_boundaries', json={"corners": corners})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "boundary_id" in data

def test_result_no_street_data(client):
    boundary_id = '123e4567-e89b-12d3-a456-426614174000'
    response = client.get(f'/result?boundary_id={boundary_id}&start_node=0&end_node=1')
    assert response.status_code == 404
    assert b"Graph data not found" in response.data

def test_result_with_street_data(client, mocker):
    import uuid
    boundary_id = str(uuid.uuid4())
    
    # Create mock nodes and ways
    nodes = {
        1: (41.8781, -87.6298),
        2: (41.8795, -87.6288)
    }
    ways = [
        {"name": "Street 1", "nodes": [1, 2], "highway": "residential"}
    ]

    # Create the temporary file with the nodes/ways data
    temp_file_path = os.path.join("temp", f"{boundary_id}_nodes_ways.pkl")
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as temp_file:
        pickle.dump({"nodes": nodes, "ways": ways}, temp_file)

    # Create a mock graph
    mock_graph = nx.Graph()
    mock_graph.add_node(1, coordinates=(41.8781, -87.6298))
    mock_graph.add_node(2, coordinates=(41.8795, -87.6288))
    mock_graph.add_edge(1, 2, weight=1.0, distance=100, way_id="Street 1")

    mocker.patch('app.simplify_graph', return_value=mock_graph)
    mocker.patch('app.clean_up_graph', return_value=mock_graph)
    mocker.patch('app.find_route_max_coverage_optimized', return_value=[(41.8781, -87.6298), (41.8795, -87.6288)])
    mocker.patch('app.prune_common_sense_nodes', return_value=[(41.8781, -87.6298), (41.8795, -87.6288)])

    response = client.get(f'/result?boundary_id={boundary_id}&start_node=1&end_node=2')
    assert response.status_code == 200
    assert b"result.html" in response.data or b"route_variants" in response.data or len(response.data) > 0