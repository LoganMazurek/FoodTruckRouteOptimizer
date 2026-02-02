import sys
import os
import pytest
import networkx as nx
from flask import session
import json

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
    assert b"Failed to get coords for ZIP code" in response.data

def test_process_boundaries_invalid_corners(client):
    response = client.post('/process_boundaries', json={"corners": [{"lat": 41.8781, "lng": -87.6298}]})
    assert response.status_code == 400
    assert b"Please select exactly 4 corners." in response.data

def test_process_boundaries_valid_corners(client, mocker):
    mocker.patch('app.request_street_data', return_value={"Street 1": [(41.8781, -87.6298), (41.8795, -87.6288)]})
    corners = [
        {"lat": 41.8781, "lng": -87.6298},
        {"lat": 41.8795, "lng": -87.6288},
        {"lat": 41.8798, "lng": -87.6292},
        {"lat": 41.8785, "lng": -87.6305}
    ]
    response = client.post('/process_boundaries', json={"corners": corners})
    assert response.status_code == 200
    assert b"Boundaries processed successfully." in response.data

def test_result_no_street_data(client):
    with client.session_transaction() as sess:
        sess['boundary_id'] = 'test_id'

    response = client.get('/result?boundary_id=test_id')
    assert response.status_code == 500
    assert b"No street data found" in response.data

def test_result_with_street_data(client, mocker):
    boundary_id = 'test_id'
    street_data = {"Street 1": [(41.8781, -87.6298), (41.8795, -87.6288)]}

    # Create the temporary file with the street data
    temp_file_path = os.path.join("temp", f"{boundary_id}.json")
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "w") as temp_file:
        json.dump(street_data, temp_file)

    with client.session_transaction() as sess:
        sess['boundary_id'] = boundary_id

    # Create a mock graph
    mock_graph = nx.Graph()
    mock_graph.add_node((41.8781, -87.6298), coordinates=(41.8781, -87.6298))
    mock_graph.add_node((41.8795, -87.6288), coordinates=(41.8795, -87.6288))
    mock_graph.add_edge((41.8781, -87.6298), (41.8795, -87.6288), weight=1.0)

    mocker.patch('app.build_graph_from_maps_data', return_value=(mock_graph, []))
    mocker.patch('app.find_route', return_value=[(41.8781, -87.6298), (41.8795, -87.6288)])
    mocker.patch('app.create_google_maps_url', return_value="http://maps.google.com")
    mocker.patch('visualization.show_graph_in_main_thread')
    mocker.patch('visualization.animate_cpp_matching')
    mocker.patch('visualization.animate_cpp_route')

    response = client.get('/result?boundary_id=test_id')
    assert response.status_code == 200
    from urllib.parse import urlparse
    response_text = response.data.decode('utf-8')
    parsed_url = urlparse(response_text)
    assert parsed_url.hostname == "maps.google.com"