from fastapi import FastAPI
from fastapi.testclient import TestClient
from MAIN import app
client=TestClient(app)