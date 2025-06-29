Install dependencies:
pip install -r requirements.txt

Train the model (only once needed unless you change data):
python model_training.py

Run the API server:
uvicorn fraud_api:app --reload

Visit API docs at:
http://127.0.0.1:8000/docs

Test your POST /predict endpoint with input
