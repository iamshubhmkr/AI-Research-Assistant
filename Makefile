.PHONY: setup up test lint eval run ui clean

setup:           ## venv + deps
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

up:              ## start Redis + PostgreSQL
	docker-compose up -d && docker-compose ps

test:            ## unit tests
	python -m pytest tests/unit/ -v

lint:            ## ruff lint
	ruff check .

eval:            ## quick RAGAS smoke evaluation
	python -m evaluation.ragas_eval --quick

run:             ## start API (reload)
	uvicorn api.main:app --reload --port 8000

ui:              ## start Streamlit UI
	streamlit run ui/app.py

clean:           ## remove local data
	rm -rf data/ __pycache__ .pytest_cache
