.PHONY: dev prod test lint migrate clean

dev:
	docker compose up --build

prod:
	docker compose -f docker-compose.prod.yml up --build -d

test:
	cd backend && pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	cd backend && ruff check . && mypy app/
	cd frontend && npm run lint && tsc --noEmit

migrate:
	cd backend && alembic upgrade head

clean:
	docker compose down -v
	rm -rf data/storage data/mlflow data/*.db data/checkpoints.db
