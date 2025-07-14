.PHONY: build up down logs shell

build:
	docker-compose build

up:
	docker-compose up -d
	@echo " App: http://localhost:5000"

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec ml-ats-app /bin/bash

test:
	@curl -f http://localhost:5000/health && echo "✅ OK" || echo "❌ Falha"