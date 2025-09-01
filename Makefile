IMAGE := us-east1-docker.pkg.dev/optimal-aegis-470701-j8/containers/nightout-backend:latest

.PHONY: run docker-create docker-build deploy

run:
	uvicorn application:app --host 0.0.0.0 --port 8000

docker-create:
	docker buildx create --use --name ar_builder || docker buildx use ar_builder

docker-build:
	docker buildx build --platform linux/amd64 -t $(IMAGE) --push .

deploy:
	gcloud run deploy nightout-backend --image $(IMAGE) --region us-east1 --allow-unauthenticated --quiet