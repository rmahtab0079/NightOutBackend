IMAGE := us-east1-docker.pkg.dev/optimal-aegis-470701-j8/containers/nightout-backend:latest
PROJECT ?= optimal-aegis-470701-j8
REGION ?= us-east1
SERVICE ?= nightout-backend
DOTENV_SECRET ?= app-dotenv
DOTENV_MOUNT_PATH ?= /var/secrets/.env

.PHONY: run docker-create docker-build deploy gcp-enable-apis dotenv-secret-create dotenv-upload cloud-run-update-dotenv secret-grant-access

run:
	uvicorn application:app --host 0.0.0.0 --port 8000

docker-create:
	docker buildx create --use --name ar_builder || docker buildx use ar_builder

docker-build:
	docker buildx build --platform linux/amd64 -t $(IMAGE) --push .

deploy:
	gcloud run deploy nightout-backend --image $(IMAGE) --region us-east1 --allow-unauthenticated --quiet

# ---- Secret Manager .env workflow ----
gcp-enable-apis:
	@echo "Ensuring required GCP APIs are enabled (Secret Manager, Cloud Run, Artifact Registry)..."
	@gcloud services enable secretmanager.googleapis.com run.googleapis.com artifactregistry.googleapis.com --project $(PROJECT) --quiet

dotenv-secret-create: gcp-enable-apis
	@echo "Ensuring Secret Manager secret $(DOTENV_SECRET) exists in project $(PROJECT)..."
	@gcloud secrets describe $(DOTENV_SECRET) --project $(PROJECT) --format="value(name)" >/dev/null 2>&1 || \
		gcloud secrets create $(DOTENV_SECRET) --replication-policy=automatic --project $(PROJECT) --quiet

dotenv-upload: dotenv-secret-create
	@test -f .env || (echo "Error: .env file not found in NightOutBackend root" && exit 1)
	@echo "Uploading .env to Secret Manager (secret=$(DOTENV_SECRET))..."
	@gcloud secrets versions add $(DOTENV_SECRET) --data-file=.env --project $(PROJECT) --quiet
	@echo "Done. New secret version added."

cloud-run-update-dotenv:
	@echo "Deploying $(SERVICE) with mounted .env secret and DOTENV_PATH..."
	gcloud run deploy $(SERVICE) \
		--project $(PROJECT) \
		--region $(REGION) \
		--image $(IMAGE) \
		--allow-unauthenticated \
		--set-secrets=$(DOTENV_MOUNT_PATH)=$(DOTENV_SECRET):latest \
		--update-env-vars DOTENV_PATH=$(DOTENV_MOUNT_PATH) \
		--cpu 8 \
		--memory 32Gi \
		--max-instances 5 \
		--quiet
	@echo "Deployment complete."

# Grant Secret Manager Secret Accessor to the Cloud Run service account
secret-grant-access:
	@echo "Granting Secret Manager access for secret $(DOTENV_SECRET) to Cloud Run service account..."
	@SA_EMAIL=$$(gcloud run services describe $(SERVICE) --project $(PROJECT) --region $(REGION) --format='value(spec.template.spec.serviceAccountName)'); \
	if [ -z "$$SA_EMAIL" ]; then \
	  PN=$$(gcloud projects describe $(PROJECT) --format='value(projectNumber)'); \
	  SA_EMAIL="$$PN-compute@developer.gserviceaccount.com"; \
	fi; \
	echo "Using service account: $$SA_EMAIL"; \
	gcloud secrets add-iam-policy-binding $(DOTENV_SECRET) \
	  --project $(PROJECT) \
	  --member="serviceAccount:$$SA_EMAIL" \
	  --role=roles/secretmanager.secretAccessor \
	  --quiet
	@echo "Access granted. If deploy still fails, verify the service uses this SA."

ngrok:
	ngrok http http://localhost:8000