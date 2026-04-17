IMAGE := us-east1-docker.pkg.dev/optimal-aegis-470701-j8/containers/nightout-backend:latest
PROJECT ?= optimal-aegis-470701-j8
REGION ?= us-east1
SERVICE ?= nightout-backend
DOTENV_SECRET ?= app-dotenv
DOTENV_MOUNT_PATH ?= /var/secrets/.env

.PHONY: run docker-create docker-build deploy gcp-enable-apis dotenv-secret-create dotenv-upload cloud-run-update-dotenv secret-grant-access

run:
	uvicorn application:app --host 0.0.0.0 --port 8000 --reload

docker-create:
	docker buildx create --use --name ar_builder || docker buildx use ar_builder

docker-build:
	docker buildx build --platform linux/amd64 -t $(IMAGE) --push .

deploy:
	gcloud run deploy nightout-backend --image $(IMAGE) --project $(PROJECT) --region $(REGION) --allow-unauthenticated --quiet

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

# ---- EventsParser Cron Job ----
EVENTS_PARSER_IMAGE := us-east1-docker.pkg.dev/$(PROJECT)/containers/events-parser:latest
EVENTS_PARSER_SERVICE := events-parser

# Clear user_curated_events in Firebase (removes wrong food/events), then optionally rerun pipeline.
# Use after fixing pipeline bugs so the app shows correct data.
events-parser-clear-curated:
	python scripts/clear_curated_events_and_rerun.py

# Clear curated events and run the pipeline locally (50 mi radius, 14 days).
events-parser-clear-and-rerun:
	python scripts/clear_curated_events_and_rerun.py --rerun --radius 50 --days 14

events-parser-run-local:
	python -m service.events_parser.cron_app

events-parser-docker-build:
	docker buildx build --platform linux/amd64 \
		-f Dockerfile.events_parser \
		-t $(EVENTS_PARSER_IMAGE) --push .

events-parser-deploy:
	gcloud run deploy $(EVENTS_PARSER_SERVICE) \
		--project $(PROJECT) \
		--region $(REGION) \
		--image $(EVENTS_PARSER_IMAGE) \
		--no-allow-unauthenticated \
		--set-secrets=$(DOTENV_MOUNT_PATH)=$(DOTENV_SECRET):latest \
		--update-env-vars DOTENV_PATH=$(DOTENV_MOUNT_PATH) \
		--cpu 1 \
		--memory 512Mi \
		--max-instances 1 \
		--timeout 300 \
		--quiet
	@echo "Events parser deployed."

events-parser-create-scheduler:
	@echo "Creating Cloud Scheduler job to trigger events parser every 6 hours..."
	@SERVICE_URL=$$(gcloud run services describe $(EVENTS_PARSER_SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	SA_EMAIL=$$(gcloud projects describe $(PROJECT) --format='value(projectNumber)'); \
	SA_EMAIL="$$SA_EMAIL-compute@developer.gserviceaccount.com"; \
	gcloud scheduler jobs create http events-parser-cron \
		--project $(PROJECT) \
		--location $(REGION) \
		--schedule "0 */6 * * *" \
		--uri "$$SERVICE_URL/run_events_parser" \
		--http-method POST \
		--oidc-service-account-email "$$SA_EMAIL" \
		--oidc-token-audience "$$SERVICE_URL" \
		--time-zone "America/New_York" \
		--attempt-deadline 300s \
		--quiet
	@echo "Scheduler created: runs every 6 hours."

events-parser-trigger:
	@SERVICE_URL=$$(gcloud run services describe $(EVENTS_PARSER_SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	curl -X POST "$$SERVICE_URL/run_events_parser" \
		-H "Authorization: Bearer $$(gcloud auth print-identity-token)" \
		-H "Content-Type: application/json"

# ---- Daily Picks Push Notifications ----
DAILY_PICKS_TOPIC := daily-picks

daily-picks-create-topic:
	@echo "Creating Pub/Sub topic $(DAILY_PICKS_TOPIC)..."
	@gcloud pubsub topics create $(DAILY_PICKS_TOPIC) --project $(PROJECT) --quiet 2>/dev/null || \
		echo "Topic $(DAILY_PICKS_TOPIC) already exists."

daily-picks-create-scheduler:
	@echo "Creating Cloud Scheduler job to send daily picks at 6 PM ET..."
	@SERVICE_URL=$$(gcloud run services describe $(EVENTS_PARSER_SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	SA_EMAIL=$$(gcloud projects describe $(PROJECT) --format='value(projectNumber)'); \
	SA_EMAIL="$$SA_EMAIL-compute@developer.gserviceaccount.com"; \
	gcloud scheduler jobs create http daily-picks-cron \
		--project $(PROJECT) \
		--location $(REGION) \
		--schedule "0 18 * * *" \
		--uri "$$SERVICE_URL/send_daily_picks" \
		--http-method POST \
		--oidc-service-account-email "$$SA_EMAIL" \
		--oidc-token-audience "$$SERVICE_URL" \
		--time-zone "America/New_York" \
		--attempt-deadline 300s \
		--quiet
	@echo "Scheduler created: runs daily at 6 PM ET."

daily-picks-trigger:
	@SERVICE_URL=$$(gcloud run services describe $(EVENTS_PARSER_SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	curl -X POST "$$SERVICE_URL/send_daily_picks" \
		-H "Authorization: Bearer $$(gcloud auth print-identity-token)" \
		-H "Content-Type: application/json"

daily-picks-deploy-function:
	@echo "Deploying daily_picks_notify Cloud Function..."
	gcloud functions deploy daily_picks_notify \
		--project $(PROJECT) \
		--region $(REGION) \
		--runtime python312 \
		--trigger-topic $(DAILY_PICKS_TOPIC) \
		--entry-point daily_picks_notify \
		--source cloud_functions/daily_picks_notify \
		--memory 256MB \
		--timeout 60s \
		--quiet
	@echo "Cloud Function deployed."

daily-picks-setup: daily-picks-create-topic daily-picks-deploy-function daily-picks-create-scheduler
	@echo "Daily picks notification system fully set up."

# ---- Friend Activity Notifications ----
FRIEND_ACTIVITY_TOPIC := friend-activity

friend-activity-create-topic:
	@echo "Creating Pub/Sub topic $(FRIEND_ACTIVITY_TOPIC)..."
	@gcloud pubsub topics create $(FRIEND_ACTIVITY_TOPIC) --project $(PROJECT) --quiet 2>/dev/null || \
		echo "Topic $(FRIEND_ACTIVITY_TOPIC) already exists."

friend-activity-deploy-function:
	@echo "Deploying friend_activity_notify Cloud Function..."
	gcloud functions deploy friend_activity_notify \
		--project $(PROJECT) \
		--region $(REGION) \
		--runtime python312 \
		--trigger-topic $(FRIEND_ACTIVITY_TOPIC) \
		--entry-point friend_activity_notify \
		--source cloud_functions/friend_activity_notify \
		--memory 256MB \
		--timeout 60s \
		--quiet
	@echo "Cloud Function deployed."

friend-activity-setup: friend-activity-create-topic friend-activity-deploy-function
	@echo "Friend activity notification system fully set up."

# ---- Broadcast Notification ----
check-tokens:
	@echo "Checking registered FCM tokens..."
	@ADMIN_TOKEN=$$(grep ADMIN_RECS_TOKEN .env 2>/dev/null | cut -d= -f2); \
	SERVICE_URL=$$(gcloud run services describe $(SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	curl -s "$$SERVICE_URL/admin/fcm_token_count" \
		-H "X-Admin-Token: $$ADMIN_TOKEN" ; \
	echo ""

broadcast:
	@echo "Sending broadcast push notification to all registered users..."
	@ADMIN_TOKEN=$$(grep ADMIN_RECS_TOKEN .env 2>/dev/null | cut -d= -f2); \
	if [ -z "$$ADMIN_TOKEN" ]; then echo "ERROR: Set ADMIN_RECS_TOKEN in .env first"; exit 1; fi; \
	SERVICE_URL=$$(gcloud run services describe $(SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	curl -s -X POST "$$SERVICE_URL/admin/broadcast_notification" \
		-H "X-Admin-Token: $$ADMIN_TOKEN" \
		-H "Content-Type: application/json" \
		-d '{"title": "Welcome to Boredom Destroyer! 🎉", "body": "Your personalized picks are ready. Open the app to explore events, food, movies & more near you!"}' ; \
	echo ""

broadcast-custom:
	@if [ -z "$(TITLE)" ] || [ -z "$(BODY)" ]; then \
		echo "Usage: make broadcast-custom TITLE=\"Your title\" BODY=\"Your message\""; \
		exit 1; \
	fi
	@ADMIN_TOKEN=$$(grep ADMIN_RECS_TOKEN .env 2>/dev/null | cut -d= -f2); \
	if [ -z "$$ADMIN_TOKEN" ]; then echo "ERROR: Set ADMIN_RECS_TOKEN in .env first"; exit 1; fi; \
	SERVICE_URL=$$(gcloud run services describe $(SERVICE) \
		--project $(PROJECT) --region $(REGION) --format='value(status.url)'); \
	curl -s -X POST "$$SERVICE_URL/admin/broadcast_notification" \
		-H "X-Admin-Token: $$ADMIN_TOKEN" \
		-H "Content-Type: application/json" \
		-d "{\"title\": \"$(TITLE)\", \"body\": \"$(BODY)\"}" ; \
	echo ""