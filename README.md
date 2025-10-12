### This is the backend for Night Out Client

It is deployed to GCP.

Build the docker image using make docker-build
The make any changes to .env file locally
Then run make dotenv-upload
Then upload your Docker Image using the make cloud-run-update-dotenv command