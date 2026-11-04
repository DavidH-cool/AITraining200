terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required Google Cloud APIs for Vertex AI Agent Engine, Search, DLP, Secret Manager & Trace
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "discoveryengine.googleapis.com",
    "dlp.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "secretmanager.googleapis.com",
    "run.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# Least-privilege Service Account for the Work-to-Blog ADK Agent
resource "google_service_account" "blog_agent_runtime_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "Work-to-Blog ADK Agent Runtime Service Account"
  description  = "Least-privilege identity for ADK multi-agent orchestration, Vertex AI, DLP, and Secret Manager."
}

# IAM Bindings for Vertex AI, Discovery Engine (Vertex AI Search), DLP, Logging, Trace, and Secret Manager
resource "google_project_iam_member" "agent_runtime_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/dlp.user",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
    "roles/secretmanager.secretAccessor",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.blog_agent_runtime_sa.email}"
}

# Google Cloud Secret Manager Secret for CMS Publishing API Token (Zero hardcoded keys)
resource "google_secret_manager_secret" "cms_publishing_token" {
  secret_id = "cms-publishing-api-token"

  replication {
    auto {}
  }

  labels = {
    managed_by = "terraform"
    agent      = var.service_name
  }

  depends_on = [google_project_service.required_apis]
}

# Vertex AI Search Data Store for Persistent Technical Blog Knowledge & Reference Retrieval
resource "google_discovery_engine_data_store" "technical_blog_knowledge_store" {
  location                    = "global"
  data_store_id               = "${var.service_name}-knowledge-store"
  display_name                = "Engineering Work & Technical Blog Reference Knowledge Store"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false

  depends_on = [google_project_service.required_apis]
}

# Cloud Run v2 Service hosting the ADK Agent Web / API Server
resource "google_cloud_run_v2_service" "blog_writer_agent_service" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.blog_agent_runtime_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }

      env {
        name  = "VERTEX_SEARCH_DATA_STORE_ID"
        value = google_discovery_engine_data_store.technical_blog_knowledge_store.data_store_id
      }

      env {
        name  = "ENABLE_CLOUD_DLP_REDACTION"
        value = "true"
      }

      env {
        name  = "ENABLE_GCP_CLOUD_TRACE"
        value = "true"
      }

      env {
        name = "CMS_PUBLISHING_API_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.cms_publishing_token.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_project_iam_member.agent_runtime_roles,
  ]
}
