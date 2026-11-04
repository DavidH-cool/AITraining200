variable "project_id" {
  description = "Google Cloud Project ID where the Work-to-Blog ADK Agent is deployed."
  type        = string
}

variable "region" {
  description = "Google Cloud region for Cloud Run and Vertex AI Agent Engine resources."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Work-to-Blog ADK Agent service."
  type        = string
  default     = "work-to-blog-agent"
}

variable "container_image" {
  description = "Artifact Registry container image URI for the ADK agent service."
  type        = string
  default     = "us-central1-docker.pkg.dev/my-gcp-project/agents/work-to-blog-agent:latest"
}
