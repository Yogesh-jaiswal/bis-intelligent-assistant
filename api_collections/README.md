# BIS Intelligent Assistant — API Collections & Testing Guide

This directory contains pre-configured API request collections for testing and validating the BIS Intelligent Assistant endpoints (`/v1/health` and `/v1/query`).

---

## 1. Available Collections

| Collection File | Format | Primary Target Client | Description |
| :--- | :--- | :--- | :--- |
| [`./Insomnia API Collections`](./Insomnia%20API%20Collections) | Insomnia v5 Schema (YAML) | **Insomnia REST Client** | Native Insomnia collection with pre-configured requests, JSON headers, and sample payloads. |
| [`./Generic API Collections`](./Generic%20API%20Collections) | HTTP Archive 1.2 (HAR JSON) | **Postman, Hoppscotch, Thunder Client, Bruno** | Universal HAR archive containing recorded requests, headers, query bodies, and response schemas. |

---

## 2. How to Import & Use

### Option A: Using Insomnia REST Client (Recommended)
1. Open **Insomnia**.
2. Click the **Dashboard / Workspaces** dropdown menu at the top left.
3. Click **Import / Export** $\rightarrow$ **Import Data** $\rightarrow$ **From File**.
4. Select [`./Insomnia API Collections`](./Insomnia%20API%20Collections).
5. Insomnia will import the **BIS intelligent assistant** workspace containing:
   * `Health route`: `GET http://127.0.0.1:5000/v1/health`
   * `Query route`: `POST http://127.0.0.1:5000/v1/query`
6. Click **Send** on either request to verify connectivity.

---

### Option B: Using Postman
Postman natively supports importing HTTP Archive (HAR) files:
1. Open **Postman**.
2. Click the **Import** button in the top-left navigation bar.
3. Drag and drop or browse to select [`./Generic API Collections`](./Generic%20API%20Collections).
4. Postman will automatically parse the HAR entries and convert them into a Postman Collection.
5. In Postman, inspect the imported requests:
   * **Health Route**: `GET http://127.0.0.1:5000/v1/health`
   * **Query Route**: `POST http://127.0.0.1:5000/v1/query`
6. Click **Send**.

---

### Option C: Using Hoppscotch, Thunder Client (VS Code), or Bruno
1. **Thunder Client**:
   * Open VS Code $\rightarrow$ Thunder Client tab.
   * Click **Collections** $\rightarrow$ Click the menu (`...`) $\rightarrow$ **Import**.
   * Select [`./Generic API Collections`](./Generic%20API%20Collections) or choose "HAR".
2. **Hoppscotch**:
   * Visit `https://hoppscotch.io`.
   * Click **Collections** $\rightarrow$ **Import Collections** $\rightarrow$ Choose **HAR**.
   * Upload [`./Generic API Collections`](./Generic%20API%20Collections).
3. **Bruno**:
   * Select **Import Collection** $\rightarrow$ Choose **Postman / HAR** $\rightarrow$ Select [`./Generic API Collections`](./Generic%20API%20Collections).

---

## 3. Request Specifications & Sample Payloads

### 1. Health Check
* **Method**: `GET`
* **URL**: `http://127.0.0.1:5000/v1/health`
* **Response**:
  ```json
  {
    "success": true,
    "data": "BIS Intelligent agent is healthy",
    "error": null
  }
  ```

---

### 2. Conversational Query (`POST /v1/query`)
* **Method**: `POST`
* **URL**: `http://127.0.0.1:5000/v1/query`
* **Headers**: `Content-Type: application/json`

#### Example 1: Cable Standard / Lab Finder Query (Multilingual / Hindi)
```json
{
  "message": {
    "content": "What laboratories can help me get the BIS certificate for my cables business?",
    "language": "hi"
  }
}
```

#### Example 2: Packaged Water Standard Query (English)
```json
{
  "message": {
    "content": "What is the Indian Standard for packaged drinking water?",
    "language": "en"
  }
}
```

#### Example 3: Multi-turn Follow-up Query
```json
{
  "conversation_id": "conv_01K8V6X8Y9Z...",
  "message": {
    "content": "What are the mandatory testing requirements under this standard?",
    "language": "en"
  }
}
```

---

## 4. Troubleshooting

* **Connection Refused (`ECONNREFUSED`)**: Ensure the backend application is running either via Docker (`docker compose -f docker-compose.yml -f docker-compose.app.yml up -d`) or locally (`python run.py`) on port `5000`.
* **504 Gateway Timeout / Slow Response**: If using a Cloudflare tunnel to Ollama, verify that the notebook runtime is active and reachable. Local inference latency depends heavily on your GPU availability.
