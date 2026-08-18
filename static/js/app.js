/**
 * app.js — vanilla JS, no build step, no framework.
 *
 * Talks to the FastAPI backend via fetch(). Since index.html is served
 * by the same FastAPI app (see main.py), these are same-origin relative
 * URLs — no CORS configuration or base-URL env var needed on this side.
 *
 * Three responsibilities:
 * 1. Upload flow (click-to-browse + drag-and-drop)
 * 2. Document list (render, refresh, delete)
 * 3. Chat (ask a question, render answer + source citations)
 */

const API = {
  upload: "/documents/upload",
  documents: "/documents",
  query: "/query",
  evaluate: "/evaluate",
};

// ---------- DOM references ----------
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const documentList = document.getElementById("document-list");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

// ---------- Helpers ----------

/** Creates a DOM element with text content set safely (no innerHTML with
 * user-provided data anywhere in this file — avoids XSS from filenames,
 * questions, or model output that might contain HTML-like text). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ---------- Document list ----------

async function fetchDocuments() {
  try {
    const res = await fetch(API.documents);
    if (!res.ok) throw new Error(`Failed to load documents (${res.status})`);
    const data = await res.json();
    renderDocuments(data.documents);
  } catch (err) {
    console.error(err);
    documentList.innerHTML = "";
    documentList.appendChild(
      el("li", "document-item", "Couldn't load your documents. Try refreshing the page.")
    );
  }
}

function renderDocuments(docs) {
  documentList.innerHTML = "";

  if (docs.length === 0) {
    const empty = el("li", "document-item");
    empty.textContent = "No documents yet — upload one to get started.";
    documentList.appendChild(empty);
    return;
  }

  for (const doc of docs) {
    const item = el("li", "document-item");

    const name = el("span", "document-item__name", doc.filename);
    name.title = doc.filename;

    const status = el(
      "span",
      `document-item__status status--${doc.status}`,
      doc.status
    );

    const deleteBtn = el("button", "document-item__delete", "✕");
    deleteBtn.type = "button";
    deleteBtn.setAttribute("aria-label", `Delete ${doc.filename}`);
    deleteBtn.addEventListener("click", () => deleteDocument(doc.id, doc.filename));

    item.append(name, status, deleteBtn);
    documentList.appendChild(item);
  }
}

async function deleteDocument(id, filename) {
  const confirmed = confirm(`Delete "${filename}"? This can't be undone.`);
  if (!confirmed) return;

  try {
    const res = await fetch(`${API.documents}/${id}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) {
      throw new Error(`Failed to delete document (${res.status})`);
    }
    await fetchDocuments();
  } catch (err) {
    console.error(err);
    alert("Couldn't delete this document. Please try again.");
  }
}

// ---------- Upload ----------

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const statusText = document.getElementById("upload-status-text");
  statusText.textContent = `Uploading "${file.name}"…`;
  uploadZone.classList.add("upload-zone--uploading");

  try {
    const res = await fetch(API.upload, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Upload failed (${res.status})`);
    }

    if (data.status === "failed") {
      alert(`"${file.name}" couldn't be processed: ${data.error_message}`);
    }

    await fetchDocuments();
  } catch (err) {
    console.error(err);
    alert(err.message || "Upload failed. Please try again.");
  } finally {
    uploadZone.classList.remove("upload-zone--uploading");
    fileInput.value = ""; // allow re-selecting the same file later
  }
}

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
});

// Drag and drop
["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    uploadZone.classList.add("upload-zone--dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    uploadZone.classList.remove("upload-zone--dragover");
  });
});

uploadZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

// ---------- Chat ----------

function clearEmptyState() {
  const emptyState = chatMessages.querySelector(".chat-empty-state");
  if (emptyState) emptyState.remove();
}

function appendUserMessage(text) {
  clearEmptyState();
  const message = el("div", "chat-message chat-message--user", text);
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAssistantMessage(question, answer, sources) {
  clearEmptyState();
  const message = el("div", "chat-message chat-message--assistant");
  message.appendChild(el("p", null, answer));

  if (sources && sources.length > 0) {
    const sourcesWrap = el("div", "chat-message__sources");

    // Group by filename — multiple chunks from the same document
    // (e.g. one from page 5, one from page 12) should render as ONE
    // tag with both pages listed, not two near-identical-looking pills.
    const byFilename = new Map();
    for (const source of sources) {
      if (!byFilename.has(source.filename)) {
        byFilename.set(source.filename, new Set());
      }
      if (source.page_number) {
        byFilename.get(source.filename).add(source.page_number);
      }
    }

    for (const [filename, pages] of byFilename) {
      const sortedPages = [...pages].sort((a, b) => a - b);
      const label =
        sortedPages.length > 0
          ? `${filename} · p.${sortedPages.join(", ")}`
          : filename;
      sourcesWrap.appendChild(el("span", "source-tag", label));
    }

    message.appendChild(sourcesWrap);
  }

  // "Check this answer" — on-demand evaluation, not automatic, since
  // each click costs a couple of extra Groq (judge) calls. Only shown
  // when we actually have source contexts to evaluate against.
  if (sources && sources.length > 0) {
    const evalRow = el("div", "eval-row");
    const evalButton = el("button", "eval-button", "Check this answer");
    evalButton.type = "button";

    evalButton.addEventListener("click", async () => {
      evalButton.disabled = true;
      evalButton.textContent = "Checking…";

      try {
        const res = await fetch(API.evaluate, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            answer,
            contexts: sources.map((s) => s.chunk_text),
          }),
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || `Evaluation failed (${res.status})`);

        const faithfulnessPct = Math.round(data.faithfulness * 100);
        const relevancyPct = Math.round(data.answer_relevancy * 100);

        const badge = el(
          "span",
          "eval-badge",
          `Faithfulness ${faithfulnessPct}% · Relevancy ${relevancyPct}%`
        );
        evalRow.replaceChildren(badge);
      } catch (err) {
        console.error(err);
        evalButton.disabled = false;
        evalButton.textContent = "Check this answer";
        alert(err.message || "Couldn't evaluate this answer. Please try again.");
      }
    });

    evalRow.appendChild(evalButton);
    message.appendChild(evalRow);
  }

  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendLoadingMessage() {
  clearEmptyState();
  const message = el("div", "chat-message chat-message--assistant chat-message--loading", "Thinking…");
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return message;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  appendUserMessage(question);
  chatInput.value = "";
  chatInput.disabled = true;

  const loadingMessage = appendLoadingMessage();

  try {
    const res = await fetch(API.query, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    loadingMessage.remove();

    if (!res.ok) {
      throw new Error(data.detail || `Query failed (${res.status})`);
    }

    appendAssistantMessage(question, data.answer, data.sources);
  } catch (err) {
    console.error(err);
    loadingMessage.remove();
    appendAssistantMessage(
      question,
      "Something went wrong answering that question. Please try again.",
      []
    );
  } finally {
    chatInput.disabled = false;
    chatInput.focus();
  }
});

// ---------- Init ----------
fetchDocuments();