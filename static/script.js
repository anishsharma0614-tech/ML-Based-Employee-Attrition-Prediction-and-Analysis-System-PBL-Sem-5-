const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const errorBox = document.getElementById("formError");
const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const statusDot = document.getElementById("statusDot");
const apiStatus = document.getElementById("apiStatus");

async function checkApi() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.status === "ok") {
      statusDot.classList.add("online");
      apiStatus.textContent = "Ready for predictions";
    } else {
      apiStatus.textContent = "Model unavailable";
    }
  } catch (_) {
    apiStatus.textContent = "API not connected";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";
  button.disabled = true;
  button.querySelector("span").textContent = "Analysing profile…";

  const payload = {};
  for (const [key, value] of new FormData(form).entries()) {
    const input = form.elements.namedItem(key);
    payload[key] = input instanceof HTMLInputElement ? Number(value) : value;
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail || "Prediction could not be completed.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const left = String(data.prediction).toLowerCase() === "left";
    document.getElementById("predictionTitle").textContent =
      left ? "Attrition signal detected" : "Employee likely to stay";
    document.getElementById("predictionDescription").textContent =
      left
        ? "The model classifies this profile as more likely to leave."
        : "The model classifies this profile as more likely to stay.";
    const probability = data.attrition_probability;
    document.getElementById("probabilityValue").textContent =
      probability === null || probability === undefined ? "Not available" : `${probability.toFixed(1)}%`;
    document.getElementById("probabilityBar").style.width =
      probability === null || probability === undefined ? "0%" : `${Math.max(0, Math.min(100, probability))}%`;
    document.getElementById("resultNote").textContent = data.note;
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");
    document.getElementById("resultPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorBox.textContent = error.message || "Could not connect to the prediction API. Check that main.py is running.";
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Run prediction";
  }
});

document.getElementById("resetButton").addEventListener("click", () => {
  form.reset();
  resultContent.classList.add("hidden");
  emptyState.classList.remove("hidden");
  errorBox.textContent = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});

checkApi();
