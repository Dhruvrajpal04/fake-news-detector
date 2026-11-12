/**
 * FakeNewsLens - Client-side Controller
 * Handles interactive text input, AJAX prediction requests,
 * dynamic metric rendering, and sample data injection.
 */

document.addEventListener("DOMContentLoaded", function () {
  // Navigation Menu Toggle (Mobile)
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("show");
    });
  }

  // Detector Form Elements
  const detectForm = document.getElementById("detectForm");
  const newsTextArea = document.getElementById("newsText");
  const sourceUrlInput = document.getElementById("sourceUrl");
  const charCounter = document.getElementById("charCounter");
  const errorAlert = document.getElementById("errorAlert");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnIcon = document.getElementById("btnIcon");
  const btnText = document.getElementById("btnText");
  const clearBtn = document.getElementById("clearBtn");

  // Sample Buttons
  const sampleRealBtn = document.getElementById("sampleRealBtn");
  const sampleFakeBtn = document.getElementById("sampleFakeBtn");
  const sampleClickbaitBtn = document.getElementById("sampleClickbaitBtn");

  // Results Panel Elements
  const resultPanel = document.getElementById("resultPanel");
  const resultBadge = document.getElementById("resultBadge");
  const badgeIcon = document.getElementById("badgeIcon");
  const badgeText = document.getElementById("badgeText");
  const confidenceValue = document.getElementById("confidenceValue");
  const meterReal = document.getElementById("meterReal");
  const meterFake = document.getElementById("meterFake");
  const realProbText = document.getElementById("realProbText");
  const fakeProbText = document.getElementById("fakeProbText");
  const explanationText = document.getElementById("explanationText");
  const snippetText = document.getElementById("snippetText");

  // Live Verification Elements
  const googleApiKeyInput = document.getElementById("googleApiKey");
  const googleFactCheckContainer = document.getElementById("googleFactCheckContainer");
  const googleFactCheckList = document.getElementById("googleFactCheckList");
  const liveNewsWireContainer = document.getElementById("liveNewsWireContainer");
  const liveNewsWireList = document.getElementById("liveNewsWireList");
  const noLiveNewsNotice = document.getElementById("noLiveNewsNotice");

  // Curated Sample Texts for Testing
  const SAMPLES = {
    real: {
      url: "https://www.reuters.com/business/finance",
      text: "WASHINGTON (Reuters) - Federal Reserve policymakers signaled Wednesday that the central bank intends to keep interest rates steady as inflation indicators move closer to the target benchmark. In an official briefing released after the monetary committee meeting, officials stated that labor markets remained resilient with steady wage growth across manufacturing and services sectors. Financial analysts noted that the announcement aligns with market expectations.",
    },
    fake: {
      url: "https://conspiracy-wire-unfiltered.example.com",
      text: "BOMBSHELL EXCLUSIVE: Secret underground military tribunal issues emergency warrants for all mainstream media executives tonight! Whistleblowers expose hidden plan to poison city water supplies with mind-control chemicals starting Friday at midnight. The corrupt establishment is panicking and trying to ban this story from the internet before citizens wake up! SHARE THIS VIRAL TRUTH IMMEDIATELY BEFORE THEY DELETE OUR ENTIRE WEBSITE!",
    },
    clickbait: {
      url: "https://daily-miracle-health-secrets.example.net",
      text: "DOCTORS ARE TERRIFIED: Renowned 85-year-old grandmother accidentally cures all incurable chronic ailments using this 1 simple kitchen spice found in your cabinet! The pharmaceutical monopoly tried to have her arrested to protect their trillions in drug profits. Click here to discover the bizarre 30-second bedtime trick that Big Pharma does NOT want you to see before it gets censored forever!",
    },
  };

  // Live Character and Word Counter
  function updateCounters() {
    if (!newsTextArea || !charCounter) return;
    const text = newsTextArea.value.trim();
    const chars = text.length;
    const words = text ? text.split(/\s+/).length : 0;
    charCounter.textContent = `${chars.toLocaleString()} characters | ${words.toLocaleString()} words`;
  }

  if (newsTextArea) {
    newsTextArea.addEventListener("input", updateCounters);
  }

  // Load Samples
  if (sampleRealBtn) {
    sampleRealBtn.addEventListener("click", () => {
      newsTextArea.value = SAMPLES.real.text;
      if (sourceUrlInput) sourceUrlInput.value = SAMPLES.real.url;
      updateCounters();
      if (resultPanel) resultPanel.style.display = "none";
      if (errorAlert) errorAlert.style.display = "none";
    });
  }

  if (sampleFakeBtn) {
    sampleFakeBtn.addEventListener("click", () => {
      newsTextArea.value = SAMPLES.fake.text;
      if (sourceUrlInput) sourceUrlInput.value = SAMPLES.fake.url;
      updateCounters();
      if (resultPanel) resultPanel.style.display = "none";
      if (errorAlert) errorAlert.style.display = "none";
    });
  }

  if (sampleClickbaitBtn) {
    sampleClickbaitBtn.addEventListener("click", () => {
      newsTextArea.value = SAMPLES.clickbait.text;
      if (sourceUrlInput) sourceUrlInput.value = SAMPLES.clickbait.url;
      updateCounters();
      if (resultPanel) resultPanel.style.display = "none";
      if (errorAlert) errorAlert.style.display = "none";
    });
  }

  // Clear Form
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (newsTextArea) newsTextArea.value = "";
      if (sourceUrlInput) sourceUrlInput.value = "";
      updateCounters();
      if (resultPanel) resultPanel.style.display = "none";
      if (errorAlert) errorAlert.style.display = "none";
      if (newsTextArea) newsTextArea.focus();
    });
  }

  // Submit Handler
  if (detectForm) {
    detectForm.addEventListener("submit", async function (e) {
      e.preventDefault();

      if (errorAlert) errorAlert.style.display = "none";
      const text = newsTextArea.value.trim();
      const sourceUrl = sourceUrlInput ? sourceUrlInput.value.trim() : "";

      // Client-side Validation
      if (!text) {
        showError("Please enter or paste an article text or headline.");
        return;
      }

      if (text.length < 15) {
        showError("Input is too short (minimum 15 characters needed for stylistic analysis).");
        return;
      }

      if (text.length > 50000) {
        showError("Input is too long (maximum 50,000 characters allowed).");
        return;
      }

      // Enter Loading State
      setLoading(true);

      try {
        const googleApiKey = googleApiKeyInput ? googleApiKeyInput.value.trim() : "";

        const response = await fetch("/api/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            text: text,
            source_url: sourceUrl,
            google_api_key: googleApiKey,
          }),
        });

        const data = await response.json();

        if (!response.ok || data.status !== "success") {
          throw new Error(data.message || "Failed to process prediction request.");
        }

        renderPrediction(data);
      } catch (err) {
        showError(err.message || "An unexpected error occurred while communicating with the server.");
      } finally {
        setLoading(false);
      }
    });
  }

  // Display Error Message
  function showError(msg) {
    if (!errorAlert) return;
    errorAlert.textContent = msg;
    errorAlert.style.display = "block";
    errorAlert.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // Toggle Loading Spinner and Button State
  function setLoading(isLoading) {
    if (!analyzeBtn) return;
    analyzeBtn.disabled = isLoading;
    if (btnSpinner) btnSpinner.style.display = isLoading ? "inline-block" : "none";
    if (btnIcon) btnIcon.style.display = isLoading ? "none" : "inline-block";
    if (btnText) btnText.textContent = isLoading ? "Analyzing Text..." : "Analyze News";
  }

  // Render Result into DOM
  function renderPrediction(data) {
    if (!resultPanel) return;

    const isReal = data.label === "Likely Real";

    // Set panel styling
    resultPanel.className = `result-card ${isReal ? "real" : "fake"}`;
    resultBadge.className = `badge-label ${isReal ? "real" : "fake"}`;

    badgeIcon.innerHTML = isReal ? "&#9989;" : "&#10060;";
    badgeText.textContent = data.label;

    confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;

    const realProb = data.probabilities.real;
    const fakeProb = data.probabilities.fake;

    meterReal.style.width = `${realProb}%`;
    meterFake.style.width = `${fakeProb}%`;
    realProbText.textContent = `${realProb.toFixed(1)}%`;
    fakeProbText.textContent = `${fakeProb.toFixed(1)}%`;

    explanationText.textContent = data.explanation;
    snippetText.textContent = data.submitted_snippet;

    // Render Live Google Fact Check API results
    if (googleFactCheckContainer && googleFactCheckList) {
      if (data.has_fact_checks && data.live_fact_checks.length > 0) {
        googleFactCheckList.innerHTML = data.live_fact_checks
          .map(
            (fc) => `
          <div style="background: #ffffff; border-radius: 4px; padding: 0.65rem 0.85rem; margin-top: 0.4rem; border: 1px solid #bfdbfe;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
              <span style="font-weight: 700; font-size: 0.82rem; color: #1e40af;">${fc.publisher}</span>
              <span style="font-size: 0.75rem; font-weight: 800; padding: 0.15rem 0.5rem; border-radius: 999px; background: #fee2e2; color: #b91c1c;">Rating: ${fc.rating}</span>
            </div>
            <div style="font-size: 0.85rem; color: #1e293b; font-weight: 500;">"${fc.claim}"</div>
            ${fc.url ? `<a href="${fc.url}" target="_blank" rel="noopener noreferrer" style="font-size: 0.78rem; color: #2563eb; text-decoration: none; display: inline-block; margin-top: 0.25rem;">Read Fact-Check Investigation &nearr;</a>` : ""}
          </div>
        `
          )
          .join("");
        googleFactCheckContainer.style.display = "block";
      } else {
        googleFactCheckContainer.style.display = "none";
      }
    }

    // Render Live Google News Wire Search results
    if (liveNewsWireContainer && liveNewsWireList && noLiveNewsNotice) {
      if (data.has_live_news && data.live_news_coverage.length > 0) {
        liveNewsWireList.innerHTML = data.live_news_coverage
          .map(
            (nw) => `
          <div style="background: #ffffff; border-radius: 4px; padding: 0.6rem 0.8rem; border: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;">
            <div>
              <div style="font-size: 0.85rem; font-weight: 600; color: var(--navy-primary);">${nw.title}</div>
              <div style="font-size: 0.75rem; color: var(--text-muted);">${nw.source} &bull; ${nw.pub_date || "Recent"}</div>
            </div>
            ${nw.link ? `<a href="${nw.link}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary btn-sm" style="font-size: 0.75rem; padding: 0.2rem 0.5rem; white-space: nowrap;">Read Article &nearr;</a>` : ""}
          </div>
        `
          )
          .join("");
        liveNewsWireContainer.style.display = "block";
        noLiveNewsNotice.style.display = "none";
      } else {
        liveNewsWireContainer.style.display = "none";
        noLiveNewsNotice.style.display = "block";
      }
    }

    // Display Panel and Scroll Smoothly
    resultPanel.style.display = "block";
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
