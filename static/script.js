
    async function predict() {
        const text = document.getElementById("statement").value.trim();
        const btn = document.getElementById("btn");
        const error = document.getElementById("error");
        const results = document.getElementById("results");
        const tagsDiv = document.getElementById("tags");

        error.textContent = "";
        results.style.display = "none";

        if (!text) {
            error.textContent = "Please paste a problem statement first.";
            return;
        }

        btn.disabled = true;
        btn.textContent = "Predicting...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text, top_k: 6, threshold: 0.15})
            });

            const data = await response.json();
            const predictions = data.predictions;

            if (!predictions.length) {
                error.textContent = "No tags predicted with sufficient confidence.";
                return;
            }

            tagsDiv.innerHTML = predictions.map(p => `
                <div class="tag">
                    <span class="tag-name">${p.tag}</span>
                    <div class="tag-bar-wrap">
                        <div class="tag-bar" style="width: ${p.confidence * 100}%"></div>
                    </div>
                    <span class="tag-pct">${Math.round(p.confidence * 100)}%</span>
                </div>
            `).join("");

            results.style.display = "block";

        } catch (e) {
            error.textContent = "Error connecting to server.";
        } finally {
            btn.disabled = false;
            btn.textContent = "Predict Tags";
        }
    }

    async function predictRating() {
        const contestId = document.getElementById("contestId").value.trim();
        const problemIndex = document.getElementById("problemIndex").value.trim();
        const btn = document.getElementById("ratingBtn");
        const error = document.getElementById("ratingError");
        const results = document.getElementById("ratingResults");

        error.textContent = "";
        results.style.display = "none";

        if (!contestId || !problemIndex) {
            error.textContent = "Please enter both a contest ID and problem index.";
            return;
        }

        btn.disabled = true;
        btn.textContent = "Predicting...";

        try {
            const response = await fetch("/predict-rating", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    contest_id: parseInt(contestId, 10),
                    problem_index: problemIndex.toUpperCase()
                })
            });

            const data = await response.json();

            if (!response.ok) {
                error.textContent = data.error || "Could not predict rating for this problem.";
                return;
            }

            document.getElementById("ratingValue").textContent = data.predicted_rating;
            document.getElementById("solvedCount").textContent = data.solved_count;
            document.getElementById("rawRating").textContent = data.predicted_rating_raw;

            results.style.display = "block";

        } catch (e) {
            error.textContent = "Error connecting to server.";
        } finally {
            btn.disabled = false;
            btn.textContent = "Predict Rating";
        }
    }