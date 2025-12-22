document
  .getElementById("traffic-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = {
      junction: formData.get("junction"),
      datetime: formData.get("datetime"),
    };

    const response = await fetch("/predict_traffic", {
      method: "POST",
      body: new URLSearchParams(data),
    });

    const result = await response.json();

    if (result.success) {
      // Show the result card FIRST
      const resultCard = document.getElementById("result-card");
      resultCard.classList.remove("hidden");

      // Update traffic level text
      const trafficLevel = document.getElementById("traffic-level");
      trafficLevel.textContent = result.label;

      // Show advice
      const adviceEl = document.getElementById("advice");
      adviceEl.textContent = `Advice: ${result.advice}`;

      // Update Chart
      const ctx = document.getElementById("trafficChart").getContext("2d");
      if (window.trafficChart) window.trafficChart.destroy(); // remove previous chart
      window.trafficChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: ["6AM", "9AM", "12PM", "3PM", "6PM", "9PM"],
          datasets: [
            {
              label: "Predicted Congestion (%)",
              data: [20, 40, result.prediction, 50, 30, 20], // example data
              borderColor: "#00ffff",
              backgroundColor: "rgba(0,255,255,0.2)",
              fill: true,
            },
          ],
        },
        options: {
          scales: { y: { beginAtZero: true, max: 100 } },
          plugins: { legend: { display: true, labels: { color: "#00ffff" } } },
        },
      });
    } else {
      alert("Error: " + result.error);
    }
  });
