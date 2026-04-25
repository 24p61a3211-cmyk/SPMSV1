(function () {
    const payloadElement = document.getElementById("admin-data");
    if (!payloadElement || typeof Chart === "undefined") {
        return;
    }

    let payload = {};
    try {
        payload = JSON.parse(payloadElement.textContent || "{}");
    } catch (_error) {
        return;
    }

    const riskDistribution = payload.riskDistribution || { labels: [], values: [] };
    const riskCanvas = document.getElementById("cohortRiskChart");

    if (!riskCanvas || !riskDistribution.labels.length) {
        return;
    }

    new Chart(riskCanvas, {
        type: "doughnut",
        data: {
            labels: riskDistribution.labels,
            datasets: [
                {
                    data: riskDistribution.values,
                    backgroundColor: [
                        "rgba(42, 157, 143, 0.84)",
                        "rgba(233, 196, 106, 0.88)",
                        "rgba(231, 111, 81, 0.88)",
                    ],
                    borderColor: "#ffffff",
                    borderWidth: 2,
                },
            ],
        },
        options: {
            responsive: true,
            cutout: "62%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        boxWidth: 14,
                        padding: 16,
                    },
                },
            },
        },
    });
})();
