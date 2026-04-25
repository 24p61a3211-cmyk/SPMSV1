(function () {
    const payloadElement = document.getElementById("dashboard-data");
    if (!payloadElement || typeof Chart === "undefined") {
        return;
    }

    let payload = {};
    try {
        payload = JSON.parse(payloadElement.textContent || "{}");
    } catch (_error) {
        return;
    }

    const marks = payload.marks || { labels: [], values: [] };
    const attendance = payload.attendance || { labels: [], present: [], absent: [] };
    const risk = payload.risk || { score: 0, label: "Low Risk" };

    const subjectChartCanvas = document.getElementById("subjectPerformanceChart");
    if (subjectChartCanvas && marks.labels.length) {
        new Chart(subjectChartCanvas, {
            type: "bar",
            data: {
                labels: marks.labels,
                datasets: [
                    {
                        label: "Average %",
                        data: marks.values,
                        borderRadius: 10,
                        backgroundColor: [
                            "#264653",
                            "#2a9d8f",
                            "#e9c46a",
                            "#f4a261",
                            "#e76f51",
                            "#457b9d",
                        ],
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label(context) {
                                return "Average: " + context.formattedValue + "%";
                            },
                        },
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(82, 96, 109, 0.18)" },
                        ticks: { callback: (value) => value + "%" },
                    },
                    x: {
                        grid: { display: false },
                    },
                },
            },
        });
    }

    const attendanceCanvas = document.getElementById("attendanceClassicChart");
    if (attendanceCanvas && attendance.labels.length) {
        new Chart(attendanceCanvas, {
            type: "bar",
            data: {
                labels: attendance.labels,
                datasets: [
                    {
                        label: "Present",
                        data: attendance.present,
                        backgroundColor: "rgba(42, 157, 143, 0.82)",
                        borderRadius: 8,
                    },
                    {
                        label: "Absent",
                        data: attendance.absent,
                        backgroundColor: "rgba(231, 111, 81, 0.82)",
                        borderRadius: 8,
                    },
                ],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: "top" },
                },
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        grid: { color: "rgba(82, 96, 109, 0.18)" },
                    },
                },
            },
        });
    }

    const gaugeTextPlugin = {
        id: "gaugeTextPlugin",
        afterDraw(chart) {
            const value = Number(risk.score || 0).toFixed(1);
            const ctx = chart.ctx;
            const x = chart.getDatasetMeta(0).data[0].x;
            const y = chart.getDatasetMeta(0).data[0].y;

            ctx.save();
            ctx.textAlign = "center";
            ctx.fillStyle = "#102a43";
            ctx.font = "700 1.4rem Outfit";
            ctx.fillText(value + "/100", x, y + 4);
            ctx.font = "500 0.85rem Outfit";
            ctx.fillStyle = "#52606d";
            ctx.fillText(risk.label || "Risk", x, y + 26);
            ctx.restore();
        },
    };

    const riskCanvas = document.getElementById("riskGaugeChart");
    if (riskCanvas) {
        const score = Math.max(0, Math.min(100, Number(risk.score || 0)));
        new Chart(riskCanvas, {
            type: "doughnut",
            data: {
                labels: ["Risk Score", "Remaining"],
                datasets: [
                    {
                        data: [score, 100 - score],
                        backgroundColor: ["#e76f51", "#e5e7eb"],
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                cutout: "74%",
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false },
                },
            },
            plugins: [gaugeTextPlugin],
        });
    }
})();
