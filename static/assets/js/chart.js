const chartData = {
  labels: ["YA", "TIDAK"],
  data: [45, 55],
};

const pieChart = document.querySelector(".chart-pie");

new Chart(pieChart, {
  type: "doughnut",
  data: {
    labels: chartData.labels,
    datasets: [
      {
        label: "Kegagalan",
        data: chartData.data,
      },
    ],
  },
});
