document.addEventListener("DOMContentLoaded", function () {
  // Ambil data kegagalan dari elemen HTML
  const kegagalanRate = document.getElementById("failure-rate");
  const yaFailures = parseInt(kegagalanRate.getAttribute("data-ya"));
  const tidakFailures = parseInt(kegagalanRate.getAttribute("data-tidak"));

  const failureRateData = {
    labels: ["YA", "TIDAK"],
    data: [yaFailures, tidakFailures],
  };

  const pieChart = document.querySelector(".chart-pie");

  // Buat chart pie dengan data kegagalan
  new Chart(pieChart, {
    type: "doughnut", // atau bisa gunakan "pie" untuk tampilan pie chart
    data: {
      labels: failureRateData.labels,
      datasets: [
        {
          label: "Label Kegagalan",
          data: failureRateData.data,
          backgroundColor: ["#FF6384", "#36A2EB"], // Warna untuk setiap bagian
          hoverBackgroundColor: ["#FF6384", "#36A2EB"],
        },
      ],
    },
  });
});
