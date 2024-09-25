/* Created by Tivotal */

let primaryColor = getComputedStyle(document.documentElement)
  .getPropertyValue("--color-primary")
  .trim();

let labelColor = getComputedStyle(document.documentElement)
  .getPropertyValue("--color-label")
  .trim();

let fontFamily = getComputedStyle(document.documentElement)
  .getPropertyValue("--font-family")
  .trim();

let defaultOptions = {
  chart: {
    tollbar: {
      show: false,
    },
    zoom: {
      enabled: false,
    },
    width: "100%",
    height: 300,
    offsetY: 18,
  },

  dataLabels: {
    enabled: false,
  },
};

let barOptions = {
  ...defaultOptions,
  chart: {
    ...defaultOptions.chart,
    type: "line", // Menggunakan line chart
    zoom: {
      enabled: false,
    },
    animations: {
      enabled: true,
      easing: 'easeinout',
      speed: 800, // Animasi halus saat data di-update
    },
    scrollable: true, // Membuat chart bisa di-scroll
  },
  tooltip: {
    enabled: true,
    style: {
      fontFamily: fontFamily,
    },
    y: {
      formatter: (value) => `${value.toFixed(2)}`, // Menampilkan angka dalam 2 desimal
    },
  },
  series: [
    {
      name: "Accuracy",
      data: accuracies, // Data accuracy yang diambil dari Flask
    },
  ],
  colors: [primaryColor],
  stroke: {
    colors: [primaryColor],
    width: 2, // Lebih tipis agar tampilan lebih halus
    curve: 'smooth', // Menggunakan garis smooth
  },
  grid: {
    borderColor: 'rgba(255, 255, 255, 0.1)', // Grid yang lebih minimalis
    padding: {
      top: 10,
      right: 10,
      bottom: 10,
      left: 10,
    },
  },
  markers: {
    size: 3, // Ukuran marker lebih kecil
    strokeColors: primaryColor,
    strokeWidth: 2,
    hover: {
      size: 5,
    },
  },
  yaxis: {
    title: {
      text: 'Accuracy',
      style: {
        color: labelColor,
        fontFamily: fontFamily,
      },
    },
    min: 0.4, // Mengatur minimum agar sumbu Y tidak terlalu melebar
    max: 1,   // Batas maksimum sumbu Y untuk accuracy
  },
  xaxis: {
    labels: {
      show: true,
      floating: true,
      style: {
        colors: labelColor,
        fontFamily: fontFamily,
      },
    },
    tickAmount: Math.floor(randomStates.length / 5), // Kelipatan puluhan
    categories: randomStates, // Data random states yang diambil dari Flask
    axisBorder: {
      show: false,
    },
    crosshairs: {
      show: false,
    },
    labels: {
      formatter: function(value, index) {
        return index % 10 === 0 ? value : ''; // Tampilkan label pada kelipatan 10
      },
    },
  },
};


let chart = new ApexCharts(document.querySelector(".chart-area"), barOptions);

chart.render();