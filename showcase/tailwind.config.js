/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx}", "./components/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#080b13",
        panel: "#111726",
        edge: "#222c40",
        accent: "#818cf8",
        accent2: "#38bdf8",
      },
    },
  },
  plugins: [],
};
