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
      boxShadow: {
        glow: "0 0 0 1px rgba(129,140,248,0.25), 0 8px 40px -12px rgba(129,140,248,0.45)",
        panel: "0 12px 40px -16px rgba(0,0,0,0.6)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: 0, transform: "translateY(8px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.35s ease both",
      },
    },
  },
  plugins: [],
};
