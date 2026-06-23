/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#141922",
        muted: "#647084",
        bias: "#0f766e",
        fact: "#ea7a25",
        line: "#d7dde6",
        "line-strong": "#aeb8c7",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 18px 42px rgba(20, 25, 34, 0.10)",
      },
    },
  },
  plugins: [],
};
