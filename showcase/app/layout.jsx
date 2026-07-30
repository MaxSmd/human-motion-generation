import "./globals.css";
import Nav from "@/components/Nav";

export const metadata = {
  title: "RMG · Riemannian motion generation",
  description:
    "Text-conditioned human motion generated on the product manifold ℝ³ × (S³)²², with hard joint constraints applied at sampling time. Training runs, evaluation, results and an interactive studio.",
};

// Explicit viewport so the mobile browser chrome matches the dark canvas. Next
// already emits width=device-width, initial-scale=1; user zoom is left enabled.
export const viewport = {
  themeColor: "#070a11",
  colorScheme: "dark",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        {/* Google Fonts via <link>; CSS vars fall back to system fonts if offline. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Nav />
        {children}
      </body>
    </html>
  );
}
