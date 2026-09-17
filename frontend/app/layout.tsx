import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CIFAKE Detector — Real vs AI Image Classifier",
  description:
    "Upload any image to instantly detect whether it's a real photograph or AI-generated using a custom-trained convolutional neural network.",
  keywords: ["AI detector", "image classifier", "CIFAKE", "real vs fake", "deep learning"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
