import type { Metadata } from "next";
import "./globals.css";

const title = "КурсМетр — ML-оценка квартир в Курске";
const description =
  "Оценка стоимости квартиры в Курске по адресу и параметрам: встроенный датасет публичных объявлений, ML-модель, метрики и открытые источники.";

export const metadata: Metadata = {
  metadataBase: new URL("https://kursk-city.vercel.app"),
  title,
  description,
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  openGraph: {
    title,
    description,
    type: "website",
    locale: "ru_RU",
    images: [{ url: "/og.png", width: 1732, height: 909 }],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
