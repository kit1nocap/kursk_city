import rawListings from "../data/kursk_apartments_cian_snapshot.json";

export type KurskListing = {
  id: number;
  address: string;
  district: "Центральный" | "Сеймский" | "Железнодорожный";
  listingKind: "Вторичка" | "Новостройка";
  rooms: number;
  area: number;
  floor: number;
  totalFloors: number;
  price: number;
  sourceUrl: string;
};

type RawListing = [
  id: number,
  address: string,
  district: KurskListing["district"],
  listingKind: KurskListing["listingKind"],
  rooms: number,
  area: number,
  floor: number,
  totalFloors: number,
  price: number,
];

export const DATASET_META = {
  city: "Курск",
  collectedAt: "30.07.2026",
  sourceName: "ЦИАН",
  note: "Публичные цены предложений, не цены завершённых сделок",
};

export const KURSK_LISTINGS: KurskListing[] = (
  rawListings as unknown as RawListing[]
).map(
  ([
    id,
    address,
    district,
    listingKind,
    rooms,
    area,
    floor,
    totalFloors,
    price,
  ]) => ({
    id,
    address,
    district,
    listingKind,
    rooms,
    area,
    floor,
    totalFloors,
    price,
    sourceUrl: `https://kursk.cian.ru/sale/flat/${id}/`,
  }),
);

export const KURSK_ADDRESSES = Array.from(
  new Set(KURSK_LISTINGS.map((row) => row.address)),
).sort((left, right) => left.localeCompare(right, "ru"));
