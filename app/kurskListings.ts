import rawListings from "../data/kursk_apartments_enriched_all.json";

export type District = "Центральный" | "Сеймский" | "Железнодорожный";
export type ListingKind = "Вторичка" | "Новостройка";
export type BuildingType =
  | "Кирпичный"
  | "Панельный"
  | "Монолитный"
  | "Кирпично-монолитный"
  | "Блочный"
  | "Деревянный";
export type FinishType = "Без отделки" | "White box" | "С ремонтом";
export type RepairClass = "Бизнес" | "Комфорт" | "Эконом";

export type KurskListing = {
  id: number;
  address: string;
  district: District;
  listingKind: ListingKind;
  rooms: number;
  area: number;
  floor: number;
  totalFloors: number;
  buildingYear: number | null;
  buildingType: BuildingType | null;
  individualHeating: boolean | null;
  finishType: FinishType | null;
  repairClass: RepairClass | null;
  price: number;
  sourceUrl: string;
  provenance: {
    buildingYear: "listing_page" | "same_address" | null;
    buildingType: "listing_page" | "same_address" | null;
    individualHeating: "listing_page" | "same_address" | null;
    finishType: "listing_page" | null;
    repairClass: "listing_page" | null;
  };
};

export const DATASET_META = {
  city: "Курск",
  collectedAt: "30.07.2026",
  sourceName: "ЦИАН",
  note: "Публичные цены предложений, не цены завершённых сделок",
  fieldCoverage: {
    buildingYear: 246,
    buildingType: 216,
    individualHeating: 121,
    finishType: 230,
  },
};

export const KURSK_LISTINGS = rawListings as KurskListing[];

export const KURSK_ADDRESSES = Array.from(
  new Set(KURSK_LISTINGS.map((row) => row.address)),
).sort((left, right) => left.localeCompare(right, "ru"));
