import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const datasetUrl = new URL(
  "../data/kursk_apartments_cian_snapshot.json",
  import.meta.url,
);
const estimatorPageUrl = new URL("../app/page.tsx", import.meta.url);

async function loadDataset() {
  return JSON.parse(await readFile(datasetUrl, "utf8"));
}

test("bundled dataset has the expected structure and size", async () => {
  const rows = await loadDataset();

  assert.equal(rows.length, 250);
  assert.ok(rows.every((row) => Array.isArray(row) && row.length === 9));
  assert.equal(new Set(rows.map((row) => row[0])).size, rows.length);
  assert.equal(new Set(rows.map((row) => row[1])).size, 133);
});

test("dataset values pass basic domain checks", async () => {
  const rows = await loadDataset();
  const districts = new Set([
    "Центральный",
    "Сеймский",
    "Железнодорожный",
  ]);
  const listingKinds = new Set(["Вторичка", "Новостройка"]);

  for (const row of rows) {
    const [
      id,
      address,
      district,
      listingKind,
      rooms,
      area,
      floor,
      totalFloors,
      price,
    ] = row;

    assert.ok(Number.isInteger(id) && id > 0);
    assert.ok(typeof address === "string" && address.trim().length > 3);
    assert.ok(districts.has(district));
    assert.ok(listingKinds.has(listingKind));
    assert.ok(Number.isInteger(rooms) && rooms >= 1 && rooms <= 5);
    assert.ok(typeof area === "number" && area >= 15 && area <= 300);
    assert.ok(Number.isInteger(floor) && floor >= 1);
    assert.ok(Number.isInteger(totalFloors) && totalFloors >= floor);
    assert.ok(Number.isInteger(price) && price >= 500_000);
  }
});

test("estimator exposes the additional property characteristics", async () => {
  const page = await readFile(estimatorPageUrl, "utf8");
  const expectedControls = [
    "Год постройки",
    "Тип постройки",
    "Индивидуальное отопление",
    "Чистовая отделка",
    "White box",
    "С ремонтом",
  ];

  for (const label of expectedControls) {
    assert.ok(page.includes(label), `Missing estimator control: ${label}`);
  }

  assert.ok(page.includes("propertyAdjustments"));
  assert.ok(page.includes("Корректировки характеристик"));
});
