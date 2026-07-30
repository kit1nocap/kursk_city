import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const datasetUrl = new URL(
  "../data/kursk_apartments_enriched_all.json",
  import.meta.url,
);
const reportUrl = new URL(
  "../data/kursk_apartments_enrichment_report.json",
  import.meta.url,
);
const estimatorPageUrl = new URL("../app/page.tsx", import.meta.url);

async function loadJson(url) {
  return JSON.parse(await readFile(url, "utf8"));
}

test("expanded dataset contains 250 unique Kursk listings", async () => {
  const rows = await loadJson(datasetUrl);

  assert.equal(rows.length, 250);
  assert.equal(new Set(rows.map((row) => row.id)).size, rows.length);
  assert.equal(new Set(rows.map((row) => row.address)).size, 133);
  assert.ok(rows.every((row) => row.sourceUrl.endsWith(`/${row.id}/`)));
});

test("base listing values pass domain checks", async () => {
  const rows = await loadJson(datasetUrl);
  const districts = new Set([
    "Центральный",
    "Сеймский",
    "Железнодорожный",
  ]);
  const listingKinds = new Set(["Вторичка", "Новостройка"]);

  for (const row of rows) {
    assert.ok(Number.isInteger(row.id) && row.id > 0);
    assert.ok(typeof row.address === "string" && row.address.trim().length > 3);
    assert.ok(districts.has(row.district));
    assert.ok(listingKinds.has(row.listingKind));
    assert.ok(Number.isInteger(row.rooms) && row.rooms >= 1 && row.rooms <= 5);
    assert.ok(typeof row.area === "number" && row.area >= 15 && row.area <= 300);
    assert.ok(Number.isInteger(row.floor) && row.floor >= 1);
    assert.ok(Number.isInteger(row.totalFloors) && row.totalFloors >= row.floor);
    assert.ok(Number.isInteger(row.price) && row.price >= 500_000);
  }
});

test("expanded characteristics have real coverage and valid categories", async () => {
  const rows = await loadJson(datasetUrl);
  const report = await loadJson(reportUrl);
  const buildingTypes = new Set([
    "Кирпичный",
    "Панельный",
    "Монолитный",
    "Кирпично-монолитный",
    "Блочный",
    "Деревянный",
    null,
  ]);
  const finishTypes = new Set([
    "Без отделки",
    "White box",
    "С ремонтом",
    null,
  ]);
  const repairClasses = new Set(["Бизнес", "Комфорт", "Эконом", null]);

  assert.equal(report.fetchedRows, 250);
  assert.equal(report.failed.length, 0);
  assert.ok(report.coverage.buildingYear >= 240);
  assert.ok(report.coverage.buildingType >= 200);
  assert.ok(report.coverage.individualHeating >= 100);
  assert.ok(report.coverage.finishType >= 220);

  for (const row of rows) {
    assert.ok(
      row.buildingYear === null ||
        (Number.isInteger(row.buildingYear) &&
          row.buildingYear >= 1800 &&
          row.buildingYear <= 2035),
    );
    assert.ok(buildingTypes.has(row.buildingType));
    assert.ok(
      row.individualHeating === null ||
        typeof row.individualHeating === "boolean",
    );
    assert.ok(finishTypes.has(row.finishType));
    assert.ok(repairClasses.has(row.repairClass));
    if (row.finishType === "С ремонтом") {
      assert.ok(row.repairClass !== null);
    }
  }
});

test("new parameters are ML features without fixed price adjustments", async () => {
  const page = await readFile(estimatorPageUrl, "utf8");
  const expectedFeatures = [
    "Год постройки",
    "Кирпично-монолитный дом",
    "Деревянный дом",
    "Индивидуальное отопление",
    "Без отделки",
    "White box",
    "Ремонт бизнес-класса",
    "Ремонт комфорт-класса",
    "Ремонт эконом-класса",
  ];

  for (const feature of expectedFeatures) {
    assert.ok(page.includes(feature), `Missing ML feature: ${feature}`);
  }

  assert.ok(!page.includes("propertyAdjustments"));
  assert.ok(!page.includes("adjustmentRate"));
  assert.ok(page.includes("Ridge · 28 признаков"));
});
