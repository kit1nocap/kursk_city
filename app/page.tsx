"use client";

import { useMemo, useState } from "react";
import {
  DATASET_META,
  KURSK_ADDRESSES,
  KURSK_LISTINGS,
  type KurskListing,
} from "./kurskListings";

type District = KurskListing["district"];
type ListingKind = KurskListing["listingKind"];
type BuildingType = "Кирпичный" | "Панельный" | "Монолитный" | "Блочный";
type FinishType = "Чистовая отделка" | "White box" | "С ремонтом";

type EstimateInput = {
  address: string;
  district: District;
  listingKind: ListingKind;
  rooms: number;
  area: number;
  floor: number;
  totalFloors: number;
  buildingYear: number;
  buildingType: BuildingType;
  individualHeating: boolean;
  finishType: FinishType;
};

type ComparableInfo = {
  rate: number;
  count: number;
  level: "дом" | "район и тип рынка" | "район" | "весь датасет";
};

type PriceAdjustment = {
  label: string;
  rate: number;
};

type ModelMetrics = {
  r2: number;
  mae: number;
  rmse: number;
};

type TrainedModel = {
  featureMeans: number[];
  featureScales: number[];
  weights: number[];
  targetMean: number;
  residualStd: number;
  metrics: ModelMetrics;
  predict: (input: EstimateInput) => number;
};

const DISTRICTS: District[] = [
  "Центральный",
  "Сеймский",
  "Железнодорожный",
];

const LISTING_KINDS: ListingKind[] = ["Вторичка", "Новостройка"];
const BUILDING_TYPES: BuildingType[] = [
  "Кирпичный",
  "Панельный",
  "Монолитный",
  "Блочный",
];
const FINISH_TYPES: FinishType[] = [
  "Чистовая отделка",
  "White box",
  "С ремонтом",
];

const INITIAL_INPUT: EstimateInput = {
  address: "проспект Анатолия Дериглазова, 91",
  district: "Центральный",
  listingKind: "Вторичка",
  rooms: 1,
  area: 37.5,
  floor: 13,
  totalFloors: 17,
  buildingYear: 2018,
  buildingType: "Монолитный",
  individualHeating: true,
  finishType: "С ремонтом",
};

const FEATURE_LABELS = [
  "Площадь",
  "Количество комнат",
  "Этаж",
  "Этажность дома",
  "Ориентир по адресу",
  "Центральный район",
  "Сеймский район",
  "Железнодорожный район",
  "Новостройка",
];

function normalizeAddress(value: string) {
  return value
    .toLocaleLowerCase("ru-RU")
    .replace(/[.,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatPrice(value: number) {
  return `${Math.round(value).toLocaleString("ru-RU")} ₽`;
}

function formatCompactPrice(value: number) {
  return `${(value / 1_000_000).toLocaleString("ru-RU", {
    maximumFractionDigits: 2,
  })} млн ₽`;
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function propertyAdjustments(input: EstimateInput): PriceAdjustment[] {
  const yearRate = Math.min(
    0.06,
    Math.max(-0.06, (input.buildingYear - 2000) * 0.0015),
  );
  const buildingTypeRates: Record<BuildingType, number> = {
    Кирпичный: 0.02,
    Панельный: -0.015,
    Монолитный: 0.04,
    Блочный: -0.01,
  };
  const finishRates: Record<FinishType, number> = {
    "Чистовая отделка": 0.01,
    "White box": -0.06,
    "С ремонтом": 0.08,
  };

  return [
    { label: `Год постройки: ${input.buildingYear}`, rate: yearRate },
    {
      label: `Тип дома: ${input.buildingType.toLocaleLowerCase("ru-RU")}`,
      rate: buildingTypeRates[input.buildingType],
    },
    {
      label: "Индивидуальное отопление",
      rate: input.individualHeating ? 0.04 : 0,
    },
    {
      label: `Отделка: ${input.finishType.toLocaleLowerCase("ru-RU")}`,
      rate: finishRates[input.finishType],
    },
  ];
}

function comparableInfo(
  input: EstimateInput | KurskListing,
  pool: KurskListing[],
  excludeId?: number,
): ComparableInfo {
  const available = pool.filter((row) => row.id !== excludeId);
  const normalizedAddress = normalizeAddress(input.address);
  const exact = available.filter(
    (row) => normalizeAddress(row.address) === normalizedAddress,
  );

  let selected = exact;
  let level: ComparableInfo["level"] = "дом";

  if (selected.length < 2) {
    selected = available.filter(
      (row) =>
        row.district === input.district &&
        row.listingKind === input.listingKind,
    );
    level = "район и тип рынка";
  }

  if (selected.length < 3) {
    selected = available.filter((row) => row.district === input.district);
    level = "район";
  }

  if (!selected.length) {
    selected = available;
    level = "весь датасет";
  }

  return {
    rate: mean(selected.map((row) => row.price / row.area)),
    count: selected.length,
    level,
  };
}

function featureVector(
  input: EstimateInput | KurskListing,
  pool: KurskListing[],
  excludeId?: number,
) {
  const benchmark = comparableInfo(input, pool, excludeId);
  return [
    input.area,
    input.rooms,
    input.floor / Math.max(input.totalFloors, 1),
    input.totalFloors,
    benchmark.rate / 100_000,
    input.district === "Центральный" ? 1 : 0,
    input.district === "Сеймский" ? 1 : 0,
    input.district === "Железнодорожный" ? 1 : 0,
    input.listingKind === "Новостройка" ? 1 : 0,
  ];
}

function solveLinearSystem(matrix: number[][], vector: number[]) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);

  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (
        Math.abs(augmented[row][column]) >
        Math.abs(augmented[pivot][column])
      ) {
        pivot = row;
      }
    }

    [augmented[column], augmented[pivot]] = [
      augmented[pivot],
      augmented[column],
    ];

    const divisor = augmented[column][column] || 1e-12;
    for (let cell = column; cell <= size; cell += 1) {
      augmented[column][cell] /= divisor;
    }

    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let cell = column; cell <= size; cell += 1) {
        augmented[row][cell] -= factor * augmented[column][cell];
      }
    }
  }

  return augmented.map((row) => row[size]);
}

function trainModel(rows: KurskListing[]): TrainedModel {
  const trainRows = rows.filter((_, index) => index % 5 !== 0);
  const testRows = rows.filter((_, index) => index % 5 === 0);
  const trainFeatures = trainRows.map((row) =>
    featureVector(row, trainRows, row.id),
  );
  const targets = trainRows.map((row) => row.price);
  const featureCount = trainFeatures[0].length;

  const featureMeans = Array.from({ length: featureCount }, (_, index) =>
    mean(trainFeatures.map((row) => row[index])),
  );
  const featureScales = Array.from({ length: featureCount }, (_, index) => {
    const variance = mean(
      trainFeatures.map((row) => (row[index] - featureMeans[index]) ** 2),
    );
    return Math.sqrt(variance) || 1;
  });
  const standardized = trainFeatures.map((row) =>
    row.map(
      (value, index) => (value - featureMeans[index]) / featureScales[index],
    ),
  );
  const targetMean = mean(targets);
  const centeredTargets = targets.map((value) => value - targetMean);
  const ridgeLambda = 2.4;

  const xtx = Array.from({ length: featureCount }, (_, rowIndex) =>
    Array.from({ length: featureCount }, (_, columnIndex) => {
      const product = standardized.reduce(
        (sum, row) => sum + row[rowIndex] * row[columnIndex],
        0,
      );
      return product + (rowIndex === columnIndex ? ridgeLambda : 0);
    }),
  );
  const xty = Array.from({ length: featureCount }, (_, featureIndex) =>
    standardized.reduce(
      (sum, row, rowIndex) =>
        sum + row[featureIndex] * centeredTargets[rowIndex],
      0,
    ),
  );
  const weights = solveLinearSystem(xtx, xty);

  const predictFromVector = (raw: number[]) =>
    Math.max(
      650_000,
      targetMean +
        raw.reduce(
          (sum, value, index) =>
            sum +
            ((value - featureMeans[index]) / featureScales[index]) *
              weights[index],
          0,
        ),
    );

  const testPredictions = testRows.map((row) =>
    predictFromVector(featureVector(row, trainRows)),
  );
  const testTargets = testRows.map((row) => row.price);
  const errors = testTargets.map(
    (target, index) => target - testPredictions[index],
  );
  const mae = mean(errors.map((error) => Math.abs(error)));
  const rmse = Math.sqrt(mean(errors.map((error) => error ** 2)));
  const testMean = mean(testTargets);
  const totalSquares = testTargets.reduce(
    (sum, value) => sum + (value - testMean) ** 2,
    0,
  );
  const residualSquares = errors.reduce(
    (sum, error) => sum + error ** 2,
    0,
  );
  const r2 = totalSquares ? 1 - residualSquares / totalSquares : 0;

  return {
    featureMeans,
    featureScales,
    weights,
    targetMean,
    residualStd: rmse,
    metrics: { r2, mae, rmse },
    predict: (input) => predictFromVector(featureVector(input, trainRows)),
  };
}

function numericInput(
  value: string,
  fallback: number,
  min: number,
  max: number,
) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

export default function HomePage() {
  const model = useMemo(() => trainModel(KURSK_LISTINGS), []);
  const [form, setForm] = useState<EstimateInput>(INITIAL_INPUT);
  const [estimateInput, setEstimateInput] =
    useState<EstimateInput>(INITIAL_INPUT);
  const [addressTouched, setAddressTouched] = useState(false);

  const matchingListings = useMemo(() => {
    const normalized = normalizeAddress(form.address);
    if (!normalized) return [];
    return KURSK_LISTINGS.filter(
      (row) => normalizeAddress(row.address) === normalized,
    );
  }, [form.address]);

  const currentBenchmark = comparableInfo(estimateInput, KURSK_LISTINGS);
  const baseEstimatedPrice = model.predict(estimateInput);
  const priceAdjustments = propertyAdjustments(estimateInput);
  const adjustmentRate = priceAdjustments.reduce(
    (sum, adjustment) => sum + adjustment.rate,
    0,
  );
  const estimatedPrice = Math.max(
    650_000,
    baseEstimatedPrice * (1 + adjustmentRate),
  );
  const intervalSize = Math.max(model.residualStd, estimatedPrice * 0.1);
  const lowerEstimate = Math.max(500_000, estimatedPrice - intervalSize);
  const upperEstimate = estimatedPrice + intervalSize;
  const selectedVector = featureVector(estimateInput, KURSK_LISTINGS);
  const influences = FEATURE_LABELS.map((label, index) => ({
    label,
    value:
      ((selectedVector[index] - model.featureMeans[index]) /
        model.featureScales[index]) *
      model.weights[index],
  }))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 4);

  const uniqueAddressCount = KURSK_ADDRESSES.length;

  const updateAddress = (address: string) => {
    const exact = KURSK_LISTINGS.find(
      (row) => normalizeAddress(row.address) === normalizeAddress(address),
    );

    setForm((current) => ({
      ...current,
      address,
      ...(exact
        ? {
            district: exact.district,
            listingKind: exact.listingKind,
            rooms: exact.rooms,
            area: exact.area,
            floor: exact.floor,
            totalFloors: exact.totalFloors,
          }
        : {}),
    }));
  };

  const submitEstimate = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAddressTouched(true);
    if (!form.address.trim()) return;
    setEstimateInput({ ...form });
  };

  return (
    <main>
      <header className="hero">
        <nav className="nav shell" aria-label="Основная навигация">
          <a className="brand" href="#top" aria-label="КурсМетр — на главную">
            <span className="brand-mark">К</span>
            <span>
              КУРС<span>МЕТР</span>
            </span>
          </a>
          <div className="nav-links">
            <a href="#estimate">Оценка</a>
            <a href="#analytics">Аналитика</a>
            <a href="#method">Методика</a>
          </div>
          <a className="nav-pill" href="#estimate">
            Оценить квартиру
          </a>
        </nav>

        <div className="hero-grid shell" id="top">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="live-dot" />
              ML-анализ рынка Курска
            </div>
            <h1>
              Узнайте стоимость квартиры <span>по адресу</span>
            </h1>
            <p>
              Введите адрес дома и параметры квартиры. Модель сравнит объект с
              открытой выборкой объявлений по Курску и рассчитает
              ориентировочную цену.
            </p>
            <div className="hero-actions">
              <a className="button primary" href="#estimate">
                Рассчитать стоимость
              </a>
              <a className="text-link" href="#method">
                Как работает модель <span>↘</span>
              </a>
            </div>
            <div className="hero-proof">
              <div>
                <strong>{KURSK_LISTINGS.length}</strong>
                <span>объявление</span>
              </div>
              <div>
                <strong>{uniqueAddressCount}</strong>
                <span>адресов</span>
              </div>
              <div>
                <strong>{DATASET_META.collectedAt}</strong>
                <span>дата снимка</span>
              </div>
            </div>
          </div>

          <div className="hero-visual" aria-label="Пример результата оценки">
            <div className="map-grid" />
            <span className="visual-kicker">Пример ML-оценки</span>
            <div className="visual-price">{formatCompactPrice(estimatedPrice)}</div>
            <div className="visual-range">
              диапазон {formatCompactPrice(lowerEstimate)} —{" "}
              {formatCompactPrice(upperEstimate)}
            </div>
            <div className="visual-card">
              <div className="building-shape" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <div>
                <strong>{estimateInput.address}</strong>
                <span>
                  {estimateInput.rooms}-комн. · {estimateInput.area} м² ·{" "}
                  {estimateInput.floor}/{estimateInput.totalFloors} этаж ·{" "}
                  {estimateInput.buildingYear} г.
                </span>
              </div>
            </div>
            <div className="visual-footer">
              <div>
                <span>Сопоставление</span>
                <div className="confidence-track">
                  <span />
                </div>
              </div>
              <strong>{currentBenchmark.count} объектов</strong>
            </div>
          </div>
        </div>
      </header>

      <section className="estimator-section" id="estimate">
        <div className="shell">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <p className="section-kicker">Оценка квартиры</p>
              <h2>Введите адрес и параметры объекта</h2>
            </div>
            <p>
              При точном совпадении адреса поля подставятся автоматически.
              Если дома нет в выборке, оценка продолжит работать по району и
              типу рынка.
            </p>
          </div>

          <div className="estimator-grid">
            <form className="form-card" onSubmit={submitEstimate}>
              <div className="field field-wide">
                <label htmlFor="address">
                  Адрес дома <span>Курск</span>
                </label>
                <input
                  id="address"
                  list="kursk-addresses"
                  placeholder="Например: ул. Энгельса, 115"
                  value={form.address}
                  onChange={(event) => updateAddress(event.target.value)}
                  onBlur={() => setAddressTouched(true)}
                  aria-describedby="address-status"
                  required
                />
                <datalist id="kursk-addresses">
                  {KURSK_ADDRESSES.map((address) => (
                    <option value={address} key={address} />
                  ))}
                </datalist>
                <p
                  className={`address-match ${
                    addressTouched && !matchingListings.length ? "missing" : ""
                  }`}
                  id="address-status"
                >
                  {matchingListings.length
                    ? `✓ Найдено объявлений по этому дому: ${matchingListings.length}. Параметры подставлены из выборки.`
                    : "Можно выбрать адрес из подсказок или ввести другой адрес Курска — тогда модель использует аналоги по району."}
                </p>
              </div>

              <div className="field">
                <label htmlFor="district">Район</label>
                <select
                  id="district"
                  value={form.district}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      district: event.target.value as District,
                    }))
                  }
                >
                  {DISTRICTS.map((district) => (
                    <option key={district}>{district}</option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label htmlFor="listing-kind">Рынок</label>
                <select
                  id="listing-kind"
                  value={form.listingKind}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      listingKind: event.target.value as ListingKind,
                    }))
                  }
                >
                  {LISTING_KINDS.map((kind) => (
                    <option key={kind}>{kind}</option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label htmlFor="rooms">Комнат</label>
                <input
                  id="rooms"
                  type="number"
                  min="1"
                  max="5"
                  value={form.rooms}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      rooms: numericInput(event.target.value, 1, 1, 5),
                    }))
                  }
                />
              </div>

              <div className="field">
                <label htmlFor="area">
                  Площадь, м² <span>{form.area}</span>
                </label>
                <input
                  id="area"
                  type="number"
                  min="15"
                  max="250"
                  step="0.1"
                  value={form.area}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      area: numericInput(event.target.value, 15, 15, 250),
                    }))
                  }
                />
              </div>

              <div className="field">
                <label htmlFor="floor">Этаж</label>
                <input
                  id="floor"
                  type="number"
                  min="1"
                  max={form.totalFloors}
                  value={form.floor}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      floor: numericInput(
                        event.target.value,
                        1,
                        1,
                        current.totalFloors,
                      ),
                    }))
                  }
                />
              </div>

              <div className="field">
                <label htmlFor="total-floors">Этажей в доме</label>
                <input
                  id="total-floors"
                  type="number"
                  min="1"
                  max="40"
                  value={form.totalFloors}
                  onChange={(event) =>
                    setForm((current) => {
                      const totalFloors = numericInput(
                        event.target.value,
                        1,
                        1,
                        40,
                      );
                      return {
                        ...current,
                        totalFloors,
                        floor: Math.min(current.floor, totalFloors),
                      };
                    })
                  }
                />
              </div>

              <div className="field">
                <label htmlFor="building-year">Год постройки</label>
                <input
                  id="building-year"
                  type="number"
                  min="1900"
                  max="2026"
                  value={form.buildingYear}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      buildingYear: numericInput(
                        event.target.value,
                        2000,
                        1900,
                        2026,
                      ),
                    }))
                  }
                />
              </div>

              <div className="field">
                <label htmlFor="building-type">Тип постройки</label>
                <select
                  id="building-type"
                  value={form.buildingType}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      buildingType: event.target.value as BuildingType,
                    }))
                  }
                >
                  {BUILDING_TYPES.map((buildingType) => (
                    <option key={buildingType}>{buildingType}</option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label htmlFor="individual-heating">
                  Индивидуальное отопление
                </label>
                <select
                  id="individual-heating"
                  value={form.individualHeating ? "Да" : "Нет"}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      individualHeating: event.target.value === "Да",
                    }))
                  }
                >
                  <option>Да</option>
                  <option>Нет</option>
                </select>
              </div>

              <div className="field">
                <label htmlFor="finish-type">Отделка</label>
                <select
                  id="finish-type"
                  value={form.finishType}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      finishType: event.target.value as FinishType,
                    }))
                  }
                >
                  {FINISH_TYPES.map((finishType) => (
                    <option key={finishType}>{finishType}</option>
                  ))}
                </select>
              </div>

              <button className="button primary calculate" type="submit">
                Рассчитать стоимость
              </button>
            </form>

            <aside className="result-card" key={estimatedPrice}>
              <div className="result-topline">
                <span>Оценка модели</span>
                <span className="result-badge">Ridge + корректировки</span>
              </div>
              <p className="result-label">Ориентировочная стоимость</p>
              <div className="result-price">{formatPrice(estimatedPrice)}</div>
              <div className="result-square">
                {formatPrice(estimatedPrice / estimateInput.area)} за м²
              </div>

              <div className="interval">
                <div className="interval-labels">
                  <span>{formatCompactPrice(lowerEstimate)}</span>
                  <strong>Расчётный диапазон</strong>
                  <span>{formatCompactPrice(upperEstimate)}</span>
                </div>
                <div className="interval-track">
                  <span />
                </div>
                <p>
                  Ориентир: {currentBenchmark.level},{" "}
                  {currentBenchmark.count} сопоставимых объявлений.
                </p>
              </div>

              <div className="influence">
                <p>Главные факторы расчёта</p>
                {influences.map((item) => (
                  <div className="influence-row" key={item.label}>
                    <span>{item.label}</span>
                    <strong className={item.value >= 0 ? "positive" : "negative"}>
                      {item.value >= 0 ? "+" : "−"}
                      {formatCompactPrice(Math.abs(item.value))}
                    </strong>
                  </div>
                ))}
              </div>
              <div className="influence property-adjustments">
                <p>Корректировки характеристик</p>
                {priceAdjustments.map((adjustment) => (
                  <div className="influence-row" key={adjustment.label}>
                    <span>{adjustment.label}</span>
                    <strong
                      className={adjustment.rate >= 0 ? "positive" : "negative"}
                    >
                      {adjustment.rate >= 0 ? "+" : "−"}
                      {Math.abs(adjustment.rate * 100).toLocaleString("ru-RU", {
                        maximumFractionDigits: 1,
                      })}
                      % ·{" "}
                      {formatCompactPrice(
                        Math.abs(baseEstimatedPrice * adjustment.rate),
                      )}
                    </strong>
                  </div>
                ))}
              </div>
              <div className="result-note">
                <span>i</span>
                <p>
                  Базовую цену рассчитывает Ridge-модель по объявлениям.
                  Характеристики дома и отделки применяются как прозрачные
                  сценарные корректировки. Результат не является заключением
                  оценщика или гарантированной ценой сделки.
                </p>
              </div>
            </aside>
          </div>
        </div>
      </section>

      <section className="analytics-section" id="analytics">
        <div className="shell">
          <div className="section-heading">
            <div>
              <span className="section-number light">02</span>
              <p className="section-kicker light">Проверка модели</p>
              <h2>Метрики на отложенной выборке</h2>
            </div>
            <p>
              Каждая пятая строка отложена для теста. Все показатели
              пересчитываются в браузере из встроенного датасета.
            </p>
          </div>

          <div className="metrics-grid">
            <article className="metric-card accent">
              <span>R²</span>
              <strong>{model.metrics.r2.toFixed(2)}</strong>
              <p>Доля объяснённой вариативности цены на тестовой части.</p>
            </article>
            <article className="metric-card">
              <span>MAE</span>
              <strong>{formatCompactPrice(model.metrics.mae)}</strong>
              <p>Средняя абсолютная ошибка прогноза.</p>
            </article>
            <article className="metric-card">
              <span>RMSE</span>
              <strong>{formatCompactPrice(model.metrics.rmse)}</strong>
              <p>Ошибка с повышенным весом крупных отклонений.</p>
            </article>
            <article className="metric-card">
              <span>Выборка</span>
              <strong>{KURSK_LISTINGS.length}</strong>
              <p>Открытых объявлений в текущем снимке.</p>
            </article>
          </div>

        </div>
      </section>

      <section className="method-section" id="method">
        <div className="shell">
          <div className="section-heading">
            <div>
              <span className="section-number">03</span>
              <p className="section-kicker">Как работает оценка</p>
              <h2>От открытых объявлений до прогноза</h2>
            </div>
            <p>
              Данные загружаются внутри проекта автоматически, а расчёт можно
              проверить как в браузере, так и Python-скриптом.
            </p>
          </div>
          <div className="steps">
            <article className="step">
              <span>1</span>
              <div className="step-icon">CSV</div>
              <h3>Снимок рынка</h3>
              <p>
                Публичные карточки объявлений Курска сведены в единый CSV без
                персональных данных продавцов.
              </p>
            </article>
            <article className="step">
              <span>2</span>
              <div className="step-icon">QA</div>
              <h3>Проверка данных</h3>
              <p>
                Проверяются цена, площадь, адрес, этажность, число комнат,
                округ и тип рынка.
              </p>
            </article>
            <article className="step">
              <span>3</span>
              <div className="step-icon">ML</div>
              <h3>Ridge-регрессия</h3>
              <p>
                Числовые признаки нормализуются, категории кодируются, а
                регуляризация стабилизирует модель на небольшой выборке.
              </p>
            </article>
            <article className="step">
              <span>4</span>
              <div className="step-icon">₽</div>
              <h3>Адресные аналоги</h3>
              <p>
                Сначала берутся объявления того же дома, затем — того же
                округа и типа рынка. Результат показан диапазоном.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="cta-section">
        <div className="shell">
          <div className="cta">
            <div>
              <span>Попробуйте на своём адресе</span>
              <h2>Получите ориентир стоимости квартиры за минуту</h2>
            </div>
            <a className="button light-button" href="#estimate">
              Ввести адрес
            </a>
          </div>
        </div>
      </section>

      <footer>
        <div className="footer-grid shell">
          <a className="brand" href="#top">
            <span className="brand-mark">К</span>
            <span>
              КУРС<span>МЕТР</span>
            </span>
          </a>
          <p>
            Учебный исследовательский проект. Используются публичные цены
            предложений на дату снимка; результат не заменяет профессиональную
            оценку недвижимости.
          </p>
          <a href="#top">Наверх ↑</a>
        </div>
      </footer>
    </main>
  );
}
