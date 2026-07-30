import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(
  projectRoot,
  "data",
  "kursk_apartments_cian_snapshot.json",
);
const outputPath = path.join(
  projectRoot,
  "data",
  "kursk_apartments_enriched.json",
);
const reportPath = path.join(
  projectRoot,
  "data",
  "kursk_apartments_enrichment_report.json",
);
const cacheDirectory = path.join(projectRoot, ".cache", "cian-pages");

const SOURCE_FIELDS = [
  "id",
  "address",
  "district",
  "listingKind",
  "rooms",
  "area",
  "floor",
  "totalFloors",
  "price",
];

const BUILDING_TYPE_RULES = [
  [/монолитно.?кирпич|кирпично.?монолит/i, "Кирпично-монолитный"],
  [/панел/i, "Панельный"],
  [/монолит/i, "Монолитный"],
  [/кирпич/i, "Кирпичный"],
  [/блоч/i, "Блочный"],
  [/дерев/i, "Деревянный"],
];

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

function normalizeLine(value) {
  return value
    .replace(/[*_`#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function extractField(markdown, labels) {
  const labelSet = new Set(labels.map((label) => label.toLocaleLowerCase("ru-RU")));
  const lines = markdown.split(/\r?\n/).map(normalizeLine);

  for (let index = 0; index < lines.length; index += 1) {
    if (!labelSet.has(lines[index].toLocaleLowerCase("ru-RU"))) continue;

    for (
      let candidateIndex = index + 1;
      candidateIndex < Math.min(lines.length, index + 7);
      candidateIndex += 1
    ) {
      const candidate = lines[candidateIndex];
      if (
        candidate &&
        !candidate.startsWith("![") &&
        !candidate.startsWith("[Image") &&
        !labelSet.has(candidate.toLocaleLowerCase("ru-RU"))
      ) {
        return candidate;
      }
    }
  }

  return null;
}

function parseYear(markdown) {
  const exact = extractField(markdown, [
    "Год постройки",
    "Год сдачи",
    "Срок сдачи",
  ]);
  const exactYear = exact?.match(/\b(18\d{2}|19\d{2}|20[0-3]\d)\b/)?.[1];
  if (exactYear) return Number(exactYear);

  const contextual = markdown.match(
    /(?:год постройки|год сдачи|срок сдачи|построен[а-я]*)[^\d]{0,40}\b(18\d{2}|19\d{2}|20[0-3]\d)\b/i,
  )?.[1];
  return contextual ? Number(contextual) : null;
}

function parseBuildingType(markdown) {
  const value = extractField(markdown, ["Тип дома", "Материал дома"]);
  if (value) {
    const exact =
      BUILDING_TYPE_RULES.find(([pattern]) => pattern.test(value))?.[1] ?? null;
    if (exact) return exact;
  }

  const propertySection = markdown.slice(
    0,
    Math.max(
      markdown.indexOf("Похожие предложения"),
      markdown.indexOf("Другие предложения"),
      12_000,
    ),
  );
  const contextualRules = [
    [/монолитно.?кирпич|кирпично.?монолит/i, "Кирпично-монолитный"],
    [/монолитн[а-яё]*\s+каркас|дом\s+монолитн/i, "Монолитный"],
    [/дом\s+панельн|панельн[а-яё]*\s+дом/i, "Панельный"],
    [/дом\s+кирпичн|кирпичн[а-яё]*\s+дом/i, "Кирпичный"],
    [/дом\s+блочн|блочн[а-яё]*\s+дом/i, "Блочный"],
    [/дом\s+деревянн|деревянн[а-яё]*\s+дом/i, "Деревянный"],
  ];
  const candidates = contextualRules
    .map(([pattern, buildingType]) => ({
      buildingType,
      index: propertySection.search(pattern),
    }))
    .filter((candidate) => candidate.index >= 0)
    .sort((left, right) => left.index - right.index);

  return candidates[0]?.buildingType ?? null;
}

function parseHeating(markdown) {
  const value = extractField(markdown, ["Отопление"]);
  if (value) {
    if (
      /индивидуальн|поквартирн|квартирн[а-яё]*\s+отоплен/i.test(value)
    ) {
      return true;
    }
    if (/центральн|автономн[а-яё]*\s+котельн/i.test(value)) return false;
  }

  const propertySection = markdown.slice(
    0,
    Math.max(
      markdown.indexOf("Похожие предложения"),
      markdown.indexOf("Другие предложения"),
      12_000,
    ),
  );
  if (
    /поквартирн[а-яё]*\s+отоплен|индивидуальн[а-яё]*\s+отоплен|квартирн[а-яё]*\s+отоплен/i.test(
      propertySection,
    )
  ) {
    return true;
  }
  if (/центральн[а-яё]*\s+отоплен/i.test(propertySection)) return false;

  return null;
}

function classifyFinish(value) {
  if (!value) return null;
  if (/без отделк|без ремонт|чернов/i.test(value)) {
    return { finishType: "Без отделки", repairClass: null };
  }
  if (/white\s*box|вайт\s*бокс|предчистов/i.test(value)) {
    return { finishType: "White box", repairClass: null };
  }
  if (/дизайнер/i.test(value)) {
    return { finishType: "С ремонтом", repairClass: "Бизнес" };
  }
  if (/евроремонт|премиум|бизнес|качественн/i.test(value)) {
    return { finishType: "С ремонтом", repairClass: "Комфорт" };
  }
  if (/косметическ|чистов|эконом|с ремонт|ремонт/i.test(value)) {
    return { finishType: "С ремонтом", repairClass: "Эконом" };
  }
  return null;
}

function parseFinish(markdown) {
  const exact = extractField(markdown, ["Ремонт", "Отделка"]);
  const classifiedExact = classifyFinish(exact);
  if (classifiedExact) return classifiedExact;

  const propertySection = markdown.slice(
    0,
    Math.max(
      markdown.indexOf("Похожие предложения"),
      markdown.indexOf("Другие предложения"),
      12_000,
    ),
  );
  const patterns = [
    /(?:отделка|ремонт)[^\n.]{0,100}(?:без отделки|без ремонта|черновая|white\s*box|вайт\s*бокс|предчистовая|дизайнерский|евроремонт|косметический|чистовая)/i,
    /(?:без отделки|без ремонта|черновая|white\s*box|вайт\s*бокс|предчистовая|дизайнерский|евроремонт|косметический|чистовая)[^\n.]{0,100}(?:отделка|ремонт)/i,
  ];
  for (const pattern of patterns) {
    const match = propertySection.match(pattern)?.[0];
    const classified = classifyFinish(match);
    if (classified) return classified;
  }

  return null;
}

function parsePage(markdown) {
  const finish = parseFinish(markdown);
  return {
    buildingYear: parseYear(markdown),
    buildingType: parseBuildingType(markdown),
    individualHeating: parseHeating(markdown),
    finishType: finish?.finishType ?? null,
    repairClass: finish?.repairClass ?? null,
  };
}

function normalizeAddress(value) {
  return value
    .toLocaleLowerCase("ru-RU")
    .replace(/[«»"'.,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function fillBuildingFieldsFromSameAddress(rows) {
  const buildingFields = [
    "buildingYear",
    "buildingType",
    "individualHeating",
  ];
  const byAddress = new Map();

  for (const row of rows) {
    const key = normalizeAddress(row.address);
    const group = byAddress.get(key) ?? [];
    group.push(row);
    byAddress.set(key, group);
  }

  for (const group of byAddress.values()) {
    for (const field of buildingFields) {
      const known = new Set(
        group
          .map((row) => row[field])
          .filter((value) => value !== null),
      );
      if (known.size !== 1) continue;

      const [sharedValue] = known;
      for (const row of group) {
        if (row[field] === null) {
          row[field] = sharedValue;
          row.provenance[field] = "same_address";
        }
      }
    }
  }
}

async function fetchWithRetry(url, attempts = 5) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { Accept: "text/plain; charset=utf-8" },
        signal: AbortSignal.timeout(45_000),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const text = await response.text();
      if (text.length < 500 || /вы не робот|captcha/i.test(text)) {
        throw new Error("The returned page is empty or protected");
      }
      return text;
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await sleep(1_500 * attempt);
      }
    }
  }

  throw lastError;
}

async function loadPage(id) {
  const cachePath = path.join(cacheDirectory, `${id}.md`);
  if (existsSync(cachePath)) {
    return {
      markdown: await readFile(cachePath, "utf8"),
      fetched: false,
    };
  }

  const url = `https://r.jina.ai/https://kursk.cian.ru/sale/flat/${id}/`;
  const markdown = await fetchWithRetry(url);
  await writeFile(cachePath, markdown, "utf8");
  return { markdown, fetched: true };
}

async function main() {
  const sourceRows = JSON.parse(await readFile(sourcePath, "utf8"));
  await mkdir(cacheDirectory, { recursive: true });

  const queue = sourceRows.map((row) =>
    Object.fromEntries(SOURCE_FIELDS.map((field, index) => [field, row[index]])),
  );
  const enrichedRows = [];
  const failed = [];
  let completed = 0;

  async function worker() {
    while (queue.length) {
      const row = queue.shift();
      try {
        const { markdown, fetched } = await loadPage(row.id);
        const parsed = parsePage(markdown);
        enrichedRows.push({
          ...row,
          ...parsed,
          sourceUrl: `https://kursk.cian.ru/sale/flat/${row.id}/`,
          provenance: {
            buildingYear:
              parsed.buildingYear === null ? null : "listing_page",
            buildingType:
              parsed.buildingType === null ? null : "listing_page",
            individualHeating:
              parsed.individualHeating === null ? null : "listing_page",
            finishType: parsed.finishType === null ? null : "listing_page",
            repairClass:
              parsed.finishType !== "С ремонтом" ||
              parsed.repairClass !== null
                ? "listing_page"
                : null,
          },
        });
        if (fetched) await sleep(750);
      } catch (error) {
        failed.push({ id: row.id, error: String(error) });
      }

      completed += 1;
      if (completed % 10 === 0 || completed === sourceRows.length) {
        console.log(`Processed ${completed}/${sourceRows.length}`);
      }
    }
  }

  await Promise.all(Array.from({ length: 3 }, () => worker()));
  enrichedRows.sort((left, right) => left.id - right.id);
  fillBuildingFieldsFromSameAddress(enrichedRows);

  const requiredFields = [
    "buildingYear",
    "buildingType",
    "individualHeating",
    "finishType",
  ];
  const completeRows = enrichedRows.filter((row) =>
    requiredFields.every((field) => row[field] !== null),
  );
  const coverage = Object.fromEntries(
    requiredFields.map((field) => [
      field,
      enrichedRows.filter((row) => row[field] !== null).length,
    ]),
  );
  coverage.repairClass = enrichedRows.filter(
    (row) => row.finishType !== "С ремонтом" || row.repairClass !== null,
  ).length;

  const report = {
    generatedAt: new Date().toISOString(),
    sourceRows: sourceRows.length,
    fetchedRows: enrichedRows.length,
    completeRows: completeRows.length,
    coverage,
    failed,
    incomplete: enrichedRows
      .filter((row) => !completeRows.includes(row))
      .map((row) => ({
        id: row.id,
        missing: requiredFields.filter((field) => row[field] === null),
      })),
  };

  await writeFile(outputPath, `${JSON.stringify(completeRows, null, 2)}\n`, "utf8");
  await writeFile(
    path.join(projectRoot, "data", "kursk_apartments_enriched_all.json"),
    `${JSON.stringify(enrichedRows, null, 2)}\n`,
    "utf8",
  );
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  console.log(JSON.stringify(report, null, 2));
  console.log(`Saved ${completeRows.length} complete rows to ${outputPath}`);
}

await main();
