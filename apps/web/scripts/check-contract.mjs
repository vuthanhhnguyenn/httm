import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const contractPath = resolve(here, "../../../specs/001-smart-exam-proctoring/contracts/openapi.yaml");
const generatedPath = resolve(here, "../src/types/api.generated.ts");
const [contract, generated] = await Promise.all([readFile(contractPath, "utf8"), readFile(generatedPath, "utf8")]);
const version = contract.match(/^  version:\s*([^\s]+)\s*$/m)?.[1];
const generatedVersion = generated.match(/API_CONTRACT_VERSION\s*=\s*["']([^"']+)["']/)?.[1];
if (!version || version !== generatedVersion) {
  console.error(`Contract drift: OpenAPI=${version ?? "missing"}, generated=${generatedVersion ?? "missing"}`);
  process.exit(1);
}
for (const path of ["/api/health:", "/api/sessions:", "/api/sessions/{session_id}/start:", "/api/sessions/{session_id}/stop:"]) {
  if (!contract.includes(`  ${path}`)) {
    console.error(`Contract drift: missing ${path}`);
    process.exit(1);
  }
}
console.log(`Contract check passed for OpenAPI ${version}`);

