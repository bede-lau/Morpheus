#!/usr/bin/env node
/** Fetch publisher-provided test assets locally; see docs/ASSETS.md for rights. */
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, rename, rm, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { setTimeout as delay } from 'node:timers/promises';

const base = 'https://wlt-ai-cdn.art/example_exports/rustic_kitchen_with_natural_light';
export const assets = [
  {
    id: 'splat', name: 'rustic-kitchen-500k.spz',
    url: `${base}/rustic_kitchen_with_natural_light_500k.spz`,
    bytes: 7582907,
    sha256: '351f5f2ceb6ee9095a4b4100f864db9780a21cc6bdb8a70c6f88117cbb1d5fe0',
  },
  {
    id: 'mesh', name: 'rustic-kitchen-collider.glb',
    url: `${base}/rustic_kitchen_with_natural_light_collider.glb`,
    bytes: 2976256,
    sha256: '38b9f692edeb671b7106f3e36cfe7f4c55be20e98fe90c014a8714b09874f6fa',
  },
];

async function size(path) {
  try { return (await stat(path)).size; }
  catch (error) { if (error.code === 'ENOENT') return 0; throw error; }
}

export async function validFile(path, asset) {
  if (await size(path) !== asset.bytes) return false;
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex') === asset.sha256;
}

export async function fetchAsset(asset, directory, options = {}) {
  const {
    fetchImpl = fetch, maxAttempts = 6, requestTimeoutMs = 150000,
    wait = delay, log = console.log,
  } = options;
  await mkdir(directory, { recursive: true });
  const destination = join(directory, asset.name);
  const partial = `${destination}.part`;
  if (await validFile(destination, asset)) {
    log(`${asset.name}: verified existing file; skipped.`);
    return destination;
  }

  let lastError;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      let offset = await size(partial);
      if (offset >= asset.bytes) {
        if (await validFile(partial, asset)) {
          await rename(partial, destination);
          log(`${asset.name}: verified and completed existing partial file.`);
          return destination;
        }
        await rm(partial, { force: true });
        offset = 0;
      }

      log(`${asset.name}: attempt ${attempt}/${maxAttempts}, ${offset}/${asset.bytes} bytes.`);
      const headers = { 'Accept-Encoding': 'identity' };
      if (offset) headers.Range = `bytes=${offset}-`;
      const response = await fetchImpl(asset.url, {
        headers, redirect: 'follow', signal: AbortSignal.timeout(requestTimeoutMs),
      });
      if (![200, 206].includes(response.status) || !response.body) {
        await response.body?.cancel();
        throw new Error(`HTTP ${response.status} from the official asset host.`);
      }

      let append = false;
      if (response.status === 206) {
        const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('content-range') ?? '');
        if (!range || Number(range[1]) !== offset || Number(range[3]) !== asset.bytes ||
            Number(range[2]) < offset || Number(range[2]) >= asset.bytes) {
          await response.body.cancel();
          throw new Error('Server returned an incompatible Content-Range; partial file preserved.');
        }
        append = offset > 0;
      } else {
        // A host may ignore Range. Restart rather than append a second copy.
        offset = 0;
      }

      let received = offset;
      const bounded = new Transform({
        transform(chunk, encoding, callback) {
          received += chunk.length;
          callback(received > asset.bytes ? new Error('Response exceeds the expected asset size.') : null, chunk);
        },
      });
      await pipeline(
        Readable.fromWeb(response.body), bounded,
        createWriteStream(partial, { flags: append ? 'a' : 'w' }),
      );
      if (received !== asset.bytes) throw new Error(`Incomplete transfer: ${received}/${asset.bytes} bytes.`);
      if (!await validFile(partial, asset)) {
        await rm(partial, { force: true });
        throw new Error('SHA-256 mismatch. Discarded partial; original destination is unchanged.');
      }

      // Same-directory rename exposes only a fully verified final asset.
      await rename(partial, destination);
      log(`${asset.name}: verified ${asset.bytes} bytes and SHA-256; ready.`);
      return destination;
    } catch (error) {
      lastError = error;
      log(`${asset.name}: ${error.message}`);
      if (attempt < maxAttempts) await wait(Math.min(15000, 2000 * 2 ** (attempt - 1)));
    }
  }
  throw new Error(`Could not fetch ${asset.name} after ${maxAttempts} attempts. Run again to resume. ${lastError?.message}`);
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--help')) {
    console.log('Usage: node scripts/fetch-sample.mjs [--asset=splat|mesh|all]\nDownloads official World Labs sample files for local testing. See docs/ASSETS.md.');
    return;
  }
  if (args.length > 1 || (args.length && !/^--asset=(splat|mesh|all)$/.test(args[0]))) {
    throw new Error('Unsupported arguments. Use --help.');
  }
  const selected = args[0]?.split('=')[1] ?? 'all';
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  console.log('World Labs synthetic kitchen test fixture. No open redistribution license verified; see docs/ASSETS.md.');
  for (const asset of assets.filter(asset => selected === 'all' || asset.id === selected)) {
    await fetchAsset(asset, join(root, 'public', 'samples'));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
