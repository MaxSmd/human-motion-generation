// Minimal NumPy .npy reader for the browser. The backend dumps joint positions
// (T, 22, 3) as C-order float32 next to every render; we fetch + parse them to
// animate the skeleton in 3D (no serverside GIF needed for the in-browser view).
//
// Supports .npy v1.0/v2.0 headers and <f4 / <f8 dtypes, C-order only (which is
// what np.save produces by default for our arrays).

export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  // magic: \x93NUMPY
  if (bytes[0] !== 0x93 || String.fromCharCode(bytes[1], bytes[2], bytes[3]) !== "NUM") {
    throw new Error("not a .npy file");
  }
  const major = bytes[6];
  let headerLen, headerStart;
  if (major <= 1) {
    headerLen = bytes[8] | (bytes[9] << 8);
    headerStart = 10;
  } else {
    headerLen = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    headerStart = 12;
  }
  const header = new TextDecoder("latin1").decode(bytes.subarray(headerStart, headerStart + headerLen));

  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === "True";
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeStr.split(",").map((s) => s.trim()).filter(Boolean).map(Number);

  if (fortran) throw new Error("fortran-order .npy not supported");

  const dataStart = headerStart + headerLen; // 64-byte aligned ⇒ float-aligned
  let data;
  if (descr === "<f4" || descr === "|f4") {
    data = new Float32Array(buffer.slice(dataStart));
  } else if (descr === "<f8") {
    data = Float32Array.from(new Float64Array(buffer.slice(dataStart)));
  } else {
    throw new Error(`unsupported .npy dtype ${descr}`);
  }
  return { shape, data };
}

export async function loadNpy(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load joints (${res.status})`);
  return parseNpy(await res.arrayBuffer());
}
