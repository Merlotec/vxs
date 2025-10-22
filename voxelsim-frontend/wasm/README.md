# WASM Directory

This directory should contain the compiled WASM output from `voxelsim-renderer`.

## Setup

After building the Rust WASM renderer, copy the output here:

```bash
# From the voxelsim-renderer directory
wasm-pack build --target web

# Copy the output to this directory
cp -r pkg/* ../voxelsim-frontend/wasm/
```

Or use the provided build script:

```bash
# From the root of voxelsim-frontend
npm run build:wasm
```

## Expected Files

After building, this directory should contain:
- `voxelsim-renderer.js` - JavaScript bindings
- `voxelsim-renderer_bg.wasm` - WebAssembly binary
- `voxelsim-renderer.d.ts` - TypeScript definitions
- Other supporting files
