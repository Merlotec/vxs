# Quick Setup Guide

## 1. Install Dependencies

```bash
npm install
```

## 2. Build WASM Renderer

You need to build the Rust WASM renderer first:

```bash
# Option A: Use the npm script (Windows)
npm run build:wasm

# Option B: Manual (cross-platform)
cd ../voxelsim-renderer
wasm-pack build --target web
# Then copy pkg/* to ../voxelsim-frontend/wasm/
```

## 3. Start Development Server

```bash
npm run dev
```

Visit `http://localhost:3000`

## 4. Start Backend (in another terminal)

Make sure your backend/proxy is running on port 9080:

```bash
cd ../voxelsim-proxy
# Start your backend
```

## Project Structure

```
src/
├── components/       # React components
│   ├── Canvas.tsx
│   ├── LoadingScreen.tsx
│   └── StatusBar.tsx
├── services/        # Business logic
│   ├── renderer.ts  # WASM initialization
│   └── websocket.ts # WebSocket connection
├── types/           # TypeScript types
├── styles/          # Global styles
├── App.tsx          # Main app
└── main.tsx         # Entry point
```

## Next Steps

1. ✅ Project structure is ready
2. ⏳ Build WASM renderer: `npm run build:wasm`
3. ⏳ Start dev server: `npm run dev`
4. ⏳ Start backend on port 9080
5. ⏳ Open browser to `http://localhost:3000`

## Troubleshooting

**WASM not found?**
- Run `npm run build:wasm` to build the renderer
- Check that `wasm/voxelsim-renderer.js` exists

**WebSocket not connecting?**
- Make sure backend is running on port 9080
- Check browser console for errors

**TypeScript errors?**
- Run `npx tsc --noEmit` to check types
