# VoxelSim Frontend

Modern web-based frontend for the VoxelSim renderer, built with Vite, React, and TypeScript.

## Architecture

This frontend consumes the WASM-compiled `voxelsim-renderer` and provides:
- Real-time WebSocket connection to simulation backend
- Status monitoring for proxy/backend/simulation
- Clean component-based UI structure
- Fast development experience with Vite

## Tech Stack

- **Vite** - Lightning-fast build tool and dev server
- **React 19** - UI framework
- **TypeScript** - Type safety
- **WebAssembly** - Rust-compiled renderer
- **WebSocket** - Real-time communication

## Project Structure

```
voxelsim-frontend/
├── src/
│   ├── components/         # React components
│   │   ├── Canvas.tsx      # Bevy canvas element
│   │   ├── LoadingScreen.tsx
│   │   └── StatusBar.tsx
│   ├── services/           # Business logic
│   │   ├── renderer.ts     # WASM initialization
│   │   └── websocket.ts    # WebSocket service
│   ├── styles/             # Global styles
│   ├── types/              # TypeScript definitions
│   ├── App.tsx             # Main app component
│   └── main.tsx            # Entry point
├── public/                 # Static assets
├── wasm/                   # WASM output (from voxelsim-renderer)
├── vite.config.ts          # Vite configuration
└── package.json
```

## Prerequisites

1. **Node.js** (v18+ recommended)
2. **npm** or **yarn**
3. **Rust & wasm-pack** (for building the renderer)

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Build the WASM renderer

First, build the Rust WASM renderer from the `voxelsim-renderer` package:

```bash
# Option A: Use the npm script (recommended)
npm run build:wasm

# Option B: Manual build
cd ../voxelsim-renderer
wasm-pack build --target web
xcopy /E /I /Y pkg ..\voxelsim-frontend\wasm
```

This will:
- Compile the Rust code to WASM
- Copy the output to `voxelsim-frontend/wasm/`

### 3. Start the development server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Development

### Running the full stack

1. **Backend/Proxy** (in separate terminal):
   ```bash
   cd ../voxelsim-proxy
   # Start your backend server on port 9080
   ```

2. **Frontend** (this directory):
   ```bash
   npm run dev
   ```

The frontend will automatically detect if the backend is running and display connection status.

### Hot Module Replacement (HMR)

Vite provides instant HMR:
- Edit React components → See changes in <50ms
- Edit styles → Instant update
- TypeScript errors → Shown in browser overlay

### Development vs Production URLs

The WebSocket URL is auto-detected:
- **Development**: `ws://localhost:9080`
- **Production**: `wss://voxelsim-backend.your-domain.com/ws`

Edit the logic in [src/services/websocket.ts](src/services/websocket.ts) if you need custom URLs.

## Building for Production

```bash
# Type-check and build
npm run build

# Preview the production build locally
npm run preview
```

Output will be in the `dist/` directory, ready to deploy to any static hosting service.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server (port 3000) |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run build:wasm` | Build and copy WASM from renderer |
| `npm run clean` | Clean build artifacts |

## Deployment

### Static Hosting (Vercel, Netlify, etc.)

1. Build the project:
   ```bash
   npm run build:wasm
   npm run build
   ```

2. Deploy the `dist/` directory

3. Set environment variables for backend URL if needed

### Docker (example)

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Troubleshooting

### WASM not loading

Make sure you've built the renderer:
```bash
npm run build:wasm
```

Check that `wasm/voxelsim-renderer.js` exists.

### Backend not connecting

1. Check that backend is running on port 9080
2. Check browser console for WebSocket errors
3. Verify the WebSocket URL in the status bar

### TypeScript errors

```bash
# Type-check without building
npx tsc --noEmit
```

## Performance

- **Dev server start**: ~200ms (vs Next.js ~3-8s)
- **HMR**: <50ms
- **Production bundle**: Optimized with Rollup

## Why Vite over Next.js?

This is a **client-side renderer app**, not a full-stack web app:
- ✅ No SSR needed (WASM doesn't work server-side)
- ✅ Faster dev experience (instant startup)
- ✅ Better WASM support (built-in)
- ✅ Simpler configuration
- ✅ Lighter bundle size

## Contributing

When adding new features:
1. Create components in `src/components/`
2. Add services in `src/services/`
3. Define types in `src/types/`
4. Keep styles modular (component-scoped CSS)

## License

[Add your license here]
