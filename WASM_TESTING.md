# VoxelSim WASM Renderer Testing Guide

## 완성된 구성 요소

✅ **WebSocket 프록시 서버** (Node.js)
- 위치: `proxy/proxy.js`
- TCP → WebSocket 변환
- 4개 포트 브릿지: 8080→9080, 8081→9081, 8090→9090, 9090→10090

✅ **WASM 렌더러** (Rust/Bevy)
- 위치: `voxelsim-renderer/`
- WebSocket 클라이언트 구현
- Native 코드와 동일한 렌더링 로직

✅ **웹 인터페이스** (HTML)
- 위치: `voxelsim-renderer/index.html`
- 연결 상태 표시
- WASM 로딩 및 초기화

---

## 테스트 절차

### 1단계: 프록시 서버 시작

```bash
cd proxy
npm start
```

**기대 출력:**
```
VoxelSim WebSocket Proxy Server Starting...

[VoxelGrid] Proxy started: TCP :8080 ←→ WebSocket :9080
[Agents] Proxy started: TCP :8081 ←→ WebSocket :9081
[POV World] Proxy started: TCP :8090 ←→ WebSocket :9090
[POV Agents] Proxy started: TCP :9090 ←→ WebSocket :10090

✓ All proxies running!
```

**오류 메시지 (정상):**
- "TCP connection error: ECONNREFUSED" → Python이 아직 실행 안 됨
- 2초마다 자동 재연결 시도

---

### 2단계: HTTP 서버 시작 (WASM 서빙용)

새 터미널에서:

```bash
cd voxelsim-renderer
python3 -m http.server 8000
```

또는:

```bash
# npx를 사용하면 CORS 문제 없음
npx http-server -p 8000 -c-1
```

**브라우저에서 확인:**
- http://localhost:8000

---

### 3단계: Python 시뮬레이션 시작

새 터미널에서:

```bash
cd python
python povtest.py
# 또는
python sim_from_world.py
```

**기대 동작:**
1. Python이 TCP :8080, :8081에 연결
2. 프록시 로그에 "Connected to Python TCP" 표시
3. 브라우저에서 실시간 렌더링 시작!

---

## 아키텍처 흐름

```
[Python Simulation]
    ↓ TCP :8080 (VoxelGrid)
    ↓ TCP :8081 (Agents)
[Proxy Server]
    ↓ WebSocket :9080
    ↓ WebSocket :9081
[Browser: http://localhost:8000]
    ↓ WASM Renderer
    ↓ Bevy ECS
[Canvas 렌더링]
```

---

## 디버깅

### 브라우저 콘솔 확인

F12 → Console 탭:

**정상 출력:**
```
Starting VoxelSim Renderer (WASM WebSocket Mode)...
✓ WebSocket connected: ws://localhost:9080
✓ WebSocket connected: ws://localhost:9081
VoxelSim Renderer initialized!
```

**오류 케이스:**

1. **"WebSocket error: Connection refused"**
   - 원인: 프록시 서버 미실행
   - 해결: `cd proxy && npm start`

2. **"Failed to fetch dynamically imported module"**
   - 원인: HTTP 서버 미실행 또는 경로 오류
   - 해결: `python3 -m http.server 8000`

3. **"Failed to deserialize message"**
   - 원인: 데이터 포맷 불일치
   - 확인: Python과 Rust의 bincode 버전 동일한지 확인

---

## 프록시 로그 읽기

### 정상 흐름:
```
[VoxelGrid] Attempting to reconnect to TCP :8080...
[VoxelGrid] Connected to Python TCP :8080
[VoxelGrid] Browser connected via WebSocket :9080
```

### 문제 흐름:
```
[VoxelGrid] TCP connection error: connect ECONNREFUSED
→ Python이 실행 안 됨

[VoxelGrid] TCP connection closed, will retry...
→ Python이 중단됨, 자동 재연결 시도
```

---

## 성능 확인

브라우저 콘솔에서:

```javascript
// FPS 확인
setInterval(() => {
    console.log('FPS:', Math.round(1000 / performance.now()));
}, 1000);

// WebSocket 상태 확인
console.log('WebSocket state:', performance.getEntriesByType('resource'));
```

---

## 네트워크 분석

### Chrome DevTools:
1. F12 → Network 탭
2. WS 필터 클릭
3. WebSocket 연결 확인:
   - `ws://localhost:9080` (VoxelGrid)
   - `ws://localhost:9081` (Agents)

### 메시지 확인:
- 메시지 클릭 → Data 탭
- Binary 데이터 표시됨 (정상)
- 4바이트 길이 + bincode payload

---

## Native vs WASM 비교

| 항목 | Native | WASM |
|------|--------|------|
| **네트워크** | TCP 직접 | WebSocket (프록시 경유) |
| **렌더링** | Desktop Window | Browser Canvas |
| **성능** | ~60 FPS | ~30-60 FPS (브라우저 제약) |
| **배포** | 바이너리 배포 필요 | URL 공유로 즉시 접속 |
| **디버깅** | gdb/lldb | Browser DevTools |

---

## 다음 단계

### 추가 기능:
- [ ] POV (1인칭 시점) WebSocket 지원
- [ ] 양방향 통신 (브라우저 → Python 명령)
- [ ] 멀티플레이어 (여러 브라우저 동시 접속)
- [ ] 녹화 기능 (Canvas → Video)

### 최적화:
- [ ] WASM 파일 크기 최소화 (현재 ~20MB)
- [ ] 데이터 압축 (gzip/brotli)
- [ ] 프레임 스킵 로직
- [ ] WebGPU 지원 (더 빠른 렌더링)

---

## 문제 해결 체크리스트

```bash
# 1. 프록시 실행 중?
lsof -i :9080

# 2. HTTP 서버 실행 중?
lsof -i :8000

# 3. Python 실행 중?
ps aux | grep python

# 4. WASM 파일 존재?
ls -lh voxelsim-renderer/wasm/*.wasm

# 5. 빌드 재시도
cd voxelsim-renderer
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen --out-dir wasm --target web target/wasm32-unknown-unknown/release/voxelsim-renderer.wasm
```

---

## 알려진 제약사항

1. **싱글 클라이언트**: 현재 구조는 마지막 연결 브라우저만 데이터 수신
2. **로컬호스트 전용**: 외부 접속 불가 (보안상 제한)
3. **재연결 미지원**: 브라우저 새로고침 시 Python 재시작 필요
4. **키보드 입력**: 렌더러에서 시뮬레이션 조작 불가 (단방향)

---

## 참고 자료

- **Proxy 코드**: [proxy/proxy.js](proxy/proxy.js)
- **WASM 네트워크**: [voxelsim-renderer/src/network_wasm.rs](voxelsim-renderer/src/network_wasm.rs)
- **HTML 인터페이스**: [voxelsim-renderer/index.html](voxelsim-renderer/index.html)
- **Bevy WASM 가이드**: https://bevyengine.org/learn/quick-start/getting-started/wasm/
