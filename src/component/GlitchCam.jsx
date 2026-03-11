import { useState, useRef, useEffect, useCallback } from "react";

// ===================== GLSL SHADERS =====================

const VERT = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_uv;
void main() {
  v_uv = a_texCoord;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAG_GLITCH = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_time;
uniform float u_intensity;

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
  float amt = u_intensity * 0.04;
  float t = u_time;
  float block = floor(v_uv.y * 20.0 + t * 3.0);
  float glitchLine = step(0.92 - u_intensity * 0.15, rand(vec2(block, floor(t * 8.0))));
  float shift = glitchLine * (rand(vec2(block, t)) - 0.5) * amt * 6.0;
  vec2 uv = v_uv;
  uv.x += shift;
  float split = amt * (0.5 + 0.5 * sin(t * 5.0));
  float r = texture2D(u_tex, uv + vec2(split, 0.0)).r;
  float g = texture2D(u_tex, uv).g;
  float b = texture2D(u_tex, uv - vec2(split, 0.0)).b;
  float scanline = 0.95 + 0.05 * sin(v_uv.y * 800.0);
  float noise = rand(v_uv + fract(t)) * 0.06 * u_intensity;
  vec3 col = vec3(r, g, b) * scanline + noise;
  gl_FragColor = vec4(col, 1.0);
}`;

const FRAG_WARP = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_time;
uniform float u_intensity;

void main() {
  vec2 center = vec2(0.5);
  vec2 uv = v_uv;
  vec2 d = uv - center;
  float dist = length(d);
  float wave = sin(dist * 25.0 - u_time * 4.0) * u_intensity * 0.03;
  float swirl = u_intensity * 0.4 * smoothstep(0.5, 0.0, dist);
  float angle = atan(d.y, d.x) + swirl * sin(u_time * 1.5);
  float r = dist + wave;
  uv = center + vec2(cos(angle), sin(angle)) * r;
  uv = clamp(uv, 0.0, 1.0);
  vec3 col = texture2D(u_tex, uv).rgb;
  float ca = wave * 2.0;
  col.r = texture2D(u_tex, uv + vec2(ca, 0.0)).r;
  col.b = texture2D(u_tex, uv - vec2(ca, 0.0)).b;
  gl_FragColor = vec4(col, 1.0);
}`;

const FRAG_PIXEL = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_time;
uniform float u_intensity;
uniform vec2 u_resolution;

void main() {
  float baseSize = mix(2.0, 40.0, u_intensity);
  float pulse = baseSize + sin(u_time * 2.0) * u_intensity * 4.0;
  vec2 size = vec2(pulse) / u_resolution;
  vec2 uv = floor(v_uv / size + 0.5) * size;
  vec3 col = texture2D(u_tex, uv).rgb;
  vec2 cellUV = fract(v_uv / size);
  float diamond = abs(cellUV.x - 0.5) + abs(cellUV.y - 0.5);
  float border = smoothstep(0.48, 0.5, diamond) * 0.3 * u_intensity;
  col -= border;
  float levels = mix(32.0, 4.0, u_intensity * 0.6);
  col = floor(col * levels + 0.5) / levels;
  gl_FragColor = vec4(col, 1.0);
}`;

const FRAG_BLOOM = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_time;
uniform float u_intensity;
uniform vec2 u_resolution;

void main() {
  vec2 texel = 1.0 / u_resolution;
  float radius = u_intensity * 8.0;
  vec3 sum = vec3(0.0);
  float total = 0.0;
  for(float x = -1.0; x <= 1.0; x += 1.0) {
    for(float y = -1.0; y <= 1.0; y += 1.0) {
      vec2 off = vec2(x, y) * texel * radius;
      float w = 1.0 - length(vec2(x,y)) * 0.3;
      sum += texture2D(u_tex, v_uv + off).rgb * w;
      total += w;
    }
  }
  vec3 blurred = sum / total;
  vec3 original = texture2D(u_tex, v_uv).rgb;
  vec3 bright = max(original - 0.5, 0.0) * 2.0;
  vec3 bloom = blurred * bright;
  float glow = u_intensity * 0.6;
  vec3 col = mix(original, original + bloom + blurred * 0.15, glow);
  float vig = 1.0 - smoothstep(0.4, 0.9, length(v_uv - 0.5));
  col *= mix(1.0, vig, u_intensity * 0.4);
  col.r += u_intensity * 0.03;
  col.b -= u_intensity * 0.02;
  gl_FragColor = vec4(col, 1.0);
}`;

const FRAG_NONE = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
void main() {
  gl_FragColor = texture2D(u_tex, v_uv);
}`;

const EFFECTS = {
  glitch: { frag: FRAG_GLITCH, label: "GLITCH" },
  warp: { frag: FRAG_WARP, label: "WARP" },
  pixel: { frag: FRAG_PIXEL, label: "PIXEL" },
  bloom: { frag: FRAG_BLOOM, label: "BLOOM" },
  none: { frag: FRAG_NONE, label: "CLEAN" },
};

const EFFECT_KEYS = Object.keys(EFFECTS);

// ===================== WebGL HELPERS =====================

function compileShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

function linkProgram(gl, vSrc, fSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fSrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

// ===================== COMPONENT =====================

export default function GlitchCam() {
  const [started, setStarted] = useState(false);
  const [currentFx, setCurrentFx] = useState("glitch");
  const [intensity, setIntensity] = useState(50);
  const [showLabel, setShowLabel] = useState(false);
  const [flashActive, setFlashActive] = useState(false);
  const [toastVisible, setToastVisible] = useState(false);

  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const glRef = useRef(null);
  const programsRef = useRef({});
  const textureRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const animRef = useRef(null);
  const startTimeRef = useRef(Date.now());
  const facingRef = useRef("environment");
  const fxRef = useRef("glitch");
  const intensityRef = useRef(0.5);
  const labelTimerRef = useRef(null);
  const toastTimerRef = useRef(null);
  const touchStartRef = useRef(0);

  // Keep refs in sync with state for the render loop
  useEffect(() => { fxRef.current = currentFx; }, [currentFx]);
  useEffect(() => { intensityRef.current = intensity / 100; }, [intensity]);

  // ---- WebGL init ----
  const initGL = useCallback(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext("webgl", { preserveDrawingBuffer: true });
    if (!gl) { alert("WebGL not supported"); return; }
    glRef.current = gl;

    const verts = new Float32Array([
      -1, -1, 0, 1,
       1, -1, 1, 1,
      -1,  1, 0, 0,
       1,  1, 1, 0,
    ]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

    const progs = {};
    for (const [name, fx] of Object.entries(EFFECTS)) {
      progs[name] = linkProgram(gl, VERT, fx.frag);
    }
    programsRef.current = progs;

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    textureRef.current = tex;
  }, []);

  // ---- Camera ----
  const startCamera = useCallback(async () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingRef.current, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = s;
      const v = document.createElement("video");
      v.setAttribute("playsinline", "");
      v.srcObject = s;
      await v.play();
      videoRef.current = v;
      resizeCanvas();
    } catch (e) {
      alert("Camera access denied: " + e.message);
    }
  }, []);

  const resizeCanvas = useCallback(() => {
    const gl = glRef.current;
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!gl || !wrap || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = wrap.clientWidth * dpr;
    canvas.height = wrap.clientHeight * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }, []);

  // ---- Render loop ----
  const renderLoop = useCallback(() => {
    const gl = glRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!gl || !video || video.readyState < 2) {
      animRef.current = requestAnimationFrame(renderLoop);
      return;
    }

    gl.bindTexture(gl.TEXTURE_2D, textureRef.current);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    const prog = programsRef.current[fxRef.current];
    gl.useProgram(prog);

    const pos = gl.getAttribLocation(prog, "a_position");
    const tex = gl.getAttribLocation(prog, "a_texCoord");
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 16, 0);
    if (tex >= 0) {
      gl.enableVertexAttribArray(tex);
      gl.vertexAttribPointer(tex, 2, gl.FLOAT, false, 16, 8);
    }

    const timeL = gl.getUniformLocation(prog, "u_time");
    const intL = gl.getUniformLocation(prog, "u_intensity");
    const resL = gl.getUniformLocation(prog, "u_resolution");
    if (timeL) gl.uniform1f(timeL, (Date.now() - startTimeRef.current) / 1000);
    if (intL) gl.uniform1f(intL, intensityRef.current);
    if (resL) gl.uniform2f(resL, canvas.width, canvas.height);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    animRef.current = requestAnimationFrame(renderLoop);
  }, []);

  // ---- Start ----
  const handleStart = useCallback(async () => {
    initGL();
    await startCamera();
    setStarted(true);
    startTimeRef.current = Date.now();
    renderLoop();
  }, [initGL, startCamera, renderLoop]);

  // ---- Cleanup ----
  useEffect(() => {
    const onResize = () => { if (glRef.current) resizeCanvas(); };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      if (animRef.current) cancelAnimationFrame(animRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, [resizeCanvas]);

  // ---- Actions ----
  const flipCamera = useCallback(async () => {
    facingRef.current = facingRef.current === "environment" ? "user" : "environment";
    await startCamera();
  }, [startCamera]);

  const switchEffect = useCallback((fx) => {
    setCurrentFx(fx);
    setShowLabel(true);
    clearTimeout(labelTimerRef.current);
    labelTimerRef.current = setTimeout(() => setShowLabel(false), 1200);
  }, []);

  const capture = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setFlashActive(true);
    setTimeout(() => setFlashActive(false), 120);

    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `glitchcam_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
      setToastVisible(true);
      clearTimeout(toastTimerRef.current);
      toastTimerRef.current = setTimeout(() => setToastVisible(false), 2000);
    }, "image/png");
  }, []);

  // ---- Swipe on canvas ----
  const onTouchStart = useCallback((e) => {
    touchStartRef.current = e.touches[0].clientX;
  }, []);

  const onTouchEnd = useCallback((e) => {
    const dx = e.changedTouches[0].clientX - touchStartRef.current;
    if (Math.abs(dx) < 60) return;
    const idx = EFFECT_KEYS.indexOf(fxRef.current);
    const next = dx < 0
      ? (idx + 1) % EFFECT_KEYS.length
      : (idx - 1 + EFFECT_KEYS.length) % EFFECT_KEYS.length;
    switchEffect(EFFECT_KEYS[next]);
  }, [switchEffect]);

  // ===================== STYLES =====================

  const styles = {
    wrapper: {
      display: "flex", flexDirection: "column", height: "100dvh",
      background: "#0a0a0b", color: "#e8e8ec",
      fontFamily: "'Space Mono', monospace", userSelect: "none", overflow: "hidden",
    },
    canvasWrap: {
      flex: 1, position: "relative", overflow: "hidden", background: "#000",
    },
    canvas: { width: "100%", height: "100%", display: "block" },
    topBar: {
      position: "absolute", top: 0, left: 0, right: 0, padding: "12px 16px",
      display: "flex", justifyContent: "space-between", alignItems: "center",
      zIndex: 10, background: "linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)",
    },
    logo: {
      fontFamily: "'Anybody', sans-serif", fontSize: 20, fontWeight: 900,
      letterSpacing: 3, textTransform: "uppercase",
      background: "linear-gradient(135deg, #ff3366, #00eeff)",
      WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
    },
    flipBtn: {
      width: 40, height: 40, background: "rgba(255,255,255,0.1)",
      backdropFilter: "blur(10px)", border: "1px solid rgba(255,255,255,0.15)",
      borderRadius: "50%", color: "#fff", fontSize: 18,
      display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer",
    },
    effectLabel: {
      position: "absolute", top: 60, left: "50%", transform: "translateX(-50%)",
      fontFamily: "'Anybody', sans-serif", fontSize: 14, fontWeight: 900,
      letterSpacing: 4, textTransform: "uppercase", color: "#fff",
      textShadow: "0 0 20px rgba(255,51,102,0.6)",
      opacity: showLabel ? 1 : 0, transition: "opacity 0.3s",
      zIndex: 10, pointerEvents: "none",
    },
    sliderWrap: {
      position: "absolute", right: 16, top: "50%", transform: "translateY(-50%)",
      zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
    },
    sliderLabel: {
      fontSize: 9, letterSpacing: 2, color: "#68687a",
      writingMode: "vertical-rl", textOrientation: "mixed", transform: "rotate(180deg)",
    },
    flash: {
      position: "absolute", inset: 0, background: "#fff",
      opacity: flashActive ? 0.8 : 0, pointerEvents: "none", zIndex: 20,
      transition: flashActive ? "none" : "opacity 0.08s",
    },
    toast: {
      position: "absolute", bottom: 200, left: "50%",
      transform: `translateX(-50%) translateY(${toastVisible ? 0 : 20}px)`,
      background: "rgba(0,0,0,0.85)", backdropFilter: "blur(10px)",
      padding: "10px 24px", borderRadius: 20, border: "1px solid #2a2a30",
      fontSize: 12, letterSpacing: 1, color: "#bbff33",
      opacity: toastVisible ? 1 : 0, zIndex: 30, pointerEvents: "none",
      transition: "opacity 0.3s, transform 0.3s",
    },
    controls: {
      background: "#151518", borderTop: "1px solid #2a2a30",
      padding: "12px 8px 28px", display: "flex", flexDirection: "column", gap: 14,
    },
    effectStrip: {
      display: "flex", gap: 8, justifyContent: "center", overflowX: "auto", padding: "0 8px",
    },
    actionRow: {
      display: "flex", justifyContent: "center", alignItems: "center", gap: 32,
    },
    // Start overlay
    overlay: {
      position: "absolute", inset: 0, background: "#0a0a0b",
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", zIndex: 50, gap: 20,
      opacity: started ? 0 : 1, pointerEvents: started ? "none" : "auto",
      transition: "opacity 0.5s",
    },
    startTitle: {
      fontFamily: "'Anybody', sans-serif", fontSize: 36, fontWeight: 900,
      letterSpacing: 6,
      background: "linear-gradient(135deg, #ff3366, #00eeff, #bbff33)",
      WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
    },
    startSub: { color: "#68687a", fontSize: 12, letterSpacing: 2 },
    startBtn: {
      marginTop: 16, padding: "14px 36px", background: "#ff3366", border: "none",
      borderRadius: 30, color: "#fff", fontFamily: "'Space Mono', monospace",
      fontSize: 13, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase",
      cursor: "pointer", boxShadow: "0 4px 24px rgba(255,51,102,0.4)",
    },
  };

  const fxBtnStyle = (active) => ({
    flexShrink: 0, padding: "8px 16px",
    background: active ? "#ff3366" : "rgba(255,255,255,0.05)",
    border: `1px solid ${active ? "#ff3366" : "#2a2a30"}`,
    borderRadius: 20, color: active ? "#fff" : "#68687a",
    fontFamily: "'Space Mono', monospace", fontSize: 11,
    letterSpacing: 1, textTransform: "uppercase", cursor: "pointer",
    transition: "all 0.25s",
    boxShadow: active ? "0 0 16px rgba(255,51,102,0.4)" : "none",
  });

  // ===================== JSX =====================

  return (
    <div style={styles.wrapper}>
      <link
        href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Anybody:wght@900&display=swap"
        rel="stylesheet"
      />

      <div ref={wrapRef} style={styles.canvasWrap}>
        <canvas
          ref={canvasRef}
          style={styles.canvas}
          onTouchStart={onTouchStart}
          onTouchEnd={onTouchEnd}
        />

        {/* Top bar */}
        <div style={styles.topBar}>
          <div style={styles.logo}>Glitch Cam</div>
          <button style={styles.flipBtn} onClick={flipCamera}>⟲</button>
        </div>

        {/* Effect label */}
        <div style={styles.effectLabel}>{EFFECTS[currentFx].label}</div>

        {/* Intensity slider */}
        <div style={styles.sliderWrap}>
          <input
            type="range"
            min="0" max="100"
            value={intensity}
            onChange={(e) => setIntensity(Number(e.target.value))}
            style={{
              WebkitAppearance: "none", appearance: "none",
              writingMode: "vertical-lr", direction: "rtl",
              width: 32, height: 160, background: "transparent",
            }}
          />
          <span style={styles.sliderLabel}>FX</span>
        </div>

        {/* Flash */}
        <div style={styles.flash} />

        {/* Toast */}
        <div style={styles.toast}>✓ SAVED TO PHOTOS</div>

        {/* Start overlay */}
        <div style={styles.overlay}>
          <div style={styles.startTitle}>GLITCH CAM</div>
          <div style={styles.startSub}>REAL-TIME SHADER EFFECTS</div>
          <button style={styles.startBtn} onClick={handleStart}>OPEN CAMERA</button>
        </div>
      </div>

      {/* Bottom controls */}
      <div style={styles.controls}>
        <div style={styles.effectStrip}>
          {EFFECT_KEYS.map((key) => (
            <button
              key={key}
              style={fxBtnStyle(currentFx === key)}
              onClick={() => switchEffect(key)}
            >
              {EFFECTS[key].label}
            </button>
          ))}
        </div>
        <div style={styles.actionRow}>
          <button onClick={capture} style={{ all: "unset", cursor: "pointer" }}>
            <div style={{
              width: 64, height: 64, borderRadius: "50%", border: "3px solid #fff",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <div style={{
                width: 48, height: 48, borderRadius: "50%", background: "#fff",
                transition: "background 0.15s",
              }} />
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
