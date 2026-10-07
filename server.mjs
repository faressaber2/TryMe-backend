// server.mjs — Node/Express backend for TryMe

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Client } from "@gradio/client";

import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Directory to store temporary reverse-search images
const PUBLIC_DIR = path.join(__dirname, "public");
const REVERSE_DIR = path.join(PUBLIC_DIR, "reverse");
fs.mkdirSync(REVERSE_DIR, { recursive: true });

const app = express();

// Serve /public so Google can access the images
app.use("/public", express.static(PUBLIC_DIR));

// Allow JSON bodies up to ~10MB (for data URLs)
app.use(express.json({ limit: "25mb" }));
// Simple CORS so extension can call this
app.use(cors());

// Lazy-init Gradio client so we reuse the same connection
let clientPromise = null;
function getGradioClient() {
  if (!clientPromise) {
    const token = process.env.HF_TOKEN || undefined; // اختياري: توكن Hugging Face بيزوّد كوتا الـ GPU
    clientPromise = Client.connect("yisol/IDM-VTON", token ? { hf_token: token } : {}).catch((e) => {
      clientPromise = null; // لو الاتصال فشل مرة ماتفضلش مخزّنه للأبد
      throw e;
    });
  }
  return clientPromise;
}

// Gradio بيرمي object مش Error، فـ String(err) كان بيطلّع "[object Object]". ده بيطلّع السبب الحقيقي.
function errText(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  if (typeof err.message === "string" && err.message) return err.message;
  try {
    return JSON.stringify(err, Object.getOwnPropertyNames(err)).slice(0, 800);
  } catch {
    return String(err);
  }
}

// يحوّل السبب لكود HTTP مناسب عشان الواجهة تفرّق بين الزحمة والعطل الحقيقي
function errStatus(text) {
  if (/quota|zerogpu|exceeded.*gpu|gpu.*(limit|exceeded)/i.test(text)) return 429;
  if (/queue|busy|too many|overload|paused|sleep|building|starting|unavailable|503|timeout|timed out/i.test(text)) return 503;
  return 500;
}

// Helper: convert data:image/...;base64,... → Buffer
function dataUrlToBuffer(dataUrl) {
  const parts = dataUrl.split(",");
  if (parts.length !== 2) {
    throw new Error("Invalid data URL");
  }
  const base64 = parts[1];
  return Buffer.from(base64, "base64");
}

// POST /tryon  { selfieDataUrl, garmentDataUrl }
app.post("/tryon", async (req, res) => {
  try {
    const body = req.body || {};
    const selfieDataUrl = body.selfieDataUrl || body.human_img;
    const garmentDataUrl = body.garmentDataUrl || body.garm_img;
    if (!selfieDataUrl || !garmentDataUrl) {
      return res
        .status(400)
        .json({ ok: false, error: "Missing selfieDataUrl or garmentDataUrl" });
    }

    const client = await getGradioClient();

    const humanBuffer = dataUrlToBuffer(selfieDataUrl);
    const garmentBuffer = dataUrlToBuffer(garmentDataUrl);

    console.log("Calling IDM-VTON /tryon...");
    const result = await client.predict("/tryon", [
      { background: humanBuffer, layers: [], composite: null }, // human
      garmentBuffer,                                            // garment
      "Virtual try-on from TryMe",                             // text prompt
      true,                                                    // auto mask
      false,                                                   // auto crop
      30,                                                      // denoising steps
      42                                                       // seed
    ]);

    console.log("Raw result from IDM-VTON:", JSON.stringify(result));

    let [outputImage, maskedImage] = result.data || [];

    // ---------- NORMALIZE OUTPUT SO THE BROWSER CAN LOAD IT ----------

    function normalizeImage(img) {
      if (!img) return null;

      // If Gradio returns an object, try common fields
      if (typeof img === "object") {
        const candidate = img.url || img.path || img.image || null;
        if (!candidate) return null;
        img = candidate;
      }

      if (typeof img !== "string") return null;

      // Already a data URL? Use as-is.
      if (img.startsWith("data:image")) {
        return img;
      }

      // Already a full URL? Use as-is.
      if (/^https?:\/\//.test(img)) {
        return img;
      }

      // Paths like "file=/tmp/gradio/..." or "/file=/tmp/..."
      if (img.startsWith("file=") || img.startsWith("/file=")) {
        const trimmed = img.replace(/^\/+/, ""); // remove leading '/'
        const base = "https://yisol-idm-vton.hf.space/";
        return base + trimmed;
      }

      // Fallback: treat as a relative path on the Space
      return "https://yisol-idm-vton.hf.space/" + img.replace(/^\/+/, "");
    }

    const normalizedOutput = normalizeImage(outputImage);
    const normalizedMasked = normalizeImage(maskedImage);

    if (!normalizedOutput) {
      console.error("Could not normalize IDM-VTON output:", outputImage);
      return res.status(500).json({
        ok: false,
        error: "IDM-VTON returned an unsupported image format",
        raw: result
      });
    }

    return res.json({
      ok: true,
      result: normalizedOutput,
      masked: normalizedMasked
    });
  } catch (err) {
    const text = errText(err);
    console.error("TryOn error:", text, err);
    return res.status(errStatus(text)).json({
      ok: false,
      error: "Backend error",
      details: text
    });
  }
});

// POST /reverse-search  { garmentDataUrl }
app.post("/reverse-search", async (req, res) => {
  try {
    const { garmentDataUrl } = req.body || {};
    if (!garmentDataUrl) {
      return res
        .status(400)
        .json({ ok: false, error: "Missing garmentDataUrl" });
    }

    // Reuse your data URL → Buffer helper
    const buffer = dataUrlToBuffer(garmentDataUrl);

    const id =
      crypto.randomUUID?.() ?? crypto.randomBytes(16).toString("hex");
    const filename = `${id}.jpg`;
    const filePath = path.join(REVERSE_DIR, filename);

    await fs.promises.writeFile(filePath, buffer);

    // Build public URL like: https://tryme-backend-fapp.onrender.com/public/reverse/<file>
    const imageUrl = `${req.protocol}://${req.get(
      "host"
    )}/public/reverse/${filename}`;

    const encodedImageUrl = encodeURIComponent(imageUrl);
    const googleUrl = `https://lens.google.com/uploadbyurl?url=${encodedImageUrl}&hl=en`;

    return res.json({
      ok: true,
      imageUrl,
      googleUrl,
    });
  } catch (err) {
    console.error("Reverse search error:", err);
    return res.status(500).json({
      ok: false,
      error: "Backend error during reverse search",
      details: errText(err),
    });
  }
});

app.get("/health", (_req, res) => res.json({ ok: true }));

// أخطاء قراءة الـ body (حجم كبير / JSON باظ) ترجع JSON بدل صفحة HTML
app.use((err, _req, res, next) => {
  if (err && (err.type === "entity.too.large" || err.status === 413)) {
    return res.status(413).json({ ok: false, error: "Image too large" });
  }
  if (err && err.type === "entity.parse.failed") {
    return res.status(400).json({ ok: false, error: "Invalid JSON body" });
  }
  return next(err);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`TryMe backend listening on port ${PORT}`);
});
