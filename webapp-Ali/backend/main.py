import asyncio
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import PROV_DIM, PROV_KEYS, PROV_MU, PROV_SD, IN_DIM
import features
import model
import llm


def _validate():
    assert len(PROV_KEYS) <= PROV_DIM == 32, "PROV_DIM must be 32"
    assert len(PROV_MU) == len(PROV_SD) == 32, "mu/sd must be length 32"
    model.load_model()  # asserts first Linear in_features == IN_DIM == 1198


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate()
    features.init_encoders()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.post("/analyze")
async def analyze(image: UploadFile = File(None), url: str = Form(None)):
    if not image and not url:
        raise HTTPException(400, "Provide an image or a url.")

    prov = None
    if url:
        from screenshot import capture
        try:
            png, html, final_url, host = await capture(url)
        except Exception as e:
            raise HTTPException(502, f"Screenshot capture failed: {e}")
        from provenance import build_provenance
        try:
            prov = build_provenance(html, final_url, host)
        except Exception:
            prov = None  # degraded-provenance fallback
    else:
        png = await image.read()

    image_b64 = base64.b64encode(png).decode("utf-8")

    try:
        verdicts, rationale = llm.joint_verdict(image_b64)
    except Exception as e:
        raise HTTPException(502, f"Joint verdict call failed: {e}")

    x = features.assemble(png, prov, verdicts, rationale)
    label, confidences = model.predict(x)

    return {
        "factuality": {"label": label, "confidences": confidences},
        "screenshot_b64": image_b64,
    }


class ExplainRequest(BaseModel):
    screenshot_b64: str
    factuality_label: str
    bias_label: str | None = None


@app.post("/explain")
async def explain(req: ExplainRequest):
    try:
        text = llm.explain(req.screenshot_b64, req.factuality_label, req.bias_label)
    except Exception as e:
        raise HTTPException(502, f"Explanation call failed: {e}")
    return {"explanation": text}


class BiasRequest(BaseModel):
    screenshot_b64: str


@app.post("/bias")
async def bias(req: BiasRequest):
    """Bias prediction on the screenshot already captured by /analyze.

    Independent of the factuality pipeline: runs the vendored gpt-5.5 + PaddleOCR +
    SigLIP stacked ensemble in a worker thread so the event loop stays responsive.
    """
    try:
        png = base64.b64decode(req.screenshot_b64)
    except Exception:
        raise HTTPException(400, "Invalid screenshot_b64.")

    from bias import service as bias_service
    try:
        result = await asyncio.to_thread(bias_service.analyze, png)
    except Exception as e:
        raise HTTPException(502, f"Bias prediction failed: {e}")
    return {"bias": result}


class EvidenceRequest(BaseModel):
    screenshot_b64: str


@app.post("/evidence")
async def evidence(req: EvidenceRequest):
    """Factuality evidence overlay: bounding boxes + per-region reasons grounding
    an LLM-picked factuality label, ported from Oleg's bias evidence path.

    Slow (PaddleOCR + a vision call), so it is its own endpoint and runs the work
    in a worker thread, keeping /analyze fast. Returns {label, evidence}; the label
    is the LLM's own pick and may differ from the MLP factuality label on the card.

    Takes the screenshot as a JSON body (modelled on /bias) rather than a multipart
    form: a full-page screenshot's base64 exceeds Starlette's 1 MB multipart-part
    limit, whereas a JSON body has no such cap. Operates on the exact image
    /analyze already returned, so boxes draw against the displayed rendering.
    """
    try:
        png = base64.b64decode(req.screenshot_b64)
    except Exception:
        raise HTTPException(400, "Invalid screenshot_b64.")

    from bias import service as bias_service
    try:
        result = await asyncio.to_thread(bias_service.analyze_factuality_evidence, png)
    except Exception as e:
        raise HTTPException(502, f"Factuality evidence failed: {e}")
    return result
