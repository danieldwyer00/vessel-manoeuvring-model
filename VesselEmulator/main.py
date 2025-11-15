# start fastapi and run vessel emulator

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import threading
from VesselEmulator.VesselEmulator import AutonomyLoop  # your code

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = {}
    app.state.autonomy = AutonomyLoop(app.state.store)
    thread = threading.Thread(target=app.state.autonomy.loop, daemon=True)
    thread.start()
    yield
    app.state.autonomy.stop()
    thread.join()

app = FastAPI(lifespan=lifespan)

@app.get("/latest")
def get_latest(request: Request):
    return request.app.state.store.get('latest')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("VesselEmulator.main:app", host="0.0.0.0", port=8000, reload=True)