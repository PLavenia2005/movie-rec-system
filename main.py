from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

import recommender


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse(request, "index.html")


@app.get("/api/recommend")
async def api_recommend(title: str):
	try:
		results = recommender.recommend(title, top_n=12)
		return JSONResponse({"query": title, "results": results})
	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/random")
async def api_random(n: int = 12):
	try:
		results = recommender.random_movies(top_n=n)
		return JSONResponse({"results": results})
	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

