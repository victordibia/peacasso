import logging
import typer
import uvicorn
import os


app = typer.Typer()

logging.basicConfig()
logging.getLogger("peacasso").setLevel(logging.INFO)


@app.command()
def ui(
        host: str = "127.0.0.1", port: int = 8081, workers: int = 1, reload: bool = True,
        model: str = typer.Option(
            default="runwayml/stable-diffusion-v1-5",
            help="Full model name following the HuggingFace format"),
        revision: str = typer.Option(
            default="fp16",
            help="Revision of the model to use. Can be 'fp16' or 'full', or 'main'"),
        device: str = typer.Option(
            default="cuda:0",
            help="Device to use for inference. Can be 'cpu' or 'cuda:0', 'cuda:1' for example"),
        log_level: str = typer.Option(
            default="INFO",
            help="Logging level. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

):
    """
    Launch the peacasso UI. Pass in parameters host, port, workers, reload, model to override the default values.
    """
    # set log level
    logging.getLogger("peacasso").setLevel(log_level)
    os.environ["PEACASSO_MODEL"] = model
    os.environ["PEACASSO_DEVICE"] = device
    os.environ["PEACASSO_REVISION"] = revision
    uvicorn.run(
        "peacasso.web.backend.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def list():
    print("list")


def run():
    app()


if __name__ == "__main__":

    app()
