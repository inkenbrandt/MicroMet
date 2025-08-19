import typer, yaml
from pathlib import Path
import pandas as pd

from ..io.readers import load_from_ini
from ..io.writers import (
    write_ameriflux_csv,
    UploadCsvOptions,
    write_analysis_csv,
    AnalysisCsvOptions,
    write_parquet,
    ParquetOptions,
)

app = typer.Typer(add_completion=False)


@app.command()
def run_from_ini(
    ini: str = typer.Argument(..., help="Path to MicroMet INI"),
    out_dir: str = typer.Option("./build/output/", help="Output directory"),
    upload_csv: bool = typer.Option(True, help="Write AmeriFlux upload CSV"),
    analysis_csv: bool = typer.Option(True, help="Write analysis CSV"),
    parquet: bool = typer.Option(True, help="Write Parquet"),
    include_units: bool = typer.Option(
        True, help="Include units second header in upload CSV"
    ),
    gzip: bool = typer.Option(False, help="Gzip CSV outputs"),
    flavor: str = typer.Option("auto", help="auto | ameriflux | toa5 | generic"),
):
    df = load_from_ini(ini, enforce_30min=True, flavor_hint=flavor)
    out_dir = Path(out_dir)  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)  # type: ignore

    if upload_csv:
        write_ameriflux_csv(
            df,
            out_dir / "micromet_upload.csv",  # type: ignore
            UploadCsvOptions(include_units_row=include_units, gzip=gzip),
        )
    if analysis_csv:
        write_analysis_csv(
            df,
            out_dir / "micromet_analysis.csv" + (".gz" if gzip else ""),  # type: ignore
            AnalysisCsvOptions(rfc3339_timestamps=True, gzip=gzip),
        )
    if parquet:
        write_parquet(
            df,
            out_dir / "micromet.parquet",  # type: ignore
            ParquetOptions(compression="zstd", metadata={"producer": "MicroMet"}),
        )
    typer.echo(f"Done → {out_dir}")


if __name__ == "__main__":
    app()
