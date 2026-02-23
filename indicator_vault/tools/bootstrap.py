from __future__ import annotations

from .ingestion_pipeline import IndicatorWarehouse


def main() -> None:
    IndicatorWarehouse().bootstrap()


if __name__ == "__main__":
    main()
