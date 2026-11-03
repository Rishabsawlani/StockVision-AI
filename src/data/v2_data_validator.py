from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# Validate all V2 stock files
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("STOCKVISION-AI V2 DATA VALIDATION")
    print("=" * 70)

    csv_files = list(RAW_DATA_PATH.glob("*.csv"))

    print(f"\nCSV files found: {len(csv_files)}")

    for filepath in csv_files:

        print("\n" + "=" * 70)
        print(f"FILE: {filepath.name}")
        print("=" * 70)

        try:

            data = pd.read_csv(filepath)

            print(f"Rows: {len(data)}")
            print(f"Columns: {list(data.columns)}")

            # --------------------------------------------------
            # Check required columns
            # --------------------------------------------------

            required_columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                print(
                    f"Missing columns: {missing_columns}"
                )

            else:

                print("Required columns: OK")

            # --------------------------------------------------
            # Date validation
            # --------------------------------------------------

            data["Date"] = pd.to_datetime(
                data["Date"],
                errors="coerce"
            )

            invalid_dates = data["Date"].isna().sum()

            print(
                f"Invalid dates: {invalid_dates}"
            )

            duplicate_dates = data["Date"].duplicated().sum()

            print(
                f"Duplicate dates: {duplicate_dates}"
            )

            # --------------------------------------------------
            # Missing values
            # --------------------------------------------------

            missing_values = data[
                required_columns
            ].isna().sum()

            print("\nMissing values:")

            print(
                missing_values[
                    missing_values > 0
                ]
            )

            # --------------------------------------------------
            # Date range
            # --------------------------------------------------

            print(
                f"\nDate range: "
                f"{data['Date'].min().date()} "
                f"to "
                f"{data['Date'].max().date()}"
            )

            # --------------------------------------------------
            # Basic price validation
            # --------------------------------------------------

            if all(
                column in data.columns
                for column in [
                    "Open",
                    "High",
                    "Low",
                    "Close"
                ]
            ):

                invalid_prices = (
                    (data["Open"] <= 0)
                    | (data["High"] <= 0)
                    | (data["Low"] <= 0)
                    | (data["Close"] <= 0)
                ).sum()

                print(
                    f"Invalid price rows: "
                    f"{invalid_prices}"
                )

            # --------------------------------------------------
            # Volume validation
            # --------------------------------------------------

            if "Volume" in data.columns:

                invalid_volume = (
                    data["Volume"] < 0
                ).sum()

                print(
                    f"Invalid volume rows: "
                    f"{invalid_volume}"
                )

        except Exception as e:

            print(
                f"ERROR reading "
                f"{filepath.name}: {e}"
            )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)