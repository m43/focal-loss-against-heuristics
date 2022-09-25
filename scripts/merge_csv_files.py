import os

CSV_PATHS = [
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/config.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_Last Step.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_MNLI.V.M.acc.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_MNLI.V.M.loss.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_MNLI.V.MM.acc.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_MNLI.V.MM.loss.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_SNLI.V.acc.csv",
    "/mnt/terra/xoding/epfl-optml/logs/S01.results/report_metric.earlystopping_on_SNLI.V.loss.csv",
]
MERGED_CSV_PATH = "/logs/S01.results/merged.csv"

merged_csv_str = ""
for csv_path in CSV_PATHS:
    assert os.path.exists(csv_path)
    merged_csv_str += f"{os.path.basename(csv_path)}\n"
    with open(csv_path, "r") as f:
        merged_csv_str += f.read()
        merged_csv_str += "\n\n"

print(merged_csv_str)
with open(MERGED_CSV_PATH, "w") as f:
    f.write(merged_csv_str)
