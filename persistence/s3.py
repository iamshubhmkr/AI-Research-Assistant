"""S3 — Raw PDFs + golden dataset storage."""
import boto3, json
from config import settings

s3 = boto3.client("s3", region_name=settings.aws_region)

class PaperStore:
    def upload_pdf(self, paper_id: str, pdf_bytes: bytes) -> str:
        key = f"papers/{paper_id}.pdf"
        s3.put_object(Bucket=settings.s3_bucket, Key=key, Body=pdf_bytes)
        return key

    def save_golden_dataset(self, dataset: list):
        s3.put_object(Bucket=settings.s3_bucket, Key="evaluation/golden_dataset.json",
                      Body=json.dumps(dataset, indent=2).encode())

    def load_golden_dataset(self) -> list:
        try:
            obj = s3.get_object(Bucket=settings.s3_bucket, Key="evaluation/golden_dataset.json")
            return json.loads(obj["Body"].read())
        except Exception:
            return []

    def append_golden(self, entry: dict):
        existing = self.load_golden_dataset()
        existing.append(entry)
        self.save_golden_dataset(existing)
