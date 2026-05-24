from pathlib import Path
from ocr_worker import OCRWorker

file_path = Path(r"C:\Users\Dell\Downloads\PublicWaterMassMailing.pdf")  # change this

worker = OCRWorker(debug=False)
pages = worker.process_file(file_path, dpi=300)
response = worker.to_response(pages)

for page in response["pages"]:
    print(f"\n--- Page {page['page_number']} ---")
    print(page["text"])
    print("\nMetadata:")
    print(page["metadata"])

print("\nTotal pages:", response["page_count"])
