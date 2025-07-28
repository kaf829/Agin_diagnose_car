import PyPDF2

pdf_path = "Owner's_Manual.pdf"
output_path = "manual_text.txt"

with open(pdf_path, "rb") as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("PDF 텍스트 추출 완료 -> manual_text.txt")
