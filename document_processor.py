import os
import fitz  # PyMuPDF
import base64
import pandas as pd
import openpyxl
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class DocumentProcessor:
    def load_documents(self):
        """Loads text and images from all PDF and Excel files in the data directory."""
        documents = []
        images_data = []
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Process PDF files first
        for file in os.listdir(data_dir):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(data_dir, file)
                try:
                    pdf_docs, pdf_images = self.process_pdf(file_path)
                    documents.extend(pdf_docs)
                    images_data.extend(pdf_images)
                    print(f"Loaded {len(pdf_docs)} docs and {len(pdf_images)} images from {file}")
                except Exception as e:
                    print(f"Error loading PDF {file_path}: {str(e)}")

        # Then process Excel files
        for file in os.listdir(data_dir):
            if file.lower().endswith((".xlsx", ".xls")):
                file_path = os.path.join(data_dir, file)
                try:
                    excel_docs, excel_images = self.process_excel_file(file_path)
                    documents.extend(excel_docs)
                    images_data.extend(excel_images)
                    print(f"Loaded {len(excel_docs)} docs and {len(excel_images)} images from {file}")
                except Exception as e:
                    print(f"Error loading Excel {file_path}: {str(e)}")

        return documents, images_data

    def process_pdf(self, pdf_path):
        """Extracts all text and images from a given PDF file."""
        text_docs = PyPDFLoader(pdf_path).load()
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                for img_index, img in enumerate(doc.get_page_images(page_num)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    images.append({
                        'image_data': img_base64,
                        'source': os.path.basename(pdf_path),
                        'page': page_num + 1,
                        'description': f"Image {img_index + 1} from page {page_num + 1} of {os.path.basename(pdf_path)}"
                    })
            doc.close()
        except Exception as e:
            print(f"Could not extract images from {pdf_path}: {e}")
        return text_docs, images

    def process_excel_file(self, excel_path):
        """Extracts documents and embedded images from each row of an Excel file."""
        documents = []
        images_data = []
        try:
            wb = openpyxl.load_workbook(excel_path)
            ws = wb.active
            df = pd.read_excel(excel_path).fillna('')
            embedded_images = self._extract_images_from_excel(ws)

            for index, row in df.iterrows():
                product_name = str(row.get('Product Name', '')).strip()
                if not product_name:
                    continue

                product_text = f"Product Category: {row.get('Product Category', 'N/A')}\n" \
                               f"Product Name: {product_name}\n" \
                               f"Product Subtype: {row.get('Product subtype', 'N/A')}\n" \
                               f"Product Configuration: {row.get('Product Configuration', 'N/A')}"

                metadata = {
                    'source': os.path.basename(excel_path),
                    'row_index': index,
                    'product_name': product_name
                }
                doc = Document(page_content=product_text.strip(), metadata=metadata)
                documents.append(doc)

                if index < len(embedded_images):
                    img_b64 = embedded_images[index]
                    if img_b64:
                        images_data.append({
                            'image_data': img_b64,
                            'source': os.path.basename(excel_path),
                            'row_index': index,
                            'label': product_name,
                            'details': f"Category: {row.get('Product Category', '')} | Subtype: {row.get('Product subtype', '')}",
                            'description': f"Product image for {product_name}",
                            'product_name': product_name
                        })
        except Exception as e:
            print(f"Error processing Excel file {excel_path}: {e}")
        return documents, images_data

    def _extract_images_from_excel(self, worksheet):
        """Extracts all embedded images from an Excel worksheet in order."""
        images = []
        for img in worksheet._images:
            try:
                img_data = img._data()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                images.append(img_base64)
            except Exception as e:
                print(f"Could not extract an embedded image: {e}")
                images.append(None)
        return images

    def chunk_documents(self, documents):
        """Splits documents into smaller chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)