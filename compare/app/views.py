import asyncio
import os
import tempfile
from pathlib import Path
import pypdfium2 as pdfium
from rest_framework.request import Request

import cv2
import base64
from rest_framework.response import Response
from rest_framework.views import APIView

from app.algorithm import detect_differences, pixel_pairwise, align_with_phase_correlation

def decode_and_save_image(base64_str):
    img_data = base64.b64decode(base64_str)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(img_data)
    temp_file.flush()
    return temp_file.name # путь к файлу

class AlgorithmsPostView(APIView):
    async def post(self, request: Request):
        method = request.query_params.get("method")
        if method == "one":
            return await self.method_one(request)
        if method == "two":
            return await self.method_two(request)
        if method == "three":
            return await self.method_three(request)
        return None


    async def method_one(self, request: Request):
        img1_b64 = request.data.get("img1")
        img2_b64 = request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        aligned_img, changed_area = await asyncio.to_thread(detect_differences, img1_path, img2_path)

        if aligned_img is None:
            return Response({"error": "Недостаточно совпадений для гомографии"}, status=400)

        _, aligned_buffer = cv2.imencode('.jpg', aligned_img)
        aligned_b64 = base64.b64encode(aligned_buffer).decode("utf-8")

        _, changed_buffer = cv2.imencode('.jpg', changed_area)
        changed_b64 = base64.b64encode(changed_buffer).decode("utf-8")

        return Response({
            "images": {
                "aligned": aligned_b64,
                "changed": changed_b64,
            }
        })


    async def method_two(self, request: Request):
        img1_b64 = request.data.get("img1")
        img2_b64 = request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        changed_area = await asyncio.to_thread(pixel_pairwise, img1_path, img2_path)

        if changed_area is None:
            return Response({"error": "Failed to calculate difference"}, status=400)

        # Конвертируем изображение в base64 для отправки в ответ
        _, buffer = cv2.imencode('.jpg', changed_area)
        changed_b64 = base64.b64encode(buffer).decode("utf-8")

        return Response({
            "images": {
                "changed": changed_b64,
            }
        })

    async def method_three(self, request: Request):
        img1_b64 = request.data.get("img1")
        img2_b64 = request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        # Совмещение изображений фазовой корреляцией
        aligned, _ = await asyncio.to_thread(align_with_phase_correlation, img1_path, img2_path)

        if aligned is None:
            return Response({"error": "Failed to calculate difference"}, status=400)

        _, buffer = cv2.imencode('.jpg', aligned)
        changed_b64 = base64.b64encode(buffer).decode("utf-8")

        return Response({
            "images": {
                "aligned": changed_b64,
            }
        })


class ConvertPdfView(APIView):
    async def post(self, request: Request):
        base64_pdf = request.data.get("pdf")
        page = int(request.data.get("page", 1))

        if not base64_pdf:
            return Response({"error": "Поле 'pdf' обязательно."}, status=400)

        try:
            image = await asyncio.to_thread(convert_pdf_page, base64_pdf, page)
        except Exception as e:
            return Response({"error": f"Ошибка при обработке PDF: {str(e)}"}, status=500)

        return Response({"image": image})


def convert_pdf_page(base64_pdf, page):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(base64.b64decode(base64_pdf))
        f.flush()

    output_dir = tempfile.mkdtemp()
    convert_pdf_to_images(f.name, output_dir, page=page)
    image_file = next(Path(output_dir).glob("*.png"))
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def convert_pdf_to_images(pdf_path, output_dir, image_format="png", dpi=300, page=None):
    """
    Converts each page or a specific page of a PDF to images using pypdfium2.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Path to the folder where images will be saved.
        image_format (str): Image format (e.g., "png", "jpeg").
        dpi (int): Resolution in dots per inch.
        page (int, optional): Specific page to convert (1-based index). If None, converts all pages.
    """
    pdf = pdfium.PdfDocument(pdf_path)

    if page is not None:
        if page < 1 or page > len(pdf):
            raise ValueError(f"Page number {page} is out of range. The PDF has {len(pdf)} pages.")
        page_index = page - 1
        page_obj = pdf.get_page(page_index)
        pil_image = page_obj.render(scale=dpi / 72).to_pil()
        output_filename = f"page_{page}.{image_format}"
        output_path = os.path.join(output_dir, output_filename)
        pil_image.save(output_path)
    else:
        for page_index in range(len(pdf)):
            page_obj = pdf.get_page(page_index)
            pil_image = page_obj.render(scale=dpi / 72).to_pil()
            output_filename = f"page_{page_index + 1}.{image_format}"
            output_path = os.path.join(output_dir, output_filename)
            pil_image.save(output_path)
