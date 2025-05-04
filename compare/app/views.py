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
    def post(self, request: Request):
        method = request.query_params.get("method")
        if method == "one":
            return self.method_one()
        if method == "two":
            return self.method_two()
        if method == "three":
            return self.method_three()
        return None


    def method_one(self):
        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        aligned_img, changed_area = detect_differences(img1_path, img2_path)

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


    def method_two(self):
        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        changed_area = pixel_pairwise(img1_path, img2_path)

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

    def method_three(self):
        img1_b64 = self.request.data.get("img1")
        img2_b64 = self.request.data.get("img2")

        if not img1_b64 or not img2_b64:
            return Response({"error": "Both image paths are required"}, status=400)

        img1_path = decode_and_save_image(img1_b64)
        img2_path = decode_and_save_image(img2_b64)

        if img1_path is None or img2_path is None:
            return Response({"error": "Не удалось прочитать одно из изображений"}, status=400)

        # Совмещение изображений фазовой корреляцией
        aligned, _ = align_with_phase_correlation(img1_path, img2_path)

        if aligned is None:
            return Response({"error": "Failed to calculate difference"}, status=400)

        _, buffer = cv2.imencode('.jpg', aligned)
        changed_b64 = base64.b64encode(buffer).decode("utf-8")

        return Response({
            "images": {
                "aligned": changed_b64,
            }
        })
