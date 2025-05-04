import os

import cv2
import base64
from rest_framework.response import Response
from rest_framework.views import APIView

from app.algorithm import detect_differences, pixel_pairwise, align_with_phase_correlation

class AlgorithmsGetView(APIView):
    def post(self, request):
        method = request.query_params.get("method")
        if method == "one":
            return self.method_one()
        if method == "two":
            return self.method_two()
        if method == "three":
            return self.method_three()
        return None

    def method_one(self):
        img1_path = self.request.data.get("img1_path")
        img2_path = self.request.data.get("img2_path")

        if not img1_path or not img2_path:
            return Response({"error": "Both image paths are required"}, status=400)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
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
        img1_path = self.request.data.get("img1_path")
        img2_path = self.request.data.get("img2_path")

        if not img1_path or not img2_path:
            return Response({"error": "Both image paths are required"}, status=400)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
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
        img1_path = self.request.data.get("img1_path")
        img2_path = self.request.data.get("img2_path")

        if not img1_path or not img2_path:
            return Response({"error": "Both image paths are required"}, status=400)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
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
